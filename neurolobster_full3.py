"""
NeuroLobster v3 — STG + CPG + MuJoCo + Brain Visualization
Быстрое движение, смена направлений, 3D мозг + 2D схема + heatmap
"""
import mujoco
import mujoco.viewer
from brian2 import *
import numpy as np
import time
import threading
import subprocess

prefs.codegen.target = 'numpy'
start_scope()

# ─── STG ────────────────────────────────────────────────
eqs = '''
dV/dt     = (gNa*m**3*h*(50*mV-V) + gK*n**4*(-77*mV-V) + gL*(-54*mV-V) + I_app - g_inh*(V+80*mV)) / Cm : volt
dm/dt     = (0.1/mV)*10*mV/exprel((-V+40*mV)/(10*mV))/ms*(1-m) - 4*exp((-V-65*mV)/(18*mV))/ms*m : 1
dh/dt     = 0.07*exp((-V-65*mV)/(20*mV))/ms*(1-h) - 1/(exp((-V-35*mV)/(10*mV))+1)/ms*h : 1
dn/dt     = (0.01/mV)*10*mV/exprel((-V+55*mV)/(10*mV))/ms*(1-n) - 0.125*exp((-V-65*mV)/(80*mV))/ms*n : 1
dg_inh/dt = -g_inh / (80*ms) : siemens*meter**-2
gNa : siemens*meter**-2
gK  : siemens*meter**-2
gL  : siemens*meter**-2
I_app : amp*meter**-2
Cm  : farad*meter**-2
'''

G = NeuronGroup(3, eqs, method='exponential_euler',
                threshold='V > -21*mV',
                reset='V=-65*mV; m=0.05; h=0.6; n=0.32',
                refractory=3*ms, namespace={})

G.Cm=1*ufarad*cm**-2; G.gNa=120*msiemens*cm**-2
G.gK=36*msiemens*cm**-2; G.gL=0.3*msiemens*cm**-2
G.V=[-65,-62,-59]*mV; G.m=[0.05,0.08,0.11]
G.h=[0.60,0.55,0.50]; G.n=[0.32,0.35,0.38]
G.g_inh=0*msiemens*cm**-2
G.I_app=[14.0, 15.5, 15.2]*uamp*cm**-2

SYN = Synapses(G, G, model='w : siemens*meter**-2', on_pre='g_inh_post += w')
SYN.connect(i=[0,0,1,1,2], j=[1,2,2,0,0])
SYN.w=[0.04, 0.032, 0.032, 0.02, 0.01]*msiemens*cm**-2
S = SpikeMonitor(G)

state = {
    'v':       np.zeros(3),
    'freq':    3.0,
    'amp':     0.7,
    'flash':   np.zeros(3),
    'sensors': np.zeros(8),
    'direction': 0.0,  # угол поворота
    'behavior':  'forward',
}

DATA_FILE = '/tmp/neurolobster_state.npy'

# Смена поведения каждые ~5 секунд
behavior_timer = [0.0]
behaviors = ['forward', 'forward', 'forward', 'turn_left', 'turn_right', 'forward', 'forward', 'backward']
behavior_idx = [0]

@network_operation(dt=50*ms)
def update_state():
    v = np.array(G.V[:] / mV, dtype=float)
    state['v'] = v
    ab = float(v[0] > -30)
    lp = float(v[1] > -30)
    py = float(v[2] > -30)
    state['freq'] = 3.0 + ab * 2.0
    state['amp']  = 0.65 + ab * 0.25 + lp * 0.1
    for i in range(3):
        state['flash'][i] = 1.0 if v[i] > -22 else state['flash'][i] * 0.75

    # Смена поведения по таймеру
    behavior_timer[0] += 0.05
    if behavior_timer[0] > 4.0:
        behavior_timer[0] = 0.0
        behavior_idx[0] = (behavior_idx[0] + 1) % len(behaviors)
        state['behavior'] = behaviors[behavior_idx[0]]

    np.save(DATA_FILE, np.array([
        v[0], v[1], v[2],
        state['flash'][0], state['flash'][1], state['flash'][2],
        state['freq'], state['amp'],
        ab, lp, py,
        float(behavior_idx[0])
    ]))

    sm = float(np.mean(np.abs(state['sensors'])))
    G.I_app = [14.0+sm*0.5, 15.5+sm*0.3, 15.2+sm*0.2]*uamp*cm**-2

# ─── MUJOCO ─────────────────────────────────────────────
m_mj = mujoco.MjModel.from_xml_path('lobster.xml')
d_mj = mujoco.MjData(m_mj)

hip_names  = ['a_L1_hip','a_L2_hip','a_L3_hip','a_L4_hip',
              'a_R1_hip','a_R2_hip','a_R3_hip','a_R4_hip']
knee_names = ['a_L1_knee','a_L2_knee','a_L3_knee','a_L4_knee',
              'a_R1_knee','a_R2_knee','a_R3_knee','a_R4_knee']
hip_ids  = [mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hip_names]
knee_ids = [mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in knee_names]

# Фазы для разных направлений
phases_forward  = np.array([0.00,0.25,0.50,0.75, 0.50,0.75,0.00,0.25]) * 2*np.pi
phases_backward = np.array([0.75,0.50,0.25,0.00, 0.25,0.00,0.75,0.50]) * 2*np.pi
phases_left     = np.array([0.00,0.25,0.50,0.75, 0.00,0.25,0.50,0.75]) * 2*np.pi  # левые быстрее
phases_right    = np.array([0.50,0.75,0.00,0.25, 0.50,0.75,0.00,0.25]) * 2*np.pi

# ─── BRAIN OVERLAY ──────────────────────────────────────
overlay_code = r'''
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time

DATA_FILE = "/tmp/neurolobster_state.npy"
colors_n = ["#E85D24","#3B8BD4","#1D9E75"]
labels_n = ["AB/PD\n(pacemaker)","LP","PY"]
behaviors = ["forward","forward","forward","turn_left","turn_right","forward","forward","backward"]
behavior_names = {"forward":"→ Вперёд","backward":"← Назад","turn_left":"↰ Влево","turn_right":"↱ Вправо"}

fig = plt.figure(figsize=(16, 9), facecolor="#050510")
fig.suptitle("NeuroLobster — Lobster Brain Activity Visualization",
             color="white", fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 3, left=0.04, right=0.98, top=0.92, bottom=0.06,
                       hspace=0.35, wspace=0.3)

# ── 1. Потенциалы действия ──────────────────────────────
ax_v = fig.add_subplot(gs[0, :2])
ax_v.set_facecolor("#0a0a18")
ax_v.set_title("Потенциалы действия (mV)", color="white", fontsize=9)
ax_v.set_ylim(-85, 65)
ax_v.set_xlim(0, 300)
ax_v.tick_params(colors="#555566", labelsize=7)
for sp in ax_v.spines.values(): sp.set_edgecolor("#1a1a2e")
ax_v.axhline(-21, color="#333344", lw=0.5, ls="--")
ax_v.text(2, -18, "порог", color="#444455", fontsize=7)

v_hist = [[] for _ in range(3)]
lines_v = [ax_v.plot([], [], color=c, lw=1.0, label=l.replace("\n"," "), alpha=0.9)[0]
           for c, l in zip(colors_n, labels_n)]
ax_v.legend(loc="upper right", fontsize=7, facecolor="#0a0a18",
            labelcolor="white", framealpha=0.8)

# ── 2. Схема STG нейронов ───────────────────────────────
ax_stg = fig.add_subplot(gs[0, 2])
ax_stg.set_facecolor("#0a0a18")
ax_stg.set_title("Схема STG", color="white", fontsize=9)
ax_stg.set_xlim(-2, 2); ax_stg.set_ylim(-2, 2)
ax_stg.set_aspect("equal"); ax_stg.axis("off")

pos_n = {"AB": (0.0, 1.1), "LP": (-1.0, -0.6), "PY": (1.0, -0.6)}
circles_stg = {}
for name, (x, y) in pos_n.items():
    i = ["AB","LP","PY"].index(name)
    c = Circle((x,y), 0.42, color=colors_n[i], alpha=0.25, zorder=2)
    ax_stg.add_patch(c)
    circles_stg[name] = c
    ax_stg.text(x, y+0.02, name, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=3)

# Стрелки с метками
arrow_props = dict(arrowstyle="-|>", color="#334433", lw=1.2)
connections = [("AB","LP","-"), ("AB","PY","-"), ("LP","PY","-"),
               ("LP","AB","-"), ("PY","AB","-")]
for src, dst, typ in connections:
    xs,ys = pos_n[src]; xd,yd = pos_n[dst]
    dx,dy = xd-xs, yd-ys
    l = np.sqrt(dx**2+dy**2)
    ax_stg.annotate("", xy=(xd-dx/l*0.45, yd-dy/l*0.45),
                    xytext=(xs+dx/l*0.45, ys+dy/l*0.45),
                    arrowprops=dict(arrowstyle="-|>",color="#334455",lw=1.2))

info_stg = ax_stg.text(0,-1.7,"CPG: 3.0 Hz",ha="center",
                        color="#888899",fontsize=8)
behavior_txt = ax_stg.text(0, 1.75, "→ Вперёд", ha="center",
                            color="#aaaacc", fontsize=9, fontweight="bold")

# ── 3. Анатомия мозга лобстера (схема) ─────────────────
ax_brain = fig.add_subplot(gs[1, :2])
ax_brain.set_facecolor("#050510")
ax_brain.set_title("Нервная система лобстера — активные зоны", color="white", fontsize=9)
ax_brain.set_xlim(0, 14); ax_brain.set_ylim(-1, 4)
ax_brain.axis("off")

# Тело лобстера (силуэт)
body = Ellipse((7, 1.5), 12, 2.2, color="#1a1a2e", zorder=1)
ax_brain.add_patch(body)
ax_brain.text(7, 3.5, "← Хвост                              Голова →",
              ha="center", color="#333355", fontsize=7)

# Ганглии нервной системы
ganglia = {
    "STG\n(Желудок)":  (2.0, 2.8, 0.55, "#E85D24", "STG"),
    "CoG\n(Commissural)": (4.0, 2.8, 0.45, "#BA7517", "CoG"),
    "OG\n(Esophageal)":   (5.5, 2.8, 0.38, "#BA7517", "OG"),
    "Brain\n(Cerebral)":  (7.5, 2.8, 0.52, "#534AB7", "Brain"),
    "VNC L1": (9.5, 2.2, 0.32, "#3B8BD4", "VNC1"),
    "VNC L2": (10.3, 1.8, 0.30, "#3B8BD4", "VNC2"),
    "VNC L3": (11.1, 1.4, 0.28, "#1D9E75", "VNC3"),
    "VNC L4": (11.9, 1.0, 0.27, "#1D9E75", "VNC4"),
}

ganglion_circles = {}
for label, (x, y, r, col, key) in ganglia.items():
    c = Circle((x,y), r, color=col, alpha=0.2, zorder=3)
    ax_brain.add_patch(c)
    ganglion_circles[key] = c
    ax_brain.text(x, y, label.split("\n")[0], ha="center", va="center",
                  color="white", fontsize=6.5, fontweight="bold", zorder=4)

# Нервные пути
nerve_x = [2.0, 4.0, 5.5, 7.5, 9.5, 10.3, 11.1, 11.9]
nerve_y = [2.8, 2.8, 2.8, 2.8, 2.2, 1.8, 1.4, 1.0]
ax_brain.plot(nerve_x, nerve_y, color="#222244", lw=2, zorder=2)

# Ноги (снизу)
leg_pos_x = [9.5, 10.3, 11.1, 11.9, 9.5, 10.3, 11.1, 11.9]
leg_labels = ["L1","L2","L3","L4","R1","R2","R3","R4"]
leg_circles_vis = []
for i, (lx, ll) in enumerate(zip(leg_pos_x, leg_labels)):
    ly = 0.1 if i < 4 else -0.4
    c = Circle((lx, ly), 0.18, color="#224422", alpha=0.5, zorder=3)
    ax_brain.add_patch(c)
    leg_circles_vis.append(c)
    ax_brain.text(lx, ly, ll, ha="center", va="center",
                  color="#aaffaa", fontsize=6, zorder=4)

# ── 4. Heatmap активности ───────────────────────────────
ax_heat = fig.add_subplot(gs[1, 2])
ax_heat.set_facecolor("#050510")
ax_heat.set_title("Heatmap активности", color="white", fontsize=9)

heat_data = np.zeros((8, 50))
heat_img = ax_heat.imshow(heat_data, aspect="auto", cmap="hot",
                           vmin=0, vmax=1, interpolation="nearest")
ax_heat.set_yticks(range(8))
ax_heat.set_yticklabels(["L1","L2","L3","L4","R1","R2","R3","R4"],
                         color="white", fontsize=7)
ax_heat.set_xticks([])
ax_heat.set_xlabel("← Время", color="#555566", fontsize=7)
plt.colorbar(heat_img, ax=ax_heat).ax.tick_params(labelcolor="white", labelsize=6)

plt.ion()
plt.show()

step = 0
while plt.get_fignums():
    try:
        d = np.load(DATA_FILE)
        v = d[:3]; flash = d[3:6]
        freq=d[6]; amp=d[7]
        ab=d[8]; lp=d[9]; py=d[10]
        b_idx = int(d[11])
        behavior = behaviors[b_idx % len(behaviors)]

        # Обновляем историю потенциалов
        for i in range(3):
            v_hist[i].append(float(v[i]))
            if len(v_hist[i]) > 300: v_hist[i].pop(0)
            lines_v[i].set_data(range(len(v_hist[i])), v_hist[i])
        ax_v.set_xlim(0, max(300, len(v_hist[0])))

        # Обновляем кружки STG
        for i, name in enumerate(["AB","LP","PY"]):
            alpha = 0.15 + 0.85 * float(flash[i])
            circles_stg[name].set_alpha(alpha)
            r = 0.42 + 0.15 * float(flash[i])
            circles_stg[name].set_radius(r)
        info_stg.set_text(f"CPG: {freq:.1f} Hz  |  Amp: {amp:.2f}")
        behavior_names_dict = {"forward":"→ Вперёд","backward":"← Назад",
                                "turn_left":"↰ Влево","turn_right":"↱ Вправо"}
        behavior_txt.set_text(behavior_names_dict.get(behavior, "→ Вперёд"))

        # Обновляем анатомию мозга
        ganglion_circles["STG"].set_alpha(0.1 + 0.7 * float(flash[0]))
        ganglion_circles["VNC1"].set_alpha(0.1 + 0.6 * float(ab))
        ganglion_circles["VNC2"].set_alpha(0.1 + 0.6 * float(lp))
        ganglion_circles["VNC3"].set_alpha(0.1 + 0.5 * float(py))
        ganglion_circles["VNC4"].set_alpha(0.1 + 0.4 * float(ab))
        ganglion_circles["Brain"].set_alpha(0.15 + 0.5 * max(float(ab),float(lp)))
        ganglion_circles["CoG"].set_alpha(0.1 + 0.6 * float(lp))

        # Активность ног
        phase_offset = [0.0,0.25,0.5,0.75,0.5,0.75,0.0,0.25]
        t_now = step * 0.05
        for i in range(8):
            phi = 2*np.pi*freq*t_now + phase_offset[i]*2*np.pi
            leg_act = max(0, np.sin(phi))
            leg_circles_vis[i].set_alpha(0.2 + 0.7 * leg_act)

        # Heatmap
        new_col = np.zeros(8)
        for i in range(8):
            phi = 2*np.pi*freq*t_now + phase_offset[i]*2*np.pi
            new_col[i] = max(0, np.sin(phi))
        heat_data = np.roll(heat_data, -1, axis=1)
        heat_data[:, -1] = new_col
        heat_img.set_data(heat_data)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception as e:
        pass
    time.sleep(0.05)
    step += 1
'''

with open('/tmp/neurolobster_overlay.py', 'w') as f:
    f.write(overlay_code)

overlay_proc = subprocess.Popen(['python3', '/tmp/neurolobster_overlay.py'])
brain3d_proc = subprocess.Popen(['python3', '/tmp/neurolobster_brain3d.py'])

def run_stg():
    run(60*second)
stg_thread = threading.Thread(target=run_stg, daemon=True)
stg_thread.start()

t_start = time.time()
print("="*50)
print("NeuroLobster v3 — Full Brain Visualization")
print("="*50)

with mujoco.viewer.launch_passive(m_mj, d_mj) as viewer:
    viewer.cam.distance=0.5; viewer.cam.elevation=-30
    viewer.cam.azimuth=135
    viewer.cam.lookat=np.array([0.0,0.0,0.05])
    print("Запущено! Два окна: MuJoCo + Brain Activity\n")

    frame=0
    while viewer.is_running():
        t_sim = d_mj.time
        behavior = state['behavior']
        freq = state['freq']
        amp  = state['amp']
        omega = 2*np.pi*freq

        # Выбираем фазы по поведению
        if behavior == 'forward':
            phases = phases_forward
        elif behavior == 'backward':
            phases = phases_backward
        elif behavior == 'turn_left':
            phases = phases_left
        else:
            phases = phases_right

        for i in range(8):
            phi = omega*t_sim + phases[i]
            amp_i = amp
            # При повороте — усиливаем одну сторону
            if behavior == 'turn_left' and i >= 4:
                amp_i *= 1.6
            elif behavior == 'turn_right' and i < 4:
                amp_i *= 1.6
            d_mj.ctrl[hip_ids[i]]  =  amp_i * np.sin(phi)
            d_mj.ctrl[knee_ids[i]] = -amp_i * 0.4 * np.sin(phi+0.3)

        for i in range(min(8, m_mj.nsensor)):
            state['sensors'][i] = float(d_mj.sensordata[i])

        mujoco.mj_step(m_mj, d_mj)
        if frame%10==0: viewer.sync()
        frame+=1

        t_wall = time.time()-t_start
        if d_mj.time > t_wall: time.sleep(d_mj.time-t_wall)

overlay_proc.terminate()
brain3d_proc.terminate()
n = [len(S.t[S.i==i]) for i in range(3)]
print(f"\nAB:{n[0]} LP:{n[1]} PY:{n[2]} спайков")
