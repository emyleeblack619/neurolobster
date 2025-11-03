"""
NeuroLobster v2 — STG + CPG + MuJoCo + Neural Overlay
"""
import mujoco
import mujoco.viewer
from brian2 import *
import numpy as np
import time
import threading
import subprocess
import os

prefs.codegen.target = 'numpy'
start_scope()

# ─── STG НЕЙРОНЫ ───────────────────────────────────────
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
    'freq':    2.0,
    'amp':     0.6,
    'sensors': np.zeros(8),
    'flash':   np.zeros(3),
}

# Файл для передачи данных в overlay процесс
DATA_FILE = '/tmp/neurolobster_state.npy'

@network_operation(dt=50*ms)
def update_state():
    v = np.array(G.V[:] / mV, dtype=float)
    state['v'] = v
    ab = float(v[0] > -30)
    lp = float(v[1] > -30)
    state['freq'] = 2.0 + ab * 1.5
    state['amp']  = 0.5 + ab * 0.2 + lp * 0.1
    for i in range(3):
        state['flash'][i] = 1.0 if v[i] > -22 else state['flash'][i] * 0.8
    # Сохраняем для overlay
    np.save(DATA_FILE, np.array([v[0], v[1], v[2],
                                  state['flash'][0], state['flash'][1], state['flash'][2],
                                  state['freq'], state['amp']]))
    # Сенсоры → I_app
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
sensor_ids = list(range(m_mj.nsensor))

phase_offsets = np.array([0.00,0.25,0.50,0.75,0.50,0.75,0.00,0.25]) * 2*np.pi

# Запускаем overlay в отдельном процессе (обычный python)
overlay_code = '''
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import numpy as np
import time

DATA_FILE = "/tmp/neurolobster_state.npy"
colors = ["#E85D24","#3B8BD4","#1D9E75"]
labels = ["AB/PD","LP","PY"]

fig = plt.figure(figsize=(8,4), facecolor="#0a0a0f")
fig.suptitle("NeuroLobster — STG Neural Activity", color="white", fontsize=11)
gs = gridspec.GridSpec(1,2,left=0.05,right=0.98,top=0.85,bottom=0.15)

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#0d0d18")
ax1.set_title("Потенциал действия", color="white", fontsize=9)
ax1.set_ylim(-85,65); ax1.set_xlim(0,200)
ax1.tick_params(colors="#555566",labelsize=7)
for sp in ax1.spines.values(): sp.set_edgecolor("#222233")

v_hist = [[] for _ in range(3)]
lines = [ax1.plot([],[],color=c,lw=1.2,label=l)[0] for c,l in zip(colors,labels)]
ax1.legend(loc="upper right",fontsize=7,facecolor="#0d0d18",labelcolor="white")

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#0d0d18")
ax2.set_title("Активность", color="white", fontsize=9)
ax2.set_xlim(-1.5,1.5); ax2.set_ylim(-1.5,1.5)
ax2.set_aspect("equal"); ax2.axis("off")

pos = [(-0.8,0.5),(0.0,-0.5),(0.8,0.5)]
circles = []
for i,(x,y) in enumerate(pos):
    c = Circle((x,y),0.35,color=colors[i],alpha=0.3)
    ax2.add_patch(c); circles.append(c)
    ax2.text(x,y,labels[i],ha="center",va="center",color="white",fontsize=9,fontweight="bold")

ax2.annotate("",xy=pos[1],xytext=pos[0],arrowprops=dict(arrowstyle="->",color="#554455",lw=1.5))
ax2.annotate("",xy=pos[2],xytext=pos[1],arrowprops=dict(arrowstyle="->",color="#445544",lw=1.5))
ax2.annotate("",xy=pos[0],xytext=pos[2],arrowprops=dict(arrowstyle="->",color="#445544",lw=1.5))
info = ax2.text(0,-1.2,"CPG: 2.0 Hz",ha="center",color="#888899",fontsize=8)

plt.ion(); plt.show()
step = 0
while plt.get_fignums():
    try:
        d = np.load(DATA_FILE)
        v = d[:3]; flash = d[3:6]; freq=d[6]; amp=d[7]
        for i in range(3):
            v_hist[i].append(float(v[i]))
            if len(v_hist[i])>200: v_hist[i].pop(0)
            lines[i].set_data(range(len(v_hist[i])),v_hist[i])
            circles[i].set_alpha(0.2+0.8*float(flash[i]))
        ax1.set_xlim(0,max(200,len(v_hist[0])))
        info.set_text(f"CPG: {freq:.1f} Hz  Amp: {amp:.2f}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except: pass
    time.sleep(0.05)
    step+=1
'''

with open('/tmp/neurolobster_overlay.py', 'w') as f:
    f.write(overlay_code)

overlay_proc = subprocess.Popen(['python3', '/tmp/neurolobster_overlay.py'])

# STG в потоке
def run_stg():
    run(30*second)
stg_thread = threading.Thread(target=run_stg, daemon=True)
stg_thread.start()

t_start = time.time()
print("="*50)
print("NeuroLobster — Brain→Body + Neural Overlay")
print("="*50)

with mujoco.viewer.launch_passive(m_mj, d_mj) as viewer:
    viewer.cam.distance=0.5; viewer.cam.elevation=-30
    viewer.cam.azimuth=135
    viewer.cam.lookat=np.array([0.0,0.0,0.05])
    print("Запущено! Откройте окно matplotlib для нейронов.\n")

    frame=0
    while viewer.is_running():
        t_sim = d_mj.time
        freq = state['freq']; amp = state['amp']
        omega = 2*np.pi*freq

        for i in range(8):
            phi = omega*t_sim + phase_offsets[i]
            d_mj.ctrl[hip_ids[i]]  =  amp*np.sin(phi)
            d_mj.ctrl[knee_ids[i]] = -amp*0.4*np.sin(phi+0.3)

        for i in range(min(8, m_mj.nsensor)):
            state['sensors'][i] = float(d_mj.sensordata[i])

        mujoco.mj_step(m_mj, d_mj)
        if frame%10==0: viewer.sync()
        frame+=1

        t_wall = time.time()-t_start
        if d_mj.time > t_wall: time.sleep(d_mj.time-t_wall)

overlay_proc.terminate()
n = [len(S.t[S.i==i]) for i in range(3)]
print(f"\nAB:{n[0]} LP:{n[1]} PY:{n[2]} спайков")
