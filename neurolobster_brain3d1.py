"""
NeuroLobster — 3D Brain Point Cloud + Lobster Head Outline
"""
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time

DATA_FILE = "/tmp/neurolobster_state.npy"

np.random.seed(42)

def make_ganglion(center, n, radius):
    theta = np.random.uniform(0, 2*np.pi, n)
    phi   = np.random.uniform(0, np.pi, n)
    r     = np.random.uniform(0, radius, n)
    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)
    return x, y, z

ganglia_def = {
    'STG':   {'center': (0.0, 0.0, 0.0),    'n': 26, 'r': 0.4,  'base_color': [0.9, 0.35, 0.1]},
    'CoG_L': {'center': (-1.2, 0.5, 0.2),   'n': 40, 'r': 0.5,  'base_color': [0.7, 0.45, 0.1]},
    'CoG_R': {'center': (1.2, 0.5, 0.2),    'n': 40, 'r': 0.5,  'base_color': [0.7, 0.45, 0.1]},
    'OG':    {'center': (0.0, 0.8, 0.1),    'n': 15, 'r': 0.3,  'base_color': [0.5, 0.3, 0.6]},
    'Brain': {'center': (0.0, 2.0, 0.5),    'n': 80, 'r': 0.8,  'base_color': [0.3, 0.3, 0.7]},
    'VNC_1': {'center': (-0.3, -1.0, -0.2), 'n': 30, 'r': 0.35, 'base_color': [0.2, 0.55, 0.8]},
    'VNC_2': {'center': (0.3, -1.0, -0.2),  'n': 30, 'r': 0.35, 'base_color': [0.2, 0.55, 0.8]},
    'VNC_3': {'center': (-0.3, -1.8, -0.3), 'n': 28, 'r': 0.32, 'base_color': [0.1, 0.65, 0.4]},
    'VNC_4': {'center': (0.3, -1.8, -0.3),  'n': 28, 'r': 0.32, 'base_color': [0.1, 0.65, 0.4]},
}

all_x, all_y, all_z = [], [], []
all_colors_base = []
all_ganglion = []

for gname, gdef in ganglia_def.items():
    x, y, z = make_ganglion(gdef['center'], gdef['n'], gdef['r'])
    all_x.extend(x); all_y.extend(y); all_z.extend(z)
    all_colors_base.extend([gdef['base_color']] * gdef['n'])
    all_ganglion.extend([gname] * gdef['n'])

all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)
all_colors_base = np.array(all_colors_base)
all_ganglion = np.array(all_ganglion)
N = len(all_x)

stg_mask   = all_ganglion == 'STG'
cog_mask   = np.array([g in ('CoG_L','CoG_R') for g in all_ganglion])
vnc_mask   = np.array(['VNC' in g for g in all_ganglion])
brain_mask = all_ganglion == 'Brain'

stg_indices = np.where(stg_mask)[0]
n_stg = len(stg_indices)
ab_idx = stg_indices[:max(1, n_stg//3)]
lp_idx = stg_indices[max(1,n_stg//3):max(2,2*n_stg//3)]
py_idx = stg_indices[max(2,2*n_stg//3):]

# ── Wireframe силуэты ────────────────────────────────────
def make_ellipsoid_wireframe(cx, cy, cz, rx, ry, rz, n=20):
    """Возвращает линии wireframe эллипсоида"""
    lines = []
    # Меридианы
    for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
        t = np.linspace(0, 2*np.pi, n)
        x = cx + rx * np.cos(t) * np.cos(phi)
        y = cy + ry * np.cos(t) * np.sin(phi)
        z = cz + rz * np.sin(t)
        lines.append((x, y, z))
    # Параллели
    for t in np.linspace(-np.pi/2, np.pi/2, 6):
        phi = np.linspace(0, 2*np.pi, n)
        x = cx + rx * np.cos(t) * np.cos(phi)
        y = cy + ry * np.cos(t) * np.sin(phi)
        z = cz + rz * np.sin(t) * np.ones_like(phi)
        lines.append((x, y, z))
    return lines

# ── Фигура ───────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor='#000008')
fig.suptitle('NeuroLobster — 3D Brain Emulation\nStomatogastric Nervous System',
             color='white', fontsize=12, fontweight='bold')

ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#000008')
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('none')

# ── Рисуем wireframe силуэты ─────────────────────────────
# Голова лобстера (большой эллипсоид)
head_lines = make_ellipsoid_wireframe(0, 1.2, 0.2, 1.8, 1.6, 0.9, n=40)
wire_head = []
for (x,y,z) in head_lines:
    l, = ax.plot(x, y, z, color='#1a2a3a', lw=0.6, alpha=0.5)
    wire_head.append(l)

# Желудок/STG область (маленький эллипсоид)
stg_lines = make_ellipsoid_wireframe(0, 0.0, 0.0, 0.6, 0.5, 0.4, n=30)
wire_stg = []
for (x,y,z) in stg_lines:
    l, = ax.plot(x, y, z, color='#3a1a0a', lw=0.5, alpha=0.4)
    wire_stg.append(l)

# VNC канал (цилиндр как труба)
vnc_theta = np.linspace(0, 2*np.pi, 16)
vnc_y = np.linspace(-1.0, -2.2, 10)
for yi in vnc_y:
    xc = 0.25 * np.cos(vnc_theta)
    zc = 0.25 * np.sin(vnc_theta)
    ax.plot(xc, yi*np.ones_like(xc), zc, color='#0a1a2a', lw=0.4, alpha=0.3)

# Brain капсула
brain_lines = make_ellipsoid_wireframe(0, 2.0, 0.5, 1.0, 0.9, 0.7, n=30)
wire_brain = []
for (x,y,z) in brain_lines:
    l, = ax.plot(x, y, z, color='#0a0a3a', lw=0.5, alpha=0.4)
    wire_brain.append(l)

# ── Point cloud ───────────────────────────────────────────
colors_now = all_colors_base * 0.15
scatter = ax.scatter(all_x, all_y, all_z,
                     c=colors_now, s=2.0, alpha=0.7, depthshade=True)

# Метки
for gname, gdef in ganglia_def.items():
    cx, cy, cz = gdef['center']
    short = gname.replace('_L','').replace('_R','').replace('_','')
    ax.text(cx, cy, cz+gdef['r']+0.2, short,
            color='#334455', fontsize=6, ha='center')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.8, 3.2)
ax.set_zlim(-1.2, 1.8)

info_text = ax.text2D(0.02, 0.96, 'AB: -- Hz',
                       transform=ax.transAxes, color='white', fontsize=8)
behavior_text = ax.text2D(0.02, 0.91, '→ Вперёд',
                           transform=ax.transAxes, color='#aaaaff',
                           fontsize=10, fontweight='bold')

# Легенда
ax.text2D(0.75, 0.96, '● AB/PD', transform=ax.transAxes, color='#E85D24', fontsize=8)
ax.text2D(0.75, 0.92, '● LP',    transform=ax.transAxes, color='#3B8BD4', fontsize=8)
ax.text2D(0.75, 0.88, '● PY',    transform=ax.transAxes, color='#1D9E75', fontsize=8)
ax.text2D(0.75, 0.84, '● VNC',   transform=ax.transAxes, color='#2299DD', fontsize=8)
ax.text2D(0.75, 0.80, '● Brain', transform=ax.transAxes, color='#5555CC', fontsize=8)

behaviors = ["forward","forward","forward","turn_left","turn_right","forward","forward","backward"]
bnames = {"forward":"→ Вперёд","backward":"← Назад",
          "turn_left":"↰ Влево","turn_right":"↱ Вправо"}

plt.ion()
plt.show()

azim = 45
step = 0

while plt.get_fignums():
    try:
        d = np.load(DATA_FILE)
        flash = d[3:6]
        freq = d[6]; amp = d[7]
        ab_act = d[8]; lp_act = d[9]; py_act = d[10]
        b_idx = int(d[11])
        behavior = behaviors[b_idx % len(behaviors)]

        new_colors = all_colors_base * 0.1
        new_sizes  = np.ones(N) * 1.5

        # AB — оранжевое свечение
        ab_glow = 0.15 + 0.85 * float(flash[0])
        new_colors[ab_idx] = np.array([1.0, 0.45, 0.05]) * ab_glow
        new_sizes[ab_idx]  = 5.0 + 10.0 * float(flash[0])

        # LP — синее
        lp_glow = 0.15 + 0.85 * float(flash[1])
        new_colors[lp_idx] = np.array([0.1, 0.6, 1.0]) * lp_glow
        new_sizes[lp_idx]  = 4.0 + 8.0 * float(flash[1])

        # PY — зелёное
        py_glow = 0.15 + 0.85 * float(flash[2])
        new_colors[py_idx] = np.array([0.05, 0.9, 0.45]) * py_glow
        new_sizes[py_idx]  = 3.0 + 6.0 * float(flash[2])

        # CoG
        cog_glow = 0.08 + 0.5 * float(lp_act)
        new_colors[cog_mask] = np.array([0.8, 0.55, 0.1]) * cog_glow
        new_sizes[cog_mask]  = 1.5 + 3.0 * float(lp_act)

        # VNC — волна активации
        vnc_idx = np.where(vnc_mask)[0]
        for ii, vi in enumerate(vnc_idx):
            wave = max(0, np.sin(step*0.08 + ii*0.4)) * float(ab_act)
            new_colors[vi] = np.array([0.1, 0.6, 0.95]) * (0.08 + 0.7*wave)
            new_sizes[vi]  = 1.5 + 5.0 * wave

        # Brain
        brain_glow = 0.06 + 0.35 * max(float(ab_act), float(lp_act))
        new_colors[brain_mask] = np.array([0.3, 0.3, 0.9]) * brain_glow
        new_sizes[brain_mask]  = 1.0 + 3.0 * brain_glow

        # Wireframe STG светится при активности
        stg_alpha = 0.3 + 0.5 * float(flash[0])
        for l in wire_stg:
            l.set_alpha(stg_alpha)
            l.set_color(f'#{int(60+150*float(flash[0])):02x}1a0a')

        # Wireframe мозга
        brain_alpha = 0.2 + 0.4 * brain_glow
        for l in wire_brain:
            l.set_alpha(brain_alpha)

        scatter._offsets3d = (all_x, all_y, all_z)
        scatter.set_facecolors(np.clip(new_colors, 0, 1))
        scatter.set_sizes(new_sizes)

        info_text.set_text(f'AB: {freq:.0f} Hz  |  flash: {flash[0]:.2f}')
        behavior_text.set_text(bnames.get(behavior, '→ Вперёд'))

        # Медленное вращение
        azim = (azim + 0.25) % 360
        ax.view_init(elev=18, azim=azim)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    except:
        pass

    time.sleep(0.05)
    step += 1
