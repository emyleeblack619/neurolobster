"""
NeuroLobster — STG Pyloric Rhythm
Найденные параметры: w_fwd=0.04, w_back=0.02, I_LP=15.5, I_PY=14.0, tau=40
AB=125 Hz, LP=5 Hz, PY=1 Hz — трёхфазный ритм
"""
from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess

prefs.codegen.target = 'numpy'
start_scope()

eqs = '''
dV/dt     = (gNa*m**3*h*(50*mV-V) + gK*n**4*(-77*mV-V) + gL*(-54*mV-V) + I_app - g_inh*(V+80*mV)) / Cm : volt
dm/dt     = (0.1/mV)*10*mV/exprel((-V+40*mV)/(10*mV))/ms*(1-m) - 4*exp((-V-65*mV)/(18*mV))/ms*m : 1
dh/dt     = 0.07*exp((-V-65*mV)/(20*mV))/ms*(1-h) - 1/(exp((-V-35*mV)/(10*mV))+1)/ms*h : 1
dn/dt     = (0.01/mV)*10*mV/exprel((-V+55*mV)/(10*mV))/ms*(1-n) - 0.125*exp((-V-65*mV)/(80*mV))/ms*n : 1
dg_inh/dt = -g_inh / (80*ms) : siemens*meter**-2
gNa       : siemens*meter**-2
gK        : siemens*meter**-2
gL        : siemens*meter**-2
I_app     : amp*meter**-2
Cm        : farad*meter**-2
'''

G = NeuronGroup(3, eqs, method='exponential_euler',
                threshold='V > -21*mV',
                reset='V=-65*mV; m=0.05; h=0.6; n=0.32',
                refractory=3*ms, namespace={})

G.Cm    = 1*ufarad*cm**-2
G.gNa   = 120*msiemens*cm**-2
G.gK    = 36*msiemens*cm**-2
G.gL    = 0.3*msiemens*cm**-2
G.V     = [-65, -62, -59]*mV
G.m     = [0.05, 0.08, 0.11]
G.h     = [0.60, 0.55, 0.50]
G.n     = [0.32, 0.35, 0.38]
G.g_inh = 0*msiemens*cm**-2

# Найденные параметры
G.I_app = [14.0, 15.5, 15.2]*uamp*cm**-2

SYN = Synapses(G, G, model='w : siemens*meter**-2', on_pre='g_inh_post += w')
SYN.connect(i=[0, 0, 1, 1, 2], j=[1, 2, 2, 0, 0])
SYN.w = [0.04, 0.04*0.8, 0.04*0.8, 0.02, 0.02*0.5]*msiemens*cm**-2

M = StateMonitor(G, 'V', record=True, dt=0.1*ms)
S = SpikeMonitor(G)

print("Запуск STG pyloric rhythm...")
print("Параметры: w_fwd=0.04, I_LP=15.5, I_PY=14.0, tau=40ms")
run(2000*ms, report='text')

names  = ['AB/PD (pacemaker)', 'LP', 'PY']
colors = ['#E85D24', '#3B8BD4', '#1D9E75']

print("\n─── Результаты ───")
for i, name in enumerate(names):
    sp = S.t[S.i == i]
    if len(sp) > 1:
        freq = 1000.0 / float(np.mean(np.diff(sp/ms)))
        print(f"{name}: {len(sp)} спайков, {freq:.1f} Hz")
    else:
        print(f"{name}: {len(sp)} спайков")

# График
fig = plt.figure(figsize=(14, 9), facecolor='#0a0a0f')
fig.suptitle('NeuroLobster — STG Pyloric Rhythm\nAB/PD → LP → PY triphasic oscillation',
             color='white', fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 1, hspace=0.08, top=0.92, bottom=0.08, left=0.12, right=0.95)

t    = M.t / ms
t0   = 100
mask = t > t0
labs = ['AB/PD\n(pacemaker)', 'LP', 'PY']

for i in range(3):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor('#0d0d18')
    ax.plot(t[mask], M.V[i][mask]/mV, color=colors[i], lw=0.9)
    sp = S.t[S.i==i]/ms
    for sv in sp[sp > t0]:
        ax.axvline(sv, color=colors[i], alpha=0.2, lw=0.5)
    ax.axhline(-54, color='white', alpha=0.07, lw=0.5, ls='--')
    ax.set_ylabel(labs[i], color=colors[i], fontsize=9,
                  fontweight='bold', rotation=0, labelpad=60, va='center')
    ax.set_ylim(-85, 65)
    ax.set_xlim(t0, 2000)
    ax.tick_params(colors='#666677', labelsize=8)
    for sp2 in ax.spines.values():
        sp2.set_edgecolor('#333344')
    if i < 2:
        ax.set_xticklabels([])

ax4 = fig.add_subplot(gs[3])
ax4.set_facecolor('#0d0d18')
for i in range(3):
    sp = S.t[S.i==i]/ms
    for sv in sp[sp > t0]:
        ax4.barh(i, 6, left=sv-3, height=0.5, color=colors[i], alpha=0.85)
ax4.set_yticks([0,1,2])
ax4.set_yticklabels(['AB','LP','PY'], color='white', fontsize=8)
ax4.set_xlim(t0, 2000)
ax4.set_xlabel('Время (мс)', color='#888899', fontsize=9)
ax4.tick_params(colors='#666677', labelsize=8)
ax4.set_ylabel('Фазы', color='#888899', fontsize=8, rotation=0, labelpad=45, va='center')
for sp2 in ax4.spines.values():
    sp2.set_edgecolor('#333344')

plt.savefig('stg_pyloric_rhythm.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print("\nГрафик сохранён: stg_pyloric_rhythm.png")
subprocess.run(['open', 'stg_pyloric_rhythm.png'])
