"""
STG Pyloric Circuit — synaptic inhibition
Adding conductance-based inhibitory synapses between neurons
Target: reproduce AB->LP->PY->AB triphasic rhythm
Parameters from Marder & Bucher (2007)
"""
from brian2 import *
import numpy as np

prefs.codegen.target = 'numpy'
start_scope()

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

# Inhibitory synapses: AB->LP, AB->PY, LP->PY, LP->AB, PY->AB
SYN = Synapses(G, G, model='w : siemens*meter**-2', on_pre='g_inh_post += w')
SYN.connect(i=[0,0,1,1,2], j=[1,2,2,0,0])
SYN.w = [0.04, 0.032, 0.032, 0.02, 0.01]*msiemens*cm**-2

S = SpikeMonitor(G)
M = StateMonitor(G, 'V', record=True, dt=0.1*ms)
run(2000*ms)

for i,name in enumerate(['AB/PD','LP','PY']):
    sp = S.t[S.i==i]
    if len(sp)>1:
        freq = 1000/float(np.mean(np.diff(sp/ms)))
        print(f"{name}: {len(sp)} spikes, {freq:.1f} Hz")
