from brian2 import *
import numpy as np
prefs.codegen.target = 'numpy'

eqs = '''
dV/dt     = (gNa*m**3*h*(50*mV-V) + gK*n**4*(-77*mV-V) + gL*(-54*mV-V) + I_app - g_inh*(V+80*mV)) / Cm : volt
dm/dt     = (0.1/mV)*10*mV/exprel((-V+40*mV)/(10*mV))/ms*(1-m) - 4*exp((-V-65*mV)/(18*mV))/ms*m : 1
dh/dt     = 0.07*exp((-V-65*mV)/(20*mV))/ms*(1-h) - 1/(exp((-V-35*mV)/(10*mV))+1)/ms*h : 1
dn/dt     = (0.01/mV)*10*mV/exprel((-V+55*mV)/(10*mV))/ms*(1-n) - 0.125*exp((-V-65*mV)/(80*mV))/ms*n : 1
dg_inh/dt = -g_inh / (tau_off_val*ms) : siemens*meter**-2
gNa       : siemens*meter**-2
gK        : siemens*meter**-2
gL        : siemens*meter**-2
I_app     : amp*meter**-2
Cm        : farad*meter**-2
tau_off_val : 1
'''

def run_stg(w_fwd, w_back, I_LP, I_PY, tau_off, dur=1000):
    start_scope()
    G = NeuronGroup(3, eqs, method='exponential_euler',
                    threshold='V > -21*mV',
                    reset='V=-65*mV; m=0.05; h=0.6; n=0.32',
                    refractory=3*ms, namespace={})
    G.Cm=1*ufarad*cm**-2; G.gNa=120*msiemens*cm**-2
    G.gK=36*msiemens*cm**-2; G.gL=0.3*msiemens*cm**-2
    G.V=[-65,-62,-59]*mV; G.m=[0.05,0.08,0.11]
    G.h=[0.60,0.55,0.50]; G.n=[0.32,0.35,0.38]
    G.g_inh=0*msiemens*cm**-2
    G.tau_off_val=tau_off
    G.I_app=[14.0, I_LP, I_PY]*uamp*cm**-2
    SYN = Synapses(G, G, model='w : siemens*meter**-2', on_pre='g_inh_post += w')
    SYN.connect(i=[0,0,1,1,2], j=[1,2,2,0,0])
    SYN.w=[w_fwd, w_fwd*0.8, w_fwd*0.8, w_back, w_back*0.5]*msiemens*cm**-2
    S = SpikeMonitor(G)
    run(dur*ms)
    n = [int(len(S.t[S.i==i])) for i in range(3)]
    return n

w_fwd_vals  = np.arange(0.04, 0.22, 0.02)
w_back_vals = np.arange(0.02, 0.14, 0.02)
I_LP_vals   = np.arange(14.5, 22.0, 0.5)
I_PY_vals   = np.arange(13.0, 20.0, 0.5)
tau_vals    = [40.0, 60.0, 80.0, 120.0]

total = len(w_fwd_vals)*len(w_back_vals)*len(I_LP_vals)*len(I_PY_vals)*len(tau_vals)
print(f"Поиск: {total} комбинаций...")

results = []
count = 0
for tau in tau_vals:
    for w_fwd in w_fwd_vals:
        for w_back in w_back_vals:
            for I_LP in I_LP_vals:
                for I_PY in I_PY_vals:
                    if I_PY >= I_LP:
                        continue
                    count += 1
                    n = run_stg(w_fwd, w_back, I_LP, I_PY, tau)
                    all_fire = all(x > 5 for x in n)
                    if all_fire:
                        # Проверяем фазовый сдвиг — разные частоты
                        freqs = [n[i]/1.0 for i in range(3)]
                        phase = (freqs[0] != freqs[1]) and (freqs[1] != freqs[2])
                        tag = ' ✓ФАЗА' if phase else ''
                        score = sum(n) + (200 if phase else 0)
                        print(f"tau={tau:.0f} w_f={w_fwd:.2f} w_b={w_back:.2f} "
                              f"I_LP={I_LP:.1f} I_PY={I_PY:.1f} → {n}{tag}")
                        results.append((score, w_fwd, w_back, I_LP, I_PY, tau, n))

print(f"\nПроверено: {count} | Найдено: {len(results)}")
if results:
    results.sort(reverse=True)
    b = results[0]
    print(f"\n*** ЛУЧШИЕ ПАРАМЕТРЫ ***")
    print(f"w_fwd={b[1]:.2f}  w_back={b[2]:.2f}  I_LP={b[3]:.1f}  I_PY={b[4]:.1f}  tau={b[5]:.0f}  спайков={b[6]}")
else:
    print("Не найдено ни одной комбинации где все 3 > 5 спайков")
