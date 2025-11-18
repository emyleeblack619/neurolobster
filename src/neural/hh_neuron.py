"""
Single Hodgkin-Huxley neuron — prototype
Testing basic spike generation before building STG circuit
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dt = 0.01   # ms
T  = 100.0  # ms
t  = np.arange(0, T, dt)

# HH parameters
gNa, gK, gL = 120.0, 36.0, 0.3
ENa, EK, EL = 50.0, -77.0, -54.0
Cm = 1.0

def alpha_m(V): return 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if abs(V+40)>1e-7 else 1.0
def beta_m(V):  return 4*np.exp(-(V+65)/18)
def alpha_h(V): return 0.07*np.exp(-(V+65)/20)
def beta_h(V):  return 1/(np.exp(-(V+35)/10)+1)
def alpha_n(V): return 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if abs(V+55)>1e-7 else 0.1
def beta_n(V):  return 0.125*np.exp(-(V+65)/80)

V = -65.0; m = 0.05; h = 0.6; n = 0.32
V_rec = []

for ti in t:
    I_app = 10.0 if 20 < ti < 80 else 0.0
    INa = gNa*m**3*h*(V-ENa)
    IK  = gK*n**4*(V-EK)
    IL  = gL*(V-EL)
    V += dt*(-INa-IK-IL+I_app)/Cm
    m += dt*(alpha_m(V)*(1-m)-beta_m(V)*m)
    h += dt*(alpha_h(V)*(1-h)-beta_h(V)*h)
    n += dt*(alpha_n(V)*(1-n)-beta_n(V)*n)
    V_rec.append(V)

print(f"Max V: {max(V_rec):.1f} mV")
print(f"Spikes detected: {sum(1 for i in range(1,len(V_rec)) if V_rec[i]>0 and V_rec[i-1]<=0)}")
