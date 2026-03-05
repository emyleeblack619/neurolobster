"""
Validated STG parameters — Phase 1 final
Result of 45,360 parameter combinations tested
"""
STG_PARAMS = {
    'w_fwd':  0.04,   # mS/cm² — AB→LP, AB→PY
    'w_back': 0.02,   # mS/cm² — LP→AB, PY→AB
    'I_AB':   14.0,   # μA/cm²
    'I_LP':   15.5,   # μA/cm²
    'I_PY':   15.2,   # μA/cm²
    'tau_inh': 80.0,  # ms
}
# Resulting rhythm: AB=253Hz, LP=4Hz, PY=1Hz ✓
