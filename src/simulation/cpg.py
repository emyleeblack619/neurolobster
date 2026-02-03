"""
Central Pattern Generator — metachronal wave locomotion
8-legged CPG with biologically validated phase offsets
Based on Ayers Lab lobster locomotion data
"""
import numpy as np

# Metachronal wave phase offsets (Ayers & Davis 1977)
# L1-L4: left legs anterior to posterior
# R1-R4: right legs, antiphase to left
PHASE_OFFSETS = np.array([
    0.00, 0.25, 0.50, 0.75,   # Left L1-L4
    0.50, 0.75, 0.00, 0.25,   # Right R1-R4
]) * 2 * np.pi

PHASE_BACKWARD = np.array([
    0.75, 0.50, 0.25, 0.00,
    0.25, 0.00, 0.75, 0.50,
]) * 2 * np.pi

def cpg_step(t, freq=2.0, amp=0.6, gait='forward'):
    """
    Compute CPG control signals for all 8 legs.
    Returns hip and knee torques.
    """
    phases = PHASE_OFFSETS if gait != 'backward' else PHASE_BACKWARD
    omega = 2 * np.pi * freq
    hip_ctrl  = np.zeros(8)
    knee_ctrl = np.zeros(8)

    for i in range(8):
        phi = omega * t + phases[i]
        # Asymmetric for turns
        amp_i = amp
        if gait == 'turn_left' and i >= 4:
            amp_i *= 1.6
        elif gait == 'turn_right' and i < 4:
            amp_i *= 1.6
        hip_ctrl[i]  =  amp_i * np.sin(phi)
        knee_ctrl[i] = -amp_i * 0.4 * np.sin(phi + 0.3)

    return hip_ctrl, knee_ctrl


if __name__ == '__main__':
    # Test: print one cycle
    for t in np.linspace(0, 1.0, 20):
        hip, knee = cpg_step(t)
        print(f"t={t:.2f} | hip={hip.round(2)}")
