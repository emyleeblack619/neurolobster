"""
Test MuJoCo body loading — v1
Simple 2-leg placeholder before full 8-leg model
"""
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('src/simulation/lobster_v1.xml')
d = mujoco.MjData(m)

print(f"Model loaded: {m.njnt} joints, {m.nu} actuators")
print(f"Body mass: {sum(m.body_mass):.3f} kg")

# Test basic physics step
for _ in range(100):
    mujoco.mj_step(m, d)

print(f"Physics OK — torso Z after 100 steps: {d.qpos[2]:.4f}")
