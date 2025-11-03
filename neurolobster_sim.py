"""
NeuroLobster — CPG + MuJoCo
Лобстер с движением ног через CPG
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path('lobster.xml')
d = mujoco.MjData(m)

# CPG параметры
freq = 1.0   # Hz
amp  = 0.4   # радианы

# Metachronal wave — фазы для 8 ног
# L1, L2, L3, L4, R1, R2, R3, R4
phase_offsets = np.array([
    0.00, 0.25, 0.50, 0.75,   # левые
    0.50, 0.75, 0.00, 0.25,   # правые (противофаза)
]) * 2 * np.pi

hip_names  = ['a_L1_hip','a_L2_hip','a_L3_hip','a_L4_hip',
              'a_R1_hip','a_R2_hip','a_R3_hip','a_R4_hip']
knee_names = ['a_L1_knee','a_L2_knee','a_L3_knee','a_L4_knee',
              'a_R1_knee','a_R2_knee','a_R3_knee','a_R4_knee']

hip_ids  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hip_names]
knee_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in knee_names]

# Нейронная активность STG (симуляция pyloric ритма)
def neuron_activity(t):
    ab = 0.5 + 0.5 * np.sin(2*np.pi*1.3*t)
    lp = 0.5 + 0.5 * np.sin(2*np.pi*1.3*t - 2.1)
    py = 0.5 + 0.5 * np.sin(2*np.pi*1.3*t - 4.2)
    return ab, lp, py

t_start = time.time()

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.distance  = 0.6
    viewer.cam.elevation = -25
    viewer.cam.azimuth   = 135
    viewer.cam.lookat    = np.array([0.0, 0.0, 0.05])

    print("NeuroLobster запущен!")
    print("Лобстер идёт вперёд используя CPG metachronal wave")
    print("Нажми ESC для выхода\n")

    step = 0
    while viewer.is_running():
        t_sim = d.time

        # CPG управляет ногами
        omega = 2 * np.pi * freq
        for i in range(8):
            phi = omega * t_sim + phase_offsets[i]
            hip_ctrl  =  amp * np.sin(phi)
            knee_ctrl = -amp * 0.4 * np.sin(phi + 0.3)
            d.ctrl[hip_ids[i]]  = hip_ctrl
            d.ctrl[knee_ids[i]] = knee_ctrl

        mujoco.mj_step(m, d)

        if step % 5 == 0:
            ab, lp, py = neuron_activity(t_sim)
            viewer.sync()

        step += 1

        # Синхронизация с реальным временем
        t_wall = time.time() - t_start
        if d.time > t_wall:
            time.sleep(d.time - t_wall)

print("Симуляция завершена")
