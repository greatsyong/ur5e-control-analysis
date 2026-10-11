# UR5e SAC PhysX deployment (Korean instructions)

- SAC best checkpoint from `best_policy(2).zip` (27,000 steps, trained model3->actual5); **no retraining**.
- G025 Kp=256, Kd=41.6, model 3kg, 3s trajectory, residual action limits unchanged.
- SB3 SAC deterministic actor: 49 input, ReLU 256->256, tanh(mean), 6 outputs.
- Actual mass selectable 5,3,1 kg. Existing 2s start dwell retained, regardless of J5 stability.
- NumPy actor match to PyTorch 200 samples: max abs difference 1.64e-06.

### Install

Download zip to `~/Downloads` and extract the **three executable files** to the ROS2 Python package:

```bash
cd ~/robotics_ws
unzip -o ~/Downloads/UR5e_SAC_PhysX_Deployment_20261011.zip -d src/ur5e_payload_control/ur5e_payload_control/
gedit ~/robotics_ws/src/ur5e_payload_control/setup.py
```

Add ONE entry to `console_scripts` (keep prior entries):

```python
    'trajectory_sac = ur5e_payload_control.trajectory_sac:main',
```

Build and source:

```bash
cd ~/robotics_ws
python3 -m py_compile src/ur5e_payload_control/ur5e_payload_control/trajectory_sac.py
colcon build --packages-select ur5e_payload_control --symlink-install
source install/setup.bash
ros2 pkg executables ur5e_payload_control
```

### Execute (5 -> 3 -> 1 kg)

Stop Isaac, set payload mass exactly equal to runtime `actual_mass`, and stop previous ROS controller. Start one of:

```bash
ros2 run ur5e_payload_control trajectory_sac --ros-args -p actual_mass:=5.0
ros2 run ur5e_payload_control trajectory_sac --ros-args -p actual_mass:=3.0
ros2 run ur5e_payload_control trajectory_sac --ros-args -p actual_mass:=1.0
```

Then Isaac Play, wait for RETURN, call start in a second terminal:

```bash
ros2 service call /ur5e/start_trajectory std_srvs/srv/Trigger "{}"
```

CSV output at `~/Documents/projects/12-rl-isaac-ur5e/results/ros2_physx_rl/` with `SAC_G025_model3kg_actualXkg_...csv` prefix.

DO NOT use on physical robot without safety engineering and separate validation. Input observation and controller logic copied from the PPO deployment; only policy inference type and files changed.
