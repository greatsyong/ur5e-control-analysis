UR5e G025 + TD3 | model 3 kg, physical 5 kg

FILES
- trajectory_td3.py : new ROS 2 node (preserves original trajectory_pd.py)
- td3_numpy_policy.py : numpy-only TD3 actor inference
- export_td3_policy.py : SB3 .zip -> NumPy .npz exporter (run in RL virtualenv)

STEP 1 - Copy ROS files
cp trajectory_td3.py td3_numpy_policy.py ~/robotics_ws/src/ur5e_payload_control/ur5e_payload_control/

STEP 2 - Register ROS entrypoint in ~/robotics_ws/src/ur5e_payload_control/setup.py:
    'trajectory_td3 = ur5e_payload_control.trajectory_td3:main',
Place inside entry_points['console_scripts'] (do not remove existing entries).

STEP 3 - Export model in RL .venv
cd ~/robotics_ws/src/ur5e_payload_learning
source .venv/bin/activate
python /path/to/export_td3_policy.py \
    --model /path/to/TD3_MODEL_3KG_ACTUAL5KG/best_policy.zip \
    --output ~/robotics_ws/src/ur5e_payload_learning/results/rl/td3_21000_numpy.npz

Exporter verifies NumPy actor against SB3 actor on 17 observations.

STEP 4 - Compile and run ROS 2 (deactivate RL .venv)
deactivate
source /opt/ros/humble/setup.bash
cd ~/robotics_ws
colcon build --packages-select ur5e_payload_control --symlink-install
source install/setup.bash
ros2 run ur5e_payload_control trajectory_td3

STEP 5 - Isaac Sim: physical 5 kg, initial joint pose and action graph as G025.
Start node BEFORE Play. After RETURN, call:
ros2 service call /ur5e/start_trajectory std_srvs/srv/Trigger "{}"

WARNING: Do not simultaneously run trajectory_pd and trajectory_td3. Both publish /joint_command.
SIMULATOR ACTUAL PAYLOAD MASS MUST BE 5 KG.
Use the original table visuals, pose, 3-s reference, effort limits, 2-second settle gate.
ROS timer is wall-clock 20 ms, not hard 20 ms simulation-time. Check log timing.

RESULT CSV:
~/Documents/projects/12-rl-isaac-ur5e/results/ros2_physx_rl/TD3_G025_model3kg_actual5kg_*.csv
CSV contains per-joint PD feedback, feedforward, RL residual, normalized TD3 action and actual torque.

NOTE: Pure offline RL forward dynamics ignores some PhysX effects; this is the transfer test.
