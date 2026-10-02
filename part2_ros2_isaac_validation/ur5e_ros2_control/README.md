# UR5e ROS2 Control Validation

ROS2 / Isaac Sim validation package for the UR5e trajectory-tracking controllers developed in the preceding modeling and control-analysis stage.

The package does **not** redesign the controllers online. PID and LQR parameters are fixed from the validated design, while the MPC controller loads precomputed model/reference/QP data from `config/mpc_cpp_data.bin` and solves the constrained QP online with OSQP.

This package forms the runtime validation stage of the larger UR5e control project:

```text
Python modeling and controller design
        ↓
MPC data export
        ↓
C++ controller implementation
        ↓
ROS2 runtime
        ↓
Isaac Sim validation
        ↓
CSV experimental results
```

## Architecture

The validation is split into four roles so that controller execution, initialization, logging, and experiment sequencing remain separate.

### 1. Isaac initializer

`scripts/isaac_initializer.py`

- Reads `/validation/hold` from the Isaac Action Graph.
- While HOLD is true, forces the UR5e to the benchmark initial joint pose.
- Resets joint velocity to zero at every pre-physics step while HOLD is active.

Initial pose:

```text
q_start = [0, -90, 90, -90, -90, 0] deg
```

### 2. Controller node

`src/controller_node.cpp`

- Subscribes to `/joint_states`.
- Selects PID, LQR, or MPC with the `controller` launch argument.
- Uses the same 4 s quintic joint-space reference used during modeling.
- Publishes commanded joint torque on `/joint_command`.
- Publishes measured state, commanded torque, and controller compute time on `/validation/telemetry`.
- MPC is indexed by Isaac simulation time.
- Duplicate state samples are ignored.
- Missed MPC indices are skipped rather than replayed.

Available controllers:

```text
pid
lqr
mpc
```

### 3. Validation logger

`src/logger_node.cpp`

- Records only the RUNNING phase.
- Saves CSV files to a persistent user-selected directory.
- Stores measured `q`, measured `qdot`, reconstructed `q_ref/qdot_ref`, commanded torque, and controller compute time.

### 4. Validation manager

`src/validation_manager.cpp`

- Owns the experiment lifecycle.
- Verifies the initial joint pose for 25 consecutive samples.
- Releases HOLD.
- Runs the 4 s benchmark.
- Commands zero effort after the benchmark.
- Allows a 1.5 s free-fall phase.
- Shuts the validation launch down automatically.

State sequence:

```text
WAITING -> INITIALIZING -> RUNNING -> COMPLETE/FREE_FALL -> SHUTDOWN
```

## Benchmark

```text
Physics timestep = 0.002 s
Nominal rate     = 500 Hz
Duration         = 4.0 s
Nominal samples  = 2001

q_start = [ 0, -90,  90, -90, -90,  0] deg
q_goal  = [20, -60,  60, -70, -70, 20] deg
```

Torque limits:

```text
J1-J3 = ±150 Nm
J4-J6 = ±28 Nm
```

## MPC validation and practical tuning

The MPC controller was first deployed using the setting selected in the preceding modeling stage:

```text
Np    = 160
Nc    = 40
RHO_U = 10
```

This configuration performed well in the model-based simulation, but its ROS2 / Isaac Sim implementation showed substantial runtime difficulty, including long QP solve times, skipped control updates, and actuator saturation.

To investigate the implementation sensitivity, the MPC input-weight scaling and control horizon were varied while keeping the prediction horizon fixed.

Additional configurations tested:

```text
RHO_U = 120, Nc = 20
RHO_U = 120, Nc = 40
RHO_U = 10,  Nc = 20
```

Among the tested configurations, increasing the input penalty and reducing the control horizon to:

```text
Np    = 160
Nc    = 20
RHO_U = 120
```

produced a practical runtime configuration that completed the full 4 s validation trajectory while maintaining sub-millisecond-scale average controller computation time in the recorded run.

This tuning is an implementation adjustment for the ROS2 / Isaac Sim runtime rather than a redesign of the underlying MPC formulation.

`RHO_U` is the MPC input-weight scaling used when generating the QP Hessian. It is **not** the OSQP ADMM `rho` parameter.

## Experimental results

Recorded validation runs are included in:

```text
results/
```

```text
pid_validation.csv
lqr_validation.csv

mpc_rho10_nc40.csv
mpc_rho10_nc20.csv
mpc_rho120_nc40.csv
mpc_rho120_nc20.csv
```

The MPC result files document the transition from the model-selected configuration to the practical runtime setting, together with intermediate sensitivity tests.

## Isaac Sim setup

The existing Action Graph must contain a Generic ROS2 Subscriber:

```text
Message type : std_msgs/msg/Bool
Topic        : /validation/hold
Graph path   : /Graph/ROS_JointStates/ros2_subscriber
```

With Isaac Sim in Play, run `scripts/isaac_initializer.py` once from the Isaac Sim Script Editor.

The package assumes that the existing Isaac ROS graph:

- publishes `/joint_states`
- accepts effort commands through `/joint_command`
- uses the fixed UR5e joint order used in `controller_node.cpp`

Joint order:

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

## Dependencies

- Ubuntu 22.04
- ROS2 Humble
- Isaac Sim 6.1
- Eigen3
- OSQP C library and headers

## Build

```bash
cd ~/robotics_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select ur5e_ros2_control

source install/setup.bash
```

## Run

Create a persistent results directory once:

```bash
mkdir -p ~/robotics_ws/src/ur5e_ros2_control/results
```

Run MPC:

```bash
ros2 launch ur5e_ros2_control validation.launch.py \
  controller:=mpc \
  output_dir:=$HOME/robotics_ws/src/ur5e_ros2_control/results
```

Run PID:

```bash
ros2 launch ur5e_ros2_control validation.launch.py \
  controller:=pid \
  output_dir:=$HOME/robotics_ws/src/ur5e_ros2_control/results
```

Run LQR:

```bash
ros2 launch ur5e_ros2_control validation.launch.py \
  controller:=lqr \
  output_dir:=$HOME/robotics_ws/src/ur5e_ros2_control/results
```

If `output_dir` is omitted, CSV files are saved in the user's home directory.

Output filenames include the controller and timestamp, for example:

```text
ur5e_mpc_validation_20261002_132825.csv
```

## CSV columns

The validation CSV contains:

- sample index
- simulation time
- ROS timestamp
- six measured joint positions
- six measured joint velocities
- six reference joint positions
- six reference joint velocities
- six commanded joint torques
- controller compute time in milliseconds

This is intended to make the final plots and performance metrics reproducible without reconstructing the commanded trajectory in a separate analysis script.

## Notes on the MPC data file

`config/mpc_cpp_data.bin` is generated by the Python modeling/export pipeline.

The C++ runtime loads:

- equilibrium state/input
- torque limits
- QP Hessian
- state-gradient map
- time-varying reference-gradient terms
- feedforward torque trajectory
- nominal Python state/torque histories used for numerical diagnostics

The final C++ controller then solves the online constrained QP using the live Isaac Sim state.

The C++ implementation is therefore the deployment and validation layer of the controller developed and numerically validated in the preceding Python analysis stage.
