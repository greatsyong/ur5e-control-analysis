# UR5e Residual Reinforcement Learning and Payload Robustness Validation

ROS 2 / NVIDIA Isaac Sim validation package for learning-augmented Cartesian trajectory tracking of a six-degree-of-freedom UR5e manipulator.

This stage extends the preceding analytical modeling and model-based control studies by introducing a bounded reinforcement-learning residual torque to a nominal computed-torque PD controller.

The objective is to investigate whether a learned correction can improve tracking performance when the payload mass used in the controller model differs from the actual payload mass.

## Research Workflow

```text
Analytical UR5e dynamics
        ↓
Cartesian transport trajectory
        ↓
Computed-torque PD baseline
        ↓
Payload-mass mismatch definition
        ↓
Residual RL environment
        ↓
PPO / SAC / TD3 training
        ↓
Offline policy evaluation
        ↓
Policy export to NumPy
        ↓
ROS 2 / Isaac Sim PhysX validation
        ↓
Payload robustness and cross-evaluation
```

## Control Architecture

The nominal controller computes joint torque from the reference trajectory, robot dynamics, and joint tracking errors.

The reinforcement-learning policy generates an additional bounded torque correction.

The final commanded torque is:

```text
tau_command = clip(tau_G025 + delta_tau_RL, -tau_max, tau_max)
```

The nominal controller remains active throughout the experiment. Reinforcement learning does not replace the analytical dynamics model or the computed-torque feedback controller.

The selected nominal PD gains are:

```text
Kp = 256
Kd = 41.6
```

Residual torque bounds:

```text
J1: ±5.00 Nm
J2: ±5.00 Nm
J3: ±5.00 Nm
J4: ±0.50 Nm
J5: ±0.50 Nm
J6: ±0.01 Nm
```

## Trajectory and Simulation

The benchmark uses a three-second Cartesian transport trajectory with a rigidly attached payload.

The trajectory consists of two parabolic segments representing the ascent and descent phases, with a transition at approximately 1.7944 seconds.

The reference is converted into joint-space commands using inverse kinematics.

Simulation settings:

```text
Robot                 UR5e, 6 DOF
Trajectory duration   3.0 s
Reference points      121
Controller period     0.020 s
PhysX timestep        0.002 s
Training seed         42
```

The experiment focuses on payload transportation after rigid attachment. Grasping, contact acquisition, and release are outside the research scope.

## Residual RL Environment

The learning environment evaluates the closed-loop tracking behavior of the nominal controller combined with a residual policy.

The policy observation contains 49 dimensions.

The action consists of six continuous residual joint-torque commands, bounded independently for each joint.

The comparison includes three off-policy/on-policy reinforcement-learning algorithms:

- PPO
- SAC
- TD3

Each algorithm was trained for 30,000 environment steps under the model 3 kg / actual 5 kg condition.

The training outputs include selected model checkpoints, learning curves, and evaluation summaries.

## Repository Architecture

### 1. Learning and Optimization

`ur5e_payload_learning/`

Contains the residual RL training environment, training script, nominal-controller optimization, and analysis utilities.

Key files:

```text
rl/
    train.py
    ur5e_residual_env.py

optimization/
    pd_gain_search.py
    mpc_gain_search.py

analysis/
    payload_torque_difference.py
```

### 2. ROS 2 Controller Package

`ur5e_payload_control/`

Contains the runtime controllers, exported policy parameters, and Python inference implementations.

Important files:

```text
ur5e_payload_control/
    trajectory_pd.py
    trajectory_ppo.py
    trajectory_sac.py
    trajectory_td3.py

    trajectory_ppo_reverse.py
    trajectory_ppo_m3_a4.py
    trajectory_ppo_m5_a4.py

    ppo_numpy_policy.py
    sac_numpy_policy.py
    td3_numpy_policy.py

    ppo_27000_numpy.npz
    ppo_reverse_30000_numpy.npz
    sac_27000_numpy.npz
    td3_21000_numpy.npz
```

The exported NumPy policies allow ROS 2 inference without loading the full Stable-Baselines3 training framework.

### 3. Training Results

`training_results/`

Stores the selected PPO, SAC, and TD3 policy checkpoints together with learning curves and experiment summaries.

The reverse PPO training experiment is also included.

Each selected experiment contains:

```text
best_policy.zip
learning_curve.csv
summary.json
```

### 4. PhysX Results

`physx_results/`

Contains ROS 2 / Isaac Sim tracking logs for the nominal controller and learning-augmented controllers.

These files support comparisons across payload conditions and controller configurations.

## Experimental Results

### Model 3 kg — Actual 5 kg

| Controller | Cartesian RMSE |
|---|---:|
| G025 | 10.560 mm |
| PPO | 6.061 mm |
| TD3 | 6.103 mm |
| SAC | 6.935 mm |

PPO achieved approximately 42.6% lower Cartesian RMSE than the nominal G025 controller under this mismatch condition.

### Model 3 kg — Actual 3 kg

| Controller | Cartesian RMSE |
|---|---:|
| G025 | 1.195 mm |
| PPO | 5.009 mm |
| TD3 | 5.048 mm |
| SAC | 4.884 mm |

Under the matched-mass condition, the nominal controller outperformed the learned residual policies.

This establishes that the improvement under the training mismatch does not imply better performance across all payload conditions.

### Model 3 kg — Actual 1 kg

| Controller | Cartesian RMSE |
|---|---:|
| G025 | 10.920 mm |
| PPO | 15.508 mm |
| TD3 | 15.270 mm |
| SAC | 14.818 mm |

The policies trained under a heavier-than-modeled payload did not generalize favorably to the opposite mismatch direction.

## Reverse-Mismatch Training

An additional PPO policy was trained with:

```text
Controller model mass = 5 kg
Actual payload mass   = 3 kg
Training steps        = 30,000
```

PhysX validation:

| Controller | Cartesian RMSE |
|---|---:|
| G025 | 8.039 mm |
| Reverse PPO | 4.585 mm |

The reverse policy improved Cartesian tracking under its own training mismatch.

Analysis of the residual-torque directions indicates that the learned corrections depend on the direction of model error.

## Intermediate-Mass Cross-Evaluation

Both PPO policies were tested under an actual 4 kg payload.

| Policy | Model mass | Actual mass | RMSE |
|---|---:|---:|---:|
| PPO A | 3 kg | 4 kg | 1.931 mm |
| PPO B | 5 kg | 4 kg | 1.763 mm |

The experiment also examines ascending and descending motion separately.

Similar overall Cartesian RMSE values do not necessarily indicate identical vertical tracking behavior.

The direction of payload-model mismatch and the motion phase both affect the interpretation of the residual correction.

## Dependencies

```text
Ubuntu 22.04
ROS 2 Humble
NVIDIA Isaac Sim 6.1
Python
NumPy
SciPy
Stable-Baselines3
PyTorch
```

The learning environment and the ROS 2 deployment environment have different runtime requirements.

## Research Scope and Limitations

- The payload is rigidly attached to the end effector.
- The policy is trained using a numerical dynamics environment and validated in Isaac Sim PhysX.
- The experiments examine a prescribed Cartesian transportation task.
- The results are based on the recorded payload conditions and selected policies.
- Improvements under one model-mismatch condition should not be interpreted as universal robustness.
- Hardware-level validation and grasp/release behavior are not included.

## Relationship to the Full Project

This package extends the earlier stages of the UR5e research:

1. Analytical Robot Modeling and Model-Based Control
2. ROS 2 / Isaac Sim Controller Validation
3. Residual Reinforcement Learning and Payload-Mass Robustness

The detailed technical reports provide the derivations, control-design rationale, learning methodology, quantitative comparisons, and physical interpretation of the experiments.

## Author

Sooyong Kim

[Engineering Portfolio](https://sooyongtech.dev)

[Full UR5e Research Repository](https://github.com/greatsyong/ur5e-control-analysis)
