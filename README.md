# UR5e Heavy-Payload Transport — Modeling, Control, and Residual Reinforcement Learning

A reproducible six-degree-of-freedom UR5e robotic manipulation study integrating analytical dynamics, model-based control, reinforcement learning, and ROS 2–NVIDIA Isaac Sim PhysX validation.

The research investigates Cartesian trajectory tracking during rigid-payload transportation under uncertainty in the payload mass assumed by the controller.

## Research Structure

### Part 1 — Robot Modeling and Model-Based Control

- UR5e forward/inverse kinematics and Jacobian analysis
- Rigid-body dynamics and payload modeling
- Cartesian transport trajectory planning and time parameterization
- Python/C++ numerical cross-validation
- Computed-torque PD, PID, LQR, and MPC investigations
- PD/MPC controller screening and nominal PhysX validation

A computed-torque PD controller, **G025 (Kp = 256, Kd = 41.6)**, was selected as the nominal baseline following physical-simulator validation.

### Part 2 — Residual Reinforcement Learning

A bounded residual torque policy is added to the nominal controller:

τ_cmd = clip(τ_G025 + Δτ_RL, −τ_max, τ_max)

- PPO, SAC, and TD3 continuous-action policy comparison
- 49-dimensional observations and six-dimensional torque actions
- 30,000 training interactions per algorithm, seed 42
- Deterministic policy evaluation and checkpoint selection
- ROS 2–Isaac Sim PhysX controller validation

Residual torque limits:

`[5, 5, 5, 0.5, 0.5, 0.01] Nm`

### Part 3 — Payload-Mass Mismatch and Cross-Evaluation

The study evaluates nominal-controller model mass against different actual payload masses.

- PPO trained with model 3 kg / actual 5 kg
- Reverse PPO trained with model 5 kg / actual 3 kg
- Intermediate actual 4 kg cross-evaluation of both policies
- Signed vertical tracking error and residual-torque analysis
- Comparison of ascending and descending transport segments

These experiments examine how a controller trained for one direction of payload-model mismatch responds under different loading conditions.

## Quantitative Results

### ROS 2–PhysX Cartesian RMSE

Nominal controller model mass: 3 kg.

| Actual payload | G025 | PPO | TD3 | SAC |
|---|---:|---:|---:|---:|
| 1 kg | 10.920 mm | 15.508 mm | 15.270 mm | 14.818 mm |
| 3 kg | 1.195 mm | 5.009 mm | 5.048 mm | 4.884 mm |
| 5 kg | 10.560 mm | 6.061 mm | 6.103 mm | 6.935 mm |

For the model 3 kg / actual 5 kg condition, PPO reduced Cartesian RMSE by approximately 42.6% compared with G025.

### Bidirectional PPO Validation

| Nominal model | Actual payload | G025 RMSE | PPO RMSE |
|---|---:|---:|---:|
| 3 kg | 5 kg | 10.560 mm | 6.061 mm |
| 5 kg | 3 kg | 8.039 mm | 4.585 mm |

### Intermediate-Mass Cross-Evaluation

Both PPO policies were evaluated under an actual 4 kg payload.

| Policy | Assumed model mass | Cartesian RMSE |
|---|---:|---:|
| PPO A (trained 3 → 5 kg) | 3 kg | 1.931 mm |
| PPO B (trained 5 → 3 kg) | 5 kg | 1.763 mm |

The policies exhibit different residual-torque directions and vertical tracking behavior despite their relatively similar overall position errors.

## Repository Organization

```text
part1_modeling_control_analysis/
    Analytical kinematics, dynamics, model-based controllers, and tests

part2_ros2_isaac_validation/
    Initial ROS 2 controller integration and Isaac Sim validation

part3_residual_rl/
    ur5e_payload_learning/
        RL environment, training, and offline evaluation
    ur5e_payload_control/
        ROS 2 PhysX controller package and exported NumPy policies
    training_results/
        Selected PPO, SAC, and TD3 policies and training summaries
    physx_results/
        Timestamped physical-simulator validation CSVs

references/
    Accepted three-second transport reference
