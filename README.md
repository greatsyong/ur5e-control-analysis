# UR5e Control Analysis

Analytical modeling, dynamics validation, and controller development for the Universal Robots UR5e manipulator.

This repository begins with **Pre-Part 1 — Model Setup & Cross-Validation**, which establishes a consistent nominal rigid-body model between a Python analytical implementation, the official ROS 2 UR5e robot description, and NVIDIA Isaac Sim.

The objective is not to treat simulation as ground truth. Before controller design, the analytical model and simulation environment are independently checked and then cross-validated at the kinematic and dynamic levels.

---

## Project Roadmap

### Pre-Part 1 — Model Setup & Cross-Validation
**Status: Completed**

- UR5e nominal parameter definition
- Standard DH kinematic model
- Forward and inverse kinematics
- Geometric Jacobian
- Singularity analysis
- URDF / physical-link frame reconciliation
- Link mass, center-of-mass, and inertia modeling
- Joint-space mass matrix
- Gravity generalized-force calculation
- Isaac Sim fixed-base configuration
- Physics and joint-drive validation
- Python ↔ Isaac Sim cross-validation

### Part 1A — Kinematics & Dynamics Characterization
**Status: In Development**

Planned work includes:

- Extended kinematic characterization
- Workspace and singularity analysis
- Nonlinear joint-space dynamics
- Configuration-dependent dynamic behavior
- Linearization
- Pole / eigenvalue analysis
- Basic transient-response characterization

### Part 1B — Controller Design & Tuning
**Status: Planned**

Planned controllers and evaluation:

- PID control
- Computed-torque control
- LQR
- MPC
- Tracking RMSE
- Overshoot and settling time
- Convergence behavior
- Control effort
- Computational cost
- Constraint handling
- Tuning sensitivity

---

# Pre-Part 1 — Model Setup & Cross-Validation

## 1. Motivation

A controller should not be evaluated against a simulation model whose physical assumptions have not been verified.

The analytical model and the simulator may describe the same robot while using different:

- coordinate-frame conventions,
- center-of-mass representations,
- inertia frames,
- base constraints,
- gravity definitions,
- and joint-drive assumptions.

For this reason, I treated the simulation environment as another model that required validation rather than assuming that disagreement with simulation implied an error in the analytical implementation.

The validation sequence was therefore:

```text
Manufacturer / ROS model data
            │
            ├──────────────► Python analytical model
            │
            └──────────────► Isaac Sim model
                              │
                              ▼
                    Simulation validation
                              │
                              ▼
                        Link poses
                              │
                              ▼
                         Jacobian
                              │
                              ▼
                           M(q)
                              │
                              ▼
                           g(q)
                              │
                              ▼
                   Validated common model
```

---

## 2. Reference Robot

The project uses the nominal **Universal Robots UR5e** model.

The analytical model is based on manufacturer geometry and the official ROS 2 UR robot description.

### Standard DH Parameters

| Joint | a [m] | d [m] | alpha [rad] |
|---|---:|---:|---:|
| 1 | 0 | 0.1625 | +pi/2 |
| 2 | -0.4250 | 0 | 0 |
| 3 | -0.3922 | 0 | 0 |
| 4 | 0 | 0.1333 | +pi/2 |
| 5 | 0 | 0.0997 | -pi/2 |
| 6 | 0 | 0.0996 | 0 |

### Moving-Link Masses

```text
[3.761, 8.058, 2.846, 1.370, 1.300, 0.365] kg
```

The six modeled moving links have a total mass of **17.7 kg**.

This value is intentionally not forced to equal the complete product-level arm weight because the two values have different accounting boundaries.

---

## 3. Analytical Model

The Python implementation separates kinematic and physical-link representations.

### Kinematics

Standard DH transformations are used for:

- forward kinematics,
- inverse kinematics,
- geometric Jacobian calculation,
- and singularity analysis.

### Dynamics

URDF-aligned physical-link frames are used for:

- center-of-mass locations,
- inertia tensors,
- link Jacobians,
- mass-matrix construction,
- and gravitational potential energy.

This distinction is important because a DH frame is not automatically the correct frame in which a physical inertia tensor is defined.

---

## 4. Kinematic Validation

Forward kinematics were independently checked using finite differences and rotation-matrix consistency tests.

At the zero configuration:

```text
TCP position:

[-0.8172, -0.2329, 0.0628] m
```

The geometric Jacobian was also validated numerically.

At

```text
q = [10, -40, 60, -30, 45, 20] deg
```

the Jacobian was full rank.

The minimum singular value was approximately:

```text
0.15682
```

with condition number:

```text
12.95
```

A known singular configuration at `q = 0` produced rank 5 and a zero minimum singular value, providing an additional structural check of the implementation.

---

## 5. Inverse Kinematics

Numerical inverse kinematics was implemented using both:

- Moore-Penrose pseudoinverse
- Damped Least Squares (DLS)

Multiple initial configurations were tested.

The experiments confirmed that different joint configurations can produce the same end-effector pose.

A DLS implementation with a joint-update limiter was also evaluated near difficult initial conditions.

The pseudoinverse method converged faster in the tested case but generated extremely large intermediate joint updates.

The DLS + step-limited implementation bounded the maximum update to:

```text
0.2 rad = 11.46 deg
```

at the cost of slightly slower convergence.

---

## 6. Isaac Sim Model Validation

The official ROS 2 UR5e description was converted from Xacro/URDF and imported into **NVIDIA Isaac Sim 6.1**.

Several simulation assumptions had to be checked before comparing dynamics.

### Fixed Base

The first simulator mass matrix had dimension:

```text
12 x 12
```

instead of the expected:

```text
6 x 6
```

This indicated that the imported robot was being interpreted as a floating-base system.

A temporary manually created fixed joint physically constrained the robot but introduced Isaac robot-schema errors.

The manual joint was therefore removed and the articulation was converted using Isaac's fixed-base utilities.

The final model is a clean:

```text
6-DOF fixed-base articulation
```

---

## 7. Gravity and Joint Drives

The initial PhysicsScene did not contain a normal finite gravity definition.

The validated environment uses:

```text
gravity = [0, 0, -9.81] m/s^2
```

The imported joints initially had zero stiffness and zero damping.

Therefore, commanded joint positions were not guaranteed to remain fixed under gravity.

Position drives were configured for **model-validation purposes only**.

These gains are not interpreted as identified physical UR5e servo parameters.

A shoulder-lift step validation produced:

```text
Peak position error: 0.036 deg
RMS position error:  0.015 deg
```

The simulator could therefore hold the configurations required for static cross-validation.

---

## 8. Common Validation Configuration

The primary nonzero validation configuration was:

```text
q = [10, -40, 60, -30, 45, 20] deg
```

The simulator held:

```text
[9.999999, -39.999996, 60.000000,
 -30.000000, 45.000000, 19.999998] deg
```

A nonzero configuration was selected because agreement only at the zero pose can hide axis, sign, and transformation errors.

---

## 9. Physical-Link Pose Cross-Validation

At the validation configuration, the wrist-3 frame origin was:

### Python

```text
[0.733548, 0.336215, 0.215589] m
```

### Isaac Sim

```text
[0.733552, 0.336198, 0.215773] m
```

The Euclidean position difference is approximately:

```text
0.185 mm
```

Intermediate physical-link positions and orientations also showed close agreement.

---

## 10. Jacobian Cross-Validation

The wrist-3 geometric Jacobian was independently calculated in Python and obtained from Isaac Sim.

Representative joint-axis comparison:

```text
Python J2 angular axis:
[-0.173648, 0.984808, 0.000000]

Isaac Sim:
[-0.173630, 0.984811, 0.000000]
```

The close agreement validates the physical joint origins, rotation axes, and differential kinematic propagation used by the analytical dynamics model.

---

## 11. Mass Matrix

The analytical joint-space mass matrix is assembled as

```text
M(q) =
sum_i [
    m_i Jv_i^T Jv_i
    +
    Jw_i^T (R_i I_i R_i^T) Jw_i
]
```

At the common validation configuration, selected diagonal terms were:

| Element | Python | Isaac Sim |
|---|---:|---:|
| M11 | 3.103413 | 3.103188 |
| M22 | 3.342034 | 3.342024 |
| M33 | 0.833906 | 0.833856 |
| M44 | 0.021196 | 0.021143 |
| M66 | 0.000258 | 0.000258 |

The Python mass matrix was symmetric and positive definite.

---

## 12. Diagnostic Case: Initial Mass-Matrix Disagreement

The first Python ↔ Isaac comparison produced an apparent relative mass-matrix discrepancy of approximately:

```text
64.7%
```

I initially considered inertia-frame or transformation errors because these are common causes of rigid-body dynamics disagreement.

However, the analytical kinematics and Jacobian had already passed independent validation.

I therefore considered changing the analytical model before validating the simulator to be the less defensible approach.

The investigation identified several simulator-side issues:

1. floating-base interpretation,
2. an inappropriate temporary manual base constraint,
3. invalid gravity configuration,
4. zero-stiffness / zero-damping joint drives,
5. and the resulting possibility that the simulator was not evaluating the same joint configuration as Python.

After correcting these conditions, the mass matrices converged to close agreement without forcing the analytical model to reproduce the original invalid simulation result.

This became one of the main engineering conclusions of the project:

> **Model validation requires simulation validation first.**

---

## 13. Gravity Generalized Forces

The analytical gravity vector was independently calculated from gravitational potential energy:

```text
U(q) = -sum_i m_i g^T p_ci(q)

g(q) = dU/dq
```

At the validation configuration:

| Joint | Python [Nm] | Isaac Sim [Nm] | Absolute Difference [Nm] |
|---|---:|---:|---:|
| J1 | 0.000000 | 0.000008 | 0.000008 |
| J2 | -49.685494 | -49.681679 | 0.003815 |
| J3 | -18.034631 | -18.036629 | 0.001998 |
| J4 | -0.707495 | -0.707961 | 0.000466 |
| J5 | +0.068868 | +0.068971 | 0.000103 |
| J6 | 0.000000 | 0.000000 | 0.000000 |

The largest absolute difference occurred at J2 and corresponds to approximately:

```text
0.0077%
```

of the Python value.

The sign convention is consistent with:

```text
M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau
```

---

## 14. Validation Status

| Component | Status |
|---|---|
| Nominal UR5e parameters | Validated |
| Physical-link frames | Validated |
| Nonzero link poses | Validated |
| Joint origins / axes | Validated |
| Geometric Jacobian | Validated |
| Mass matrix M(q) | Validated |
| Gravity vector g(q) | Validated |
| Fixed-base Isaac articulation | Validated |
| Static configuration holding | Validated |

The Python analytical model and Isaac Sim now provide a consistent nominal rigid-body baseline for subsequent controller development.

---

## 15. Model Boundaries

The current validation has deliberate limits.

- Nominal UR5e geometry is used; robot-specific factory calibration is not included.
- No gripper or payload is modeled.
- Links are treated as rigid bodies.
- Detailed gearbox compliance, backlash, nonlinear friction, and actuator dynamics are not identified.
- Isaac joint gains are validation infrastructure, not physical servo identification.
- The primary dynamics cross-validation was performed at one nonzero configuration.
- Contact dynamics are outside the current scope.
- Simulator effort limits are treated as model limits rather than measured motor torque ratings.

A future extension can automate the same validation across multiple configurations throughout the robot workspace.

---

## Repository Structure

```text
ur5e_control_analysis/
├── dynamics/
│   ├── __init__.py
│   └── rigid_body_dynamics.py
├── kinematics/
│   ├── __init__.py
│   ├── forward_kinematics.py
│   ├── inverse_kinematics.py
│   └── jacobian.py
├── models/
│   ├── __init__.py
│   └── ur5e_parameters.py
├── notebooks/
│   └── 01_kinematics.ipynb
├── results/
│   ├── data/
│   └── figures/
├── tests/
└── visualization/
    ├── __init__.py
    └── robot_plot.py
```

---

## Software Environment

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- NumPy
- Jupyter
- NVIDIA Isaac Sim 6.1

---

## References

- Universal Robots — DH Parameters for calculations of kinematics and dynamics
- Universal Robots — Universal Robots ROS 2 Description
- NVIDIA Isaac Sim 6.1 — Manipulator and articulation documentation
- Universal Robots — UR5e Technical Specification

---

## Author

**Sooyong Kim**

Robotics & Autonomous Systems  
[sooyongtech.dev](https://sooyongtech.dev)
