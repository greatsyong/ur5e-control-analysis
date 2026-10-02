import numpy as np

from controllers.mpc import (
    build_prediction_matrices,
)


# ============================================================
# Small deterministic system
# ============================================================

DT = 0.1

Ad = np.array([
    [1.0, DT],
    [0.0, 1.0],
])

Bd = np.array([
    [0.5 * DT**2],
    [DT],
])

N = 5


# ============================================================
# Build prediction matrices
# ============================================================

F, G = build_prediction_matrices(
    Ad,
    Bd,
    N,
)


print("\n========================================")
print("MPC PREDICTION MATRIX TEST")
print("========================================")

print("\nF shape:")
print(F.shape)

print("\nG shape:")
print(G.shape)


# ============================================================
# Deterministic state / input sequence
# ============================================================

x0 = np.array([
    0.3,
    -0.2,
])

U = np.array([
     1.0,
    -0.5,
     0.2,
     0.0,
     0.7,
])


# ============================================================
# Matrix prediction
# ============================================================

X_matrix = (
    F @ x0
    + G @ U
)


# ============================================================
# Direct propagation
# ============================================================

x = x0.copy()

X_direct = []

for k in range(N):

    u = np.array([
        U[k]
    ])

    x = (
        Ad @ x
        + Bd @ u
    )

    X_direct.append(
        x.copy()
    )


X_direct = np.concatenate(
    X_direct
)


# ============================================================
# Compare
# ============================================================

error = (
    X_matrix
    - X_direct
)

max_error = np.max(
    np.abs(error)
)


np.set_printoptions(
    precision=15,
    suppress=False,
)


print("\nMatrix prediction:")
print(X_matrix)

print("\nDirect propagation:")
print(X_direct)

print("\nDifference:")
print(error)

print(
    "\nMaximum absolute difference:"
)

print(
    f"{max_error:.3e}"
)


TOL = 1e-12

if max_error < TOL:

    print(
        "\nPASS: stacked MPC prediction "
        "matches direct propagation."
    )

else:

    raise AssertionError(
        "MPC prediction matrix validation failed."
    )

# ============================================================
# Unconstrained finite-horizon MPC test
# ============================================================

from controllers.mpc import (
    UnconstrainedMPC,
)


print(
    "\n========================================"
)
print(
    "UNCONSTRAINED MPC TEST"
)
print(
    "========================================"
)


Q = np.diag([
    10.0,
    1.0,
])

R = np.array([
    [0.1],
])


mpc = UnconstrainedMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    horizon=N,
)


# Reference:
# drive position and velocity to zero
X_ref = np.zeros(
    N * 2
)


U_opt = mpc.solve(
    x0,
    X_ref,
)


u0 = mpc.compute(
    x0,
    X_ref,
)


print("\nOptimal input sequence:")
print(U_opt)

print("\nFirst MPC input:")
print(u0)


# ============================================================
# First-order optimality condition
#
# grad J / 2 =
#
# H U
# +
# G^T Qbar (F x - Xref)
#
# At optimum this must be zero.
# ============================================================

gradient_half = (
    mpc.H @ U_opt
    + mpc.G.T
    @ mpc.Qbar
    @ (
        mpc.F @ x0
        - X_ref
    )
)


gradient_error = np.max(
    np.abs(
        gradient_half
    )
)


print(
    "\nMaximum first-order optimality residual:"
)

print(
    f"{gradient_error:.3e}"
)


# ============================================================
# Cost comparison
#
# Verify that the optimized sequence produces lower cost
# than zero control.
# ============================================================

def mpc_cost(
    U,
):
    X = (
        mpc.F @ x0
        + mpc.G @ U
    )

    e = (
        X - X_ref
    )

    return (
        e.T @ mpc.Qbar @ e
        + U.T @ mpc.Rbar @ U
    )


J_opt = mpc_cost(
    U_opt
)

J_zero = mpc_cost(
    np.zeros_like(
        U_opt
    )
)


print(
    "\nOptimized cost:"
)

print(
    f"{J_opt:.12f}"
)

print(
    "\nZero-input cost:"
)

print(
    f"{J_zero:.12f}"
)


# ============================================================
# PASS / FAIL
# ============================================================

TOL_OPT = 1e-12

if gradient_error >= TOL_OPT:

    raise AssertionError(
        "Unconstrained MPC optimality "
        "condition failed."
    )


if not J_opt < J_zero:

    raise AssertionError(
        "Optimized MPC cost is not "
        "lower than zero-input cost."
    )


print(
    "\nPASS: unconstrained MPC solution "
    "satisfies the optimality condition "
    "and reduces the finite-horizon cost."
)

# ============================================================
# OSQP vs closed-form unconstrained MPC
# ============================================================

from controllers.mpc import OSQPMPC


print(
    "\n========================================"
)
print(
    "OSQP VS CLOSED-FORM MPC TEST"
)
print(
    "========================================"
)


mpc_osqp = OSQPMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    horizon=N,
)


U_osqp = mpc_osqp.solve(
    x0,
    X_ref,
)


difference = (
    U_osqp - U_opt
)

max_difference = np.max(
    np.abs(difference)
)


print("\nClosed-form solution:")
print(U_opt)

print("\nOSQP solution:")
print(U_osqp)

print("\nDifference:")
print(difference)

print(
    "\nMaximum absolute difference:"
)
print(
    f"{max_difference:.3e}"
)


TOL_OSQP = 1e-8

if max_difference >= TOL_OSQP:

    raise AssertionError(
        "OSQP solution does not match "
        "closed-form MPC solution."
    )


print(
    "\nPASS: OSQP reproduces the "
    "closed-form unconstrained MPC solution."
)

# ============================================================
# Constrained MPC test
# ============================================================

print(
    "\n========================================"
)
print(
    "CONSTRAINED MPC TEST"
)
print(
    "========================================"
)


u_lower = np.array([
    -0.30,
])

u_upper = np.array([
     0.30,
])


mpc_constrained = OSQPMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    horizon=N,
    input_lower=u_lower,
    input_upper=u_upper,
)


U_constrained = mpc_constrained.solve(
    x0,
    X_ref,
)


print("\nUnconstrained solution:")
print(U_opt)

print("\nConstrained solution:")
print(U_constrained)


# ============================================================
# Constraint checks
# ============================================================

lower_violation = np.max(
    u_lower[0] - U_constrained
)

upper_violation = np.max(
    U_constrained - u_upper[0]
)

max_violation = max(
    0.0,
    lower_violation,
    upper_violation,
)


print(
    "\nMaximum constraint violation:"
)
print(
    f"{max_violation:.3e}"
)


# First unconstrained input is about -0.913,
# therefore the lower bound should become active.

first_input_error = abs(
    U_constrained[0]
    - u_lower[0]
)


print(
    "\nFirst constrained input:"
)
print(
    f"{U_constrained[0]:.15f}"
)

print(
    "\nDistance from active lower bound:"
)
print(
    f"{first_input_error:.3e}"
)


# ============================================================
# Compare constrained and unconstrained costs
# ============================================================

J_constrained = mpc_cost(
    U_constrained
)


print(
    "\nUnconstrained optimal cost:"
)
print(
    f"{J_opt:.12f}"
)

print(
    "\nConstrained optimal cost:"
)
print(
    f"{J_constrained:.12f}"
)


# ============================================================
# PASS / FAIL
# ============================================================

TOL_CONSTRAINT = 1e-8

if max_violation >= TOL_CONSTRAINT:

    raise AssertionError(
        "MPC input constraint violated."
    )


if first_input_error >= TOL_CONSTRAINT:

    raise AssertionError(
        "Expected lower input bound "
        "was not active."
    )


if J_constrained < J_opt - 1e-10:

    raise AssertionError(
        "Constrained optimum cannot have "
        "lower cost than unconstrained optimum."
    )


print(
    "\nPASS: MPC respects the input bounds, "
    "activates the expected constraint, "
    "and produces the expected increase "
    "in optimal cost."
)