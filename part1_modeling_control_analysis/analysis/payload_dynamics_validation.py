"""
Validation of payload-perturbed UR5e dynamics.

This script does NOT run controller robustness yet.
It verifies that the payload model produces physically
consistent changes in M(q), g(q), and static equilibrium.
"""

import numpy as np

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
)

from dynamics.payload_dynamics import (
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_COM_TOOL,
    payload_com_position,
    mass_matrix_payload,
    gravity_vector_payload,
    state_derivative_payload,
)


np.set_printoptions(
    precision=8,
    suppress=True,
)


q = np.deg2rad([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0,
])

qdot = np.zeros(6)


print("\nPAYLOAD DYNAMICS VALIDATION")
print("=" * 60)

print(
    f"Payload mass       : "
    f"{DEFAULT_PAYLOAD_MASS:.3f} kg"
)

print(
    "Payload CoM [tool] :",
    DEFAULT_PAYLOAD_COM_TOOL,
    "m",
)


# ---------------------------------------------------------
# Payload position
# ---------------------------------------------------------

p_payload = payload_com_position(q)

print(
    "\nPayload CoM [base] :",
    p_payload,
    "m",
)


# ---------------------------------------------------------
# Mass matrix
# ---------------------------------------------------------

M_nom = mass_matrix(q)

M_payload = mass_matrix_payload(q)

delta_M = M_payload - M_nom

print("\nNominal diag(M):")
print(np.diag(M_nom))

print("\nPayload diag(M):")
print(np.diag(M_payload))

print("\nDelta diag(M):")
print(np.diag(delta_M))


# ---------------------------------------------------------
# Basic mass-matrix properties
# ---------------------------------------------------------

symmetry_error = np.max(
    np.abs(
        M_payload - M_payload.T
    )
)

eigenvalues = np.linalg.eigvalsh(
    M_payload
)

print(
    "\nMass-matrix symmetry error :",
    f"{symmetry_error:.3e}",
)

print(
    "Minimum eigenvalue          :",
    f"{np.min(eigenvalues):.6e}",
)


# ---------------------------------------------------------
# Gravity
# ---------------------------------------------------------

g_nom = gravity_vector(q)

g_payload = gravity_vector_payload(q)

delta_g = g_payload - g_nom

print("\nNominal gravity [Nm]:")
print(g_nom)

print("\nPayload gravity [Nm]:")
print(g_payload)

print("\nDelta gravity [Nm]:")
print(delta_g)


# ---------------------------------------------------------
# Static equilibrium check
# ---------------------------------------------------------

x_static = np.concatenate([
    q,
    qdot,
])

xdot = state_derivative_payload(
    x_static,
    g_payload,
)

static_residual = np.max(
    np.abs(xdot)
)

print(
    "\nStatic equilibrium residual :",
    f"{static_residual:.3e}",
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

passed = (
    symmetry_error < 1e-10
    and np.min(eigenvalues) > 0.0
    and static_residual < 1e-9
)

print("\nValidation :", end=" ")

if passed:
    print("PASS")
else:
    print("FAIL")