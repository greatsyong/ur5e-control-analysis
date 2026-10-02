import numpy as np

from controllers.lqr import DiscreteJointLQRController
from dynamics.state_space import equilibrium_input
from analysis.export_lqr_gain import Ad, Bd, Q, R, TORQUE_LIMITS


DEG = np.pi / 180.0


# ============================================================
# Final discrete LQR controller
# ============================================================

controller = DiscreteJointLQRController(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    torque_limits=TORQUE_LIMITS,
)


q_ref = np.deg2rad([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0,
])

qdot_ref = np.zeros(6)

tau_ff = equilibrium_input(
    q_ref
)


def run_test(
    name,
    q,
    qdot,
):
    dx = np.concatenate([
        q - q_ref,
        qdot - qdot_ref,
    ])

    feedback = (
        -controller.K @ dx
    )

    unsaturated = (
        tau_ff + feedback
    )

    torque = controller.compute(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        tau_ff=tau_ff,
    )

    print(
        "\n========================================"
    )

    print(name)

    print(
        "========================================"
    )

    print(
        "tau_ff =",
        np.array2string(
            tau_ff,
            precision=15,
        ),
    )

    print(
        "feedback =",
        np.array2string(
            feedback,
            precision=15,
        ),
    )

    print(
        "unsaturated =",
        np.array2string(
            unsaturated,
            precision=15,
        ),
    )

    print(
        "torque =",
        np.array2string(
            torque,
            precision=15,
        ),
    )


# ============================================================
# Test 1
# Zero tracking error
# ============================================================

q1 = q_ref.copy()
qdot1 = np.zeros(6)

run_test(
    "TEST 1: ZERO TRACKING ERROR",
    q1,
    qdot1,
)


# ============================================================
# Test 2
# +1 degree J2 position error
# ============================================================

q2 = q_ref.copy()

q2[1] += (
    1.0 * DEG
)

qdot2 = np.zeros(6)

run_test(
    "TEST 2: +1 DEG J2 POSITION ERROR",
    q2,
    qdot2,
)


# ============================================================
# Test 3
# Mixed position + velocity errors
# ============================================================

q3 = q_ref.copy()

q3 += np.array([
     0.5,
    -0.8,
     0.3,
    -1.0,
     0.7,
    -0.4,
]) * DEG


qdot3 = np.array([
     2.0,
    -3.0,
     1.5,
    -2.5,
     1.0,
    -1.5,
]) * DEG


run_test(
    "TEST 3: MIXED POSITION/VELOCITY ERROR",
    q3,
    qdot3,
)


# ============================================================
# Test 4
# Saturation
# ============================================================

q4 = (
    q_ref
    + np.full(
        6,
        90.0 * DEG,
    )
)

qdot4 = np.full(
    6,
    100.0 * DEG,
)


run_test(
    "TEST 4: SATURATION",
    q4,
    qdot4,
)

# ============================================================
# C++ reference outputs after full-precision gravity update
# ============================================================

cpp_feedback = [
    np.array([
        -0.0, -0.0, -0.0,
        -0.0, -0.0, -0.0,
    ]),

    np.array([
         1.26115231522590e-01,
        -7.82248912809420e+00,
        -5.48940446524154e-01,
         2.00954765095532e-02,
        -3.81712021473317e-02,
         3.60404517786591e-04,
    ]),

    np.array([
        -6.70068596410414e+00,
         1.10779816605744e+01,
        -2.99845959119333e+00,
         2.61758484947850e-01,
        -2.62318861570490e-01,
         8.54898496769973e-03,
    ]),

    np.array([
        -7.44788277660989e+02,
        -8.68955531462657e+02,
        -6.81312741513110e+02,
        -1.18265328378586e+02,
        -4.06428810115088e+01,
        -1.06827856696277e+00,
    ]),
]


python_cases = [
    (q1, qdot1),
    (q2, qdot2),
    (q3, qdot3),
    (q4, qdot4),
]


print(
    "\n========================================"
)
print(
    "PYTHON-C++ LQR CROSS-VALIDATION"
)
print(
    "========================================"
)


max_error = 0.0

for i, ((q, qdot), cpp_fb) in enumerate(
    zip(
        python_cases,
        cpp_feedback,
    ),
    start=1,
):

    dx = np.concatenate([
        q - q_ref,
        qdot - qdot_ref,
    ])

    python_fb = (
        -controller.K @ dx
    )

    error = np.max(
        np.abs(
            python_fb - cpp_fb
        )
    )

    max_error = max(
        max_error,
        error,
    )

    print(
        f"Test {i}: "
        f"max feedback difference = "
        f"{error:.3e}"
    )


TOL = 1e-10

print(
    "\nMaximum difference:",
    f"{max_error:.3e}"
)

if max_error < TOL:

    print(
        "\nPASS: Python and C++ "
        "LQR implementations agree."
    )

else:

    raise AssertionError(
        "Python-C++ LQR "
        "cross-validation failed."
    )