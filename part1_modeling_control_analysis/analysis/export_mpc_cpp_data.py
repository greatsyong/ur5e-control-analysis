"""
Export the finalized UR5e MPC design/reference data for the C++ online controller.

Run from the project root:
    PYTHONPATH=. python3 analysis/export_mpc_cpp_data.py

Output:
    results/mpc/cpp_data/mpc_cpp_data.npz
    results/mpc/cpp_data/mpc_cpp_data.txt

The NPZ is the authoritative machine-readable export.
The TXT file is a compact manifest for human inspection.
"""

import os
import runpy
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "analysis" / "mpc_nominal_benchmark.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "mpc" / "cpp_data"
OUTPUT_NPZ = OUTPUT_DIR / "mpc_cpp_data.npz"
OUTPUT_TXT = OUTPUT_DIR / "mpc_cpp_data.txt"


def require(ns, name):
    if name not in ns:
        raise RuntimeError(
            f"Required variable '{name}' was not produced by "
            f"{BENCHMARK.name}."
        )
    return ns[name]


def main():
    # Reuse the finalized Python implementation exactly instead of
    # duplicating the MPC design equations in a second script.
    print("Running finalized Python MPC benchmark to obtain export data...")
    ns = runpy.run_path(str(BENCHMARK), run_name="__mpc_export__")

    dt = float(require(ns, "DT"))
    np_horizon = int(require(ns, "NP"))
    nc_horizon = int(require(ns, "NC"))
    rho_u = float(require(ns, "RHO_U"))

    nx = int(require(ns, "nx"))
    nu = int(require(ns, "nu"))
    n_decision = int(require(ns, "n_decision"))

    x_eq = np.asarray(require(ns, "x_eq"), dtype=np.float64)
    tau_eq = np.asarray(require(ns, "tau_eq"), dtype=np.float64)
    torque_limits = np.asarray(
        require(ns, "TORQUE_LIMITS"), dtype=np.float64
    )

    # OSQP quadratic matrix. Python solves:
    #   0.5 * Wc' P Wc + q' Wc
    P_sparse = require(ns, "P")
    P_dense = np.asarray(P_sparse.toarray(), dtype=np.float64)

    # Final optimized online gradient:
    #   q_k = STATE_GRADIENT_MAP @ (x_k - x_eq)
    #         + reference_gradient_all[k]
    state_gradient_map = np.asarray(
        require(ns, "STATE_GRADIENT_MAP"), dtype=np.float64
    )
    reference_gradient_all = np.asarray(
        require(ns, "reference_gradient_all"), dtype=np.float64
    )

    # Known nominal reference / gravity feedforward. The C++ controller
    # uses these to reconstruct the same time-varying input bounds and
    # current feedforward torque as Python.
    tau_ff_padded = np.asarray(
        require(ns, "tau_ff_padded"), dtype=np.float64
    )
    q_ref_all = np.asarray(
        require(ns, "q_ref_all"), dtype=np.float64
    )
    qdot_ref_all = np.asarray(
        require(ns, "qdot_ref_all"), dtype=np.float64
    )
    times = np.asarray(require(ns, "times"), dtype=np.float64)

    # Reference Python outputs for later C++ numerical cross-validation.
    q_history = np.asarray(
        require(ns, "q_history"), dtype=np.float64
    )
    qdot_history = np.asarray(
    require(ns, "qdot_history"), dtype=np.float64
    )
    x_history = np.hstack([
        q_history,
        qdot_history,
    ])
    tau_history = np.asarray(
        require(ns, "tau_history"), dtype=np.float64
    )
    osqp_iterations = np.asarray(
        require(ns, "osqp_iteration_history"), dtype=np.int64
    )

    expected = {
        "x_eq": (nx,),
        "tau_eq": (nu,),
        "torque_limits": (nu,),
        "P": (n_decision, n_decision),
        "state_gradient_map": (n_decision, nx),
        "reference_gradient_all": (len(times), n_decision),
        "tau_ff_padded": (len(times) + np_horizon, nu),
        "q_ref_all": (len(times), nu),
        "qdot_ref_all": (len(times), nu),
        "q_history": (len(times), nu),
        "tau_history": (len(times), nu),
    }

    actual = {
        "x_eq": x_eq.shape,
        "tau_eq": tau_eq.shape,
        "torque_limits": torque_limits.shape,
        "P": P_dense.shape,
        "state_gradient_map": state_gradient_map.shape,
        "reference_gradient_all": reference_gradient_all.shape,
        "tau_ff_padded": tau_ff_padded.shape,
        "q_ref_all": q_ref_all.shape,
        "qdot_ref_all": qdot_ref_all.shape,
        "q_history": q_history.shape,
        "tau_history": tau_history.shape,
    }

    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise RuntimeError(
                f"{name}: expected shape {expected_shape}, "
                f"got {actual[name]}"
            )

    arrays_to_check = {
        "x_eq": x_eq,
        "tau_eq": tau_eq,
        "torque_limits": torque_limits,
        "P": P_dense,
        "state_gradient_map": state_gradient_map,
        "reference_gradient_all": reference_gradient_all,
        "tau_ff_padded": tau_ff_padded,
        "q_ref_all": q_ref_all,
        "qdot_ref_all": qdot_ref_all,
        "q_history": q_history,
        "tau_history": tau_history,
    }

    for name, arr in arrays_to_check.items():
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains non-finite values.")

    # P should be symmetric up to floating-point noise.
    p_symmetry_error = float(
        np.max(np.abs(P_dense - P_dense.T))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_NPZ,
        format_version=np.array([1], dtype=np.int64),
        dt=np.array([dt], dtype=np.float64),
        prediction_horizon=np.array([np_horizon], dtype=np.int64),
        control_horizon=np.array([nc_horizon], dtype=np.int64),
        rho_u=np.array([rho_u], dtype=np.float64),
        nx=np.array([nx], dtype=np.int64),
        nu=np.array([nu], dtype=np.int64),
        n_decision=np.array([n_decision], dtype=np.int64),
        osqp_eps_abs=np.array([1e-6], dtype=np.float64),
        osqp_eps_rel=np.array([1e-6], dtype=np.float64),
        x_eq=x_eq,
        tau_eq=tau_eq,
        torque_limits=torque_limits,
        P=P_dense,
        state_gradient_map=state_gradient_map,
        reference_gradient_all=reference_gradient_all,
        tau_ff_padded=tau_ff_padded,
        time=times,
        q_ref=q_ref_all,
        qdot_ref=qdot_ref_all,
        python_q=q_history,
        python_qdot=qdot_history,
        python_x=x_history,
        python_tau=tau_history,
        python_osqp_iterations=osqp_iterations,
    )

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write("UR5e finalized MPC C++ export manifest\n")
        f.write("=" * 52 + "\n")
        f.write(f"dt                  : {dt:.17g}\n")
        f.write(f"Np                  : {np_horizon}\n")
        f.write(f"Nc                  : {nc_horizon}\n")
        f.write(f"rho_u               : {rho_u:.17g}\n")
        f.write(f"nx                  : {nx}\n")
        f.write(f"nu                  : {nu}\n")
        f.write(f"n_decision          : {n_decision}\n")
        f.write("OSQP eps_abs/rel    : 1e-6 / 1e-6\n")
        f.write(f"samples             : {len(times)}\n")
        f.write(f"P shape             : {P_dense.shape}\n")
        f.write(
            "STATE_GRADIENT_MAP : "
            f"{state_gradient_map.shape}\n"
        )
        f.write(
            "reference gradients : "
            f"{reference_gradient_all.shape}\n"
        )
        f.write(
            "tau_ff_padded       : "
            f"{tau_ff_padded.shape}\n"
        )
        f.write(
            "P symmetry max err  : "
            f"{p_symmetry_error:.6e}\n"
        )

    print("\n" + "=" * 68)
    print("MPC C++ DATA EXPORT COMPLETE")
    print("=" * 68)
    print(f"NPZ : {OUTPUT_NPZ}")
    print(f"TXT : {OUTPUT_TXT}")
    print()
    print(f"P                      : {P_dense.shape}")
    print(f"STATE_GRADIENT_MAP     : {state_gradient_map.shape}")
    print(f"reference_gradient_all : {reference_gradient_all.shape}")
    print(f"tau_ff_padded          : {tau_ff_padded.shape}")
    print(f"Python state history   : {q_history.shape}")
    print(f"Python torque history  : {tau_history.shape}")
    print(f"P symmetry max error   : {p_symmetry_error:.3e}")
    print("\nValidation: PASS")


if __name__ == "__main__":
    main()