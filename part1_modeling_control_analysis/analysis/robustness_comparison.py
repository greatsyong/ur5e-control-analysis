import os
import numpy as np
import matplotlib.pyplot as plt

from analysis.controller_metrics import summarize_tcp_errors


# ============================================================
# Configuration
# ============================================================

CONTROLLERS = [
    ("PID", {
        "Nominal": "results/pid/pid_final_benchmark.npz",
        "Payload": "results/robustness/payload/pid_payload_2kg.npz",
        "Disturbance": "results/robustness/disturbance/pid_J2_20Nm_200ms.npz",
    }),
    ("Computed Torque", {
        "Nominal": "results/computed_torque/computed_torque_benchmark.npz",
        "Payload": "results/robustness/payload/computed_torque_payload_2kg.npz",
        "Disturbance": "results/robustness/disturbance/computed_torque_J2_20Nm_200ms.npz",
    }),
    ("LQR", {
        "Nominal": "results/lqr/nominal/lqr_nominal_benchmark.npz",
        "Payload": "results/robustness/payload/lqr_payload_2kg/lqr_nominal_benchmark.npz",
        "Disturbance": "results/robustness/disturbance/lqr_J2_20Nm_200ms/lqr_nominal_benchmark.npz",
    }),
    ("MPC", {
        "Nominal": "results/mpc/nominal/mpc_nominal_benchmark.npz",
        "Payload": "results/robustness/payload/mpc_payload_2kg/mpc_nominal_benchmark.npz",
        "Disturbance": "results/robustness/disturbance/mpc_J2_20Nm_200ms/mpc_nominal_benchmark.npz",
    }),
]

CASES = ["Nominal", "Payload", "Disturbance"]

OUTPUT_DIR = "results/robustness/comparison"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_case(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path) as d:
        return {
            key: np.array(d[key])
            for key in d.files
        }


def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def compute_tcp_metrics(q, q_ref):
    """
    Recompute TCP metrics from the saved joint histories using
    the same common FK-based metric routine used by the
    controller benchmarks.
    """
    metrics = summarize_tcp_errors(
        q,
        q_ref,
    )

    # Existing project routine returns a dictionary.
    # Accept the established key naming used by the project.
    possible = {
        "pos_rmse": [
            "tcp_position_rmse_mm",
        ],
        "pos_max": [
            "tcp_position_max_mm",
        ],
        "pos_final": [
            "tcp_position_final_mm",
        ],
        "ori_rmse": [
            "tcp_orientation_rmse_deg",
        ],
        "ori_max": [
            "tcp_orientation_max_deg",
        ],
        "ori_final": [
            "tcp_orientation_final_deg",
        ],
    }

    out = {}

    for target, candidates in possible.items():
        found = False

        for key in candidates:
            if key in metrics:
                out[target] = float(metrics[key])
                found = True
                break

        if not found:
            raise KeyError(
                f"TCP metric '{target}' not found. "
                f"Available keys: {list(metrics.keys())}"
            )

    return out


# ============================================================
# Load and summarize
# ============================================================

results = {}

for controller_name, paths in CONTROLLERS:

    results[controller_name] = {}

    for case_name, path in paths.items():

        data = load_case(path)

        tcp = compute_tcp_metrics(
            data["q"],
            data["q_ref"],
        )

        results[controller_name][case_name] = {
            "path": path,
            "overall_joint_rmse_deg":
                np.rad2deg(scalar(data["overall_joint_rmse"])),
            "joint_rmse_deg":
                np.asarray(data["joint_rmse"]),
            "joint_max_error_deg":
                np.asarray(data["joint_max_error"]),
            "joint_final_error_deg":
                np.asarray(data["joint_final_error"]),
            "max_torque_nm":
                np.asarray(data["max_torque"]),
            "rms_torque_nm":
                np.asarray(data["rms_torque"]),
            "torque_utilization_pct":
                np.asarray(data["torque_utilization"]),
            **tcp,
        }


# ============================================================
# Numerical summary
# ============================================================

print()
print("=" * 110)
print("UR5e CONTROLLER ROBUSTNESS COMPARISON")
print("=" * 110)

print()
print("R1: +2.0 kg payload, CoM = tool-frame +Z 0.10 m")
print("    Controller model remains nominal bare-arm UR5e.")
print()
print("R2: J2 external torque pulse = +20 Nm")
print("    Applied from 2.0 s to 2.2 s.")
print()


header = (
    f"{'Controller':<18}"
    f"{'Case':<14}"
    f"{'Joint RMSE [deg]':>18}"
    f"{'TCP pos RMSE [mm]':>20}"
    f"{'TCP ori RMSE [deg]':>21}"
    f"{'Max torque util [%]':>21}"
)

print(header)
print("-" * len(header))

for controller_name, _ in CONTROLLERS:

    for case_name in CASES:

        r = results[controller_name][case_name]

        print(
            f"{controller_name:<18}"
            f"{case_name:<14}"
            f"{r['overall_joint_rmse_deg']:>18.6f}"
            f"{r['pos_rmse']:>20.6f}"
            f"{r['ori_rmse']:>21.6f}"
            f"{np.max(r['torque_utilization_pct']):>21.6f}"
        )


# ============================================================
# Degradation relative to nominal
# ============================================================

print()
print("=" * 110)
print("DEGRADATION RELATIVE TO NOMINAL")
print("=" * 110)

header = (
    f"{'Controller':<18}"
    f"{'Case':<14}"
    f"{'Joint RMSE ratio':>18}"
    f"{'TCP pos ratio':>18}"
    f"{'TCP ori ratio':>18}"
)

print(header)
print("-" * len(header))

for controller_name, _ in CONTROLLERS:

    nominal = results[controller_name]["Nominal"]

    for case_name in ["Payload", "Disturbance"]:

        r = results[controller_name][case_name]

        joint_ratio = (
            r["overall_joint_rmse_deg"]
            / nominal["overall_joint_rmse_deg"]
        )

        pos_ratio = (
            r["pos_rmse"]
            / nominal["pos_rmse"]
        )

        ori_ratio = (
            r["ori_rmse"]
            / nominal["ori_rmse"]
        )

        print(
            f"{controller_name:<18}"
            f"{case_name:<14}"
            f"{joint_ratio:>18.3f}"
            f"{pos_ratio:>18.3f}"
            f"{ori_ratio:>18.3f}"
        )


# ============================================================
# Save numerical summary
# ============================================================

controllers = np.array([
    name for name, _ in CONTROLLERS
])

joint_rmse = np.array([
    [
        results[name][case]["overall_joint_rmse_deg"]
        for case in CASES
    ]
    for name, _ in CONTROLLERS
])

tcp_pos_rmse = np.array([
    [
        results[name][case]["pos_rmse"]
        for case in CASES
    ]
    for name, _ in CONTROLLERS
])

tcp_ori_rmse = np.array([
    [
        results[name][case]["ori_rmse"]
        for case in CASES
    ]
    for name, _ in CONTROLLERS
])

max_torque_util = np.array([
    [
        np.max(
            results[name][case]["torque_utilization_pct"]
        )
        for case in CASES
    ]
    for name, _ in CONTROLLERS
])

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "robustness_comparison.npz",
    ),
    controllers=controllers,
    cases=np.array(CASES),
    joint_rmse_deg=joint_rmse,
    tcp_position_rmse_mm=tcp_pos_rmse,
    tcp_orientation_rmse_deg=tcp_ori_rmse,
    max_torque_utilization_pct=max_torque_util,
)


# ============================================================
# Figures
# ============================================================

x = np.arange(
    len(controllers)
)

width = 0.25


def grouped_bar(
    values,
    ylabel,
    title,
    filename,
    log_scale=False,
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for i, case in enumerate(CASES):

        ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            label=case,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(controllers)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            FIGURE_DIR,
            filename,
        ),
        dpi=200,
    )

    plt.close(fig)


# Log scale is useful because Computed Torque nominal tracking
# error is orders of magnitude smaller than its mismatch result.
grouped_bar(
    joint_rmse,
    "Overall Joint RMSE [deg]",
    "Joint Tracking Robustness",
    "robustness_joint_rmse.png",
    log_scale=True,
)

grouped_bar(
    tcp_pos_rmse,
    "TCP Position RMSE [mm]",
    "TCP Position Robustness",
    "robustness_tcp_position_rmse.png",
    log_scale=True,
)

grouped_bar(
    tcp_ori_rmse,
    "TCP Orientation RMSE [deg]",
    "TCP Orientation Robustness",
    "robustness_tcp_orientation_rmse.png",
    log_scale=True,
)

grouped_bar(
    max_torque_util,
    "Maximum Torque Utilization [%]",
    "Control Effort Under Robustness Tests",
    "robustness_torque_utilization.png",
    log_scale=False,
)


# ============================================================
# Payload degradation figure
# ============================================================

payload_joint_ratio = np.array([
    results[name]["Payload"]["overall_joint_rmse_deg"]
    / results[name]["Nominal"]["overall_joint_rmse_deg"]
    for name, _ in CONTROLLERS
])

dist_joint_ratio = np.array([
    results[name]["Disturbance"]["overall_joint_rmse_deg"]
    / results[name]["Nominal"]["overall_joint_rmse_deg"]
    for name, _ in CONTROLLERS
])

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.bar(
    x - width / 2,
    payload_joint_ratio,
    width,
    label="Payload mismatch",
)

ax.bar(
    x + width / 2,
    dist_joint_ratio,
    width,
    label="External disturbance",
)

ax.set_xticks(x)
ax.set_xticklabels(controllers)
ax.set_ylabel(
    "Joint RMSE / Nominal Joint RMSE"
)
ax.set_title(
    "Tracking Degradation Relative to Nominal"
)
ax.set_yscale("log")
ax.legend()
ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "robustness_degradation_ratio.png",
    ),
    dpi=200,
)

plt.close(fig)


print()
print("=" * 110)
print("FILES SAVED")
print("=" * 110)
print(
    os.path.join(
        OUTPUT_DIR,
        "robustness_comparison.npz",
    )
)
print(FIGURE_DIR)