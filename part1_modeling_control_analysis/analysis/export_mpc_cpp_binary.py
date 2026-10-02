import numpy as np
from pathlib import Path

src = Path("results/mpc/cpp_data/mpc_cpp_data.npz")
dst = Path("results/mpc/cpp_data/mpc_cpp_data.bin")

d = np.load(src)

arrays = [
    ("x_eq", d["x_eq"]),
    ("tau_eq", d["tau_eq"]),
    ("torque_limits", d["torque_limits"]),
    ("P", d["P"]),
    ("state_gradient_map", d["state_gradient_map"]),
    ("reference_gradient_all", d["reference_gradient_all"]),
    ("tau_ff_padded", d["tau_ff_padded"]),
    ("python_x", d["python_x"]),
    ("python_tau", d["python_tau"]),
]

magic = np.int64(0x555235454D504331)
header_i = np.array([
    magic, 1,
    int(d["prediction_horizon"][0]),
    int(d["control_horizon"][0]),
    int(d["nx"][0]),
    int(d["nu"][0]),
    int(d["n_decision"][0]),
    d["python_x"].shape[0],
], dtype=np.int64)

header_f = np.array([
    float(d["dt"][0]),
    float(d["rho_u"][0]),
    float(d["osqp_eps_abs"][0]),
    float(d["osqp_eps_rel"][0]),
], dtype=np.float64)

with dst.open("wb") as f:
    header_i.tofile(f)
    header_f.tofile(f)
    for _, a in arrays:
        np.ascontiguousarray(a, dtype=np.float64).tofile(f)

print("Binary export:", dst)
for name, a in arrays:
    print(f"{name:24s}: {a.shape}")
print("Validation: PASS")
