#!/usr/bin/env python3
"""Offline gain sweep for UR5e computed-torque PD, using the project's dynamics.

Model-only screening, not a PhysX validation or a real-robot safety certificate.
Python 3.10+, numpy, scipy, and existing ur5e_control_analysis modules.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

JOINTS = 6
TORQUE_LIMITS = np.array([150., 150., 150., 28., 28., 28.])
INITIAL_ERROR_RAD = np.deg2rad([0., 0.5, -0.5, 0., 0., 0.])
COM_TOOL = np.array([0., 0., 0.025])
BASE_TO_WORLD = np.diag([-1., -1., 1.])


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path,
                   help='3s retimed reference CSV (121 waypoints)')
    p.add_argument('--model-dir', required=True, type=Path,
                   help='Directory containing dynamics/ and kinematics/ packages')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--mass', type=float, default=3.)
    p.add_argument('--dt', type=float, default=0.002,
                   help='Integration step [s], default 2 ms')
    p.add_argument('--control-dt', type=float, default=0.020,
                   help='Zero-order hold control period [s]')
    p.add_argument('--omega', default='8,10,12,14,16',
                   help='Comma-separated natural frequencies, rad/s')
    p.add_argument('--zeta', default='0.7,0.85,1.0,1.15,1.3',
                   help='Comma-separated damping ratios')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--max-error-deg', type=float, default=30.)
    p.add_argument('--max-speed-rad-s', type=float, default=20.)
    p.add_argument('--max-tool-rmse-mm', type=float, default=5.,
                   help='Screening threshold, NOT physical safety guarantee')
    p.add_argument('--no-all-trajectories', action='store_true',
                   help='Save full trajectories only for selected candidates')
    return p.parse_args()


def parse_values(s: str, label: str) -> list[float]:
    try:
        values = sorted(set(float(x.strip()) for x in s.split(',')))
    except ValueError as exc:
        raise ValueError(f'Invalid {label}: {s}') from exc
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f'{label} values must be finite and positive')
    return values


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    omega: float
    zeta: float
    kp: float
    kd: float


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 121:
        raise ValueError(f'Expected 121 reference samples, found {len(rows)}')
    t = np.array([float(r['t_s']) for r in rows])
    def six(prefix: str, suffix: str) -> np.ndarray:
        return np.array([[float(r[f'{prefix}{j}{suffix}']) for j in range(1, 7)]
                         for r in rows], dtype=np.float64)
    q, qd, qdd = six('q', '_rad'), six('qdot', '_rad_s'), six('qddot', '_rad_s2')
    if not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-8 or abs(t[-1] - 3.) > 1e-8:
        raise ValueError('Reference must be strictly increasing from 0 to 3 s')
    if not all(np.isfinite(x).all() for x in (t, q, qd, qdd)):
        raise ValueError('Reference contains non-finite values')
    return t, q, qd, qdd


def interp6(t: float, ts: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, ts, y[:, j]) for j in range(JOINTS)])


def write_csv(path: Path, data: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def simulate(candidate: Candidate, ts: np.ndarray, qrefs: np.ndarray,
             qdrefs: np.ndarray, ff: np.ndarray, duration: float,
             dt: float, control_dt: float, max_error_deg: float,
             max_speed: float, dynamics: tuple, tool_fn) -> tuple[dict, list[dict]]:
    mass_matrix, coriolis, gravity = dynamics
    num_steps = round(duration / dt)
    hold_n = round(control_dt / dt)
    q = qrefs[0].copy() + INITIAL_ERROR_RAD
    qd = qdrefs[0].copy()
    tau = np.zeros(JOINTS)
    tau_ff = np.zeros(JOINTS)
    tau_fb = np.zeros(JOINTS)
    raw = np.zeros(JOINTS)
    records = []
    failure = ''
    previous_tau = None
    sum_tool_sq = 0.
    sum_q_sq = 0.
    sum_qd_sq = 0.
    sum_tau_sq = np.zeros(JOINTS)
    max_tau = np.zeros(JOINTS)
    max_tau_delta = np.zeros(JOINTS)
    max_q_error = np.zeros(JOINTS)
    saturated_ticks = 0
    sample_count = 0

    for step in range(num_steps + 1):
        t = step * dt
        qr = interp6(t, ts, qrefs)
        qdr = interp6(t, ts, qdrefs)
        err = qr - q
        verr = qdr - qd

        if step % hold_n == 0 and step < num_steps:
            # SAME PD structure as validated Isaac PD:
            # FF at reference states + M(actual joint q) * feedback.
            tau_ff = interp6(t, ts, ff)
            M = mass_matrix(q)
            tau_fb = M @ (candidate.kp * err + candidate.kd * verr)
            raw = tau_ff + tau_fb
            tau = np.clip(raw, -TORQUE_LIMITS, TORQUE_LIMITS)
            saturated_ticks += int(np.any(np.abs(raw) > TORQUE_LIMITS + 1e-10))
            if previous_tau is not None:
                max_tau_delta = np.maximum(max_tau_delta, np.abs(tau - previous_tau))
            previous_tau = tau.copy()

        # Actual modeled joint acceleration under the currently held torque.
        # Compute only if state is finite; failed divergent cases terminate below.
        if np.isfinite(q).all() and np.isfinite(qd).all():
            M_plant = mass_matrix(q)
            c = coriolis(q, qd)
            g = gravity(q)
            qdd_actual = np.linalg.solve(M_plant, tau - c - g)
        else:
            qdd_actual = np.full(JOINTS, np.nan)

        tool = tool_fn(q)
        tool_r = tool_fn(qr)
        tool_error = 1000. * float(np.linalg.norm(tool - tool_r))
        q_error_deg = np.rad2deg(q - qr)
        qd_error_deg = np.rad2deg(qd - qdr)
        max_q_error = np.maximum(max_q_error, np.abs(q_error_deg))
        max_tau = np.maximum(max_tau, np.abs(tau))
        sum_tool_sq += tool_error**2
        sum_q_sq += float(np.sum(q_error_deg**2))
        sum_qd_sq += float(np.sum(qd_error_deg**2))
        sum_tau_sq += tau**2
        sample_count += 1

        r = {'t_s': t, 'tool_error_mm': tool_error, 'saturated': int(np.any(np.abs(raw) > TORQUE_LIMITS + 1e-10))}
        for j in range(JOINTS):
            k = j + 1
            r.update({f'q{k}_rad': q[j], f'qd{k}_rad_s': qd[j], f'qdd{k}_rad_s2': qdd_actual[j],
                      f'qref{k}_rad': qr[j], f'qdref{k}_rad_s': qdr[j],
                      f'joint_error{k}_deg': q_error_deg[j],
                      f'tau{k}_Nm': tau[j], f'tau_ff{k}_Nm': tau_ff[j],
                      f'tau_fb{k}_Nm': tau_fb[j], f'tau_raw{k}_Nm': raw[j]})
        for j, xyz in enumerate('xyz'):
            r[f'tool_{xyz}_m'] = tool[j]
            r[f'toolref_{xyz}_m'] = tool_r[j]
        records.append(r)

        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            failure = 'NONFINITE_STATE'
            break
        if np.max(np.abs(q_error_deg)) > max_error_deg:
            failure = 'JOINT_ERROR_LIMIT'
            break
        if np.max(np.abs(qd)) > max_speed:
            failure = 'JOINT_VELOCITY_LIMIT'
            break
        if step == num_steps:
            break

        # Semi-implicit Euler: a simple, auditable integrator.
        # Plant evolves at dt; control torque is held for control_dt.
        qd = qd + dt * qdd_actual
        q = q + dt * qd

    final = records[-1]
    completed = not failure and len(records) == num_steps + 1
    result = {
        **asdict(candidate), 'completed': int(completed),
        'failure_reason': failure,
        'last_time_s': final['t_s'], 'samples': sample_count,
        'tool_rmse_mm': math.sqrt(sum_tool_sq / sample_count),
        'joint_rmse_deg': math.sqrt(sum_q_sq / (sample_count * JOINTS)),
        'velocity_rmse_deg_s': math.sqrt(sum_qd_sq / (sample_count * JOINTS)),
        'final_tool_error_mm': final['tool_error_mm'],
        'final_joint_velocity_norm_deg_s': float(np.linalg.norm(np.rad2deg(qd))),
        'max_joint_error_deg': float(np.max(max_q_error)),
        'max_torque_Nm': float(np.max(max_tau)),
        'saturation_control_updates': saturated_ticks,
    }
    for j in range(JOINTS):
        k = j + 1
        result[f'joint{k}_peak_error_deg'] = max_q_error[j]
        result[f'joint{k}_peak_torque_Nm'] = max_tau[j]
        result[f'joint{k}_torque_rms_Nm'] = math.sqrt(sum_tau_sq[j] / sample_count)
        result[f'joint{k}_max_delta_tau_per_control_Nm'] = max_tau_delta[j]
    return result, records


def main() -> int:
    a = args_parser()
    if not a.model_dir.is_dir():
        raise FileNotFoundError(a.model_dir)
    if not a.reference.is_file():
        raise FileNotFoundError(a.reference)
    if a.mass <= 0 or a.dt <= 0 or a.control_dt <= 0:
        raise ValueError('Mass and time steps must be positive')
    if abs(3. / a.dt - round(3. / a.dt)) > 1e-8:
        raise ValueError('3 seconds must be divisible by --dt')
    if abs(a.control_dt / a.dt - round(a.control_dt / a.dt)) > 1e-8:
        raise ValueError('--control-dt must be a multiple of --dt')
    if a.top_k <= 0:
        raise ValueError('--top-k must be positive')
    if str(a.model_dir) not in sys.path:
        sys.path.insert(0, str(a.model_dir))
    from dynamics.payload_dynamics import (mass_matrix_payload,
        coriolis_centrifugal_vector_payload, gravity_vector_payload)
    from kinematics.forward_kinematics import forward_kinematics

    def M(q):
        return mass_matrix_payload(q, payload_mass=a.mass, payload_com_tool=COM_TOOL)
    def C(q, qd):
        return coriolis_centrifugal_vector_payload(q, qd, payload_mass=a.mass,
                                                   payload_com_tool=COM_TOOL)
    def G(q):
        return gravity_vector_payload(q, payload_mass=a.mass, payload_com_tool=COM_TOOL)
    def tool(q):
        return BASE_TO_WORLD @ forward_kinematics(q)[:3, 3]

    t, q, qd, qdd = load_reference(a.reference)
    print(f'Precomputing {len(t)} reference feedforward torques at mass={a.mass:g} kg ...', flush=True)
    ff = np.array([M(q[i]) @ qdd[i] + C(q[i], qd[i]) + G(q[i]) for i in range(len(t))])
    if not np.isfinite(ff).all():
        raise RuntimeError('Non-finite feedforward torque')
    omega = parse_values(a.omega, 'omega')
    zeta = parse_values(a.zeta, 'zeta')
    candidates = [Candidate(f'G{n:03d}', w, z, w*w, 2*z*w)
                  for n, (w, z) in enumerate(((w, z) for w in omega for z in zeta), start=1)]
    # Existing baseline always participates for traceability (unless already present).
    if not any(abs(c.kp - 144.) < 1e-10 and abs(c.kd - 24.) < 1e-10 for c in candidates):
        candidates.insert(0, Candidate('BASELINE', 12., 1., 144., 24.))
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise RuntimeError(f'Output directory must be empty or new: {a.output_dir}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = a.output_dir / 'trajectories'
    traj_dir.mkdir()
    all_results = []
    selected_records = {}
    for n, candidate in enumerate(candidates, start=1):
        result, trajectory = simulate(candidate, t, q, qd, ff, 3., a.dt,
            a.control_dt, a.max_error_deg, a.max_speed_rad_s, (M, C, G), tool)
        all_results.append(result)
        if not a.no_all_trajectories:
            write_csv(traj_dir / f'{candidate.candidate_id}.csv', trajectory, list(trajectory[0]))
        else:
            selected_records[candidate.candidate_id] = trajectory
        print(f'[{n:3d}/{len(candidates)}] {candidate.candidate_id:9s} '
              f'Kp={candidate.kp:7.1f} Kd={candidate.kd:6.2f} '
              f'ToolRMSE={result["tool_rmse_mm"]:7.3f}mm '
              f'completed={result["completed"]}', flush=True)

    # Feasibility FIRST; ranking is a transparent screening heuristic, not a proof of optimality.
    feasible = [r for r in all_results if r['completed']
                and r['saturation_control_updates'] == 0
                and r['tool_rmse_mm'] <= a.max_tool_rmse_mm]
    feasible.sort(key=lambda r: (r['tool_rmse_mm'],
                     sum(r[f'joint{j}_max_delta_tau_per_control_Nm'] for j in range(1, 7)),
                     r['final_tool_error_mm']))
    selected = feasible[:a.top_k]
    selected_ids = {r['candidate_id'] for r in selected}
    for r in all_results:
        r['feasible_screen'] = int(r in feasible)
        r['selected_for_physx'] = int(r['candidate_id'] in selected_ids)
    if a.no_all_trajectories:
        for r in selected:
            cid = r['candidate_id']
            rows = selected_records[cid]
            write_csv(traj_dir / f'{cid}.csv', rows, list(rows[0]))
    columns = list(all_results[0])
    write_csv(a.output_dir / 'gain_search_summary.csv', all_results, columns)
    write_csv(a.output_dir / 'selected_candidates.csv', selected, columns)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reference': str(a.reference), 'model_dir': str(a.model_dir),
        'mass_kg': a.mass, 'payload_com_tool_m': COM_TOOL.tolist(),
        'integration': 'semi_implicit_euler', 'physics_dt_s': a.dt,
        'controller_dt_s': a.control_dt,
        'total_duration_s': 3., 'initial_joint_error_rad': INITIAL_ERROR_RAD.tolist(),
        'torque_limits_Nm': TORQUE_LIMITS.tolist(),
        'controller': 'ref_trajectory_feedforward + M(q)*(Kp*error + Kd*velocity_error)',
        'search_omega_rad_s': omega, 'search_zeta': zeta,
        'candidate_count': len(candidates),
        'feasibility': {'complete': True, 'saturation_control_updates': 0,
                        'max_tool_rmse_mm': a.max_tool_rmse_mm},
        'selected_ids': [r['candidate_id'] for r in selected],
        'limitations': ['Mathematical plant uses same nominal 3kg model as controller',
                        'No collisions, friction, actuator lag or ROS delay modeled',
                        'Tool0 position is FK-derived, not independent PhysX measurement',
                        'Offline feasibility is NOT a physical collision/safety test'],
    }
    (a.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'\nCompleted {len(candidates)} candidates; {len(feasible)} feasible; '
          f'{len(selected)} selected. Results: {a.output_dir}')
    if not selected:
        print('WARNING: No feasible candidate under current screening criteria.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
