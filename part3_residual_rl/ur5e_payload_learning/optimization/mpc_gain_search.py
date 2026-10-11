#!/usr/bin/env python3
"""Offline UR5e 3 kg MPC parameter search. NO Isaac Sim / ROS execution.

Uses the SAME nonlinear plant, reference, 2 ms integration, 20 ms control and
initial error as the existing PD gain-search. Full finite-difference dynamics
linearization around reference (including M, Coriolis and gravity effects).

Dependencies: numpy, scipy, osqp; existing UR5e dynamics/kinematics modules.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse

NJ = 6
TORQUE_LIMITS = np.array([150., 150., 150., 28., 28., 28.])
INITIAL_ERROR = np.deg2rad([0., .5, -.5, 0., 0., 0.])
COM_TOOL = np.array([0., 0., .025])
BASE_TO_WORLD = np.diag([-1., -1., 1.])
BASE_Q = np.array([120., 220., 180., 100., 80., 80., 4., 6., 5., 3., 2., 2.])
BASE_R = np.array([.025, .025, .025, .04, .04, .04])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--mass', type=float, default=3.)
    p.add_argument('--dt', type=float, default=.002)
    p.add_argument('--control-dt', type=float, default=.020)
    p.add_argument('--horizons', default='5,10,15')
    p.add_argument('--q-scales', default='0.5,1,2')
    p.add_argument('--r-scales', default='0.25,1')
    p.add_argument('--terminal-scale', type=float, default=2.)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--max-tool-rmse-mm', type=float, default=5.)
    p.add_argument('--max-error-deg', type=float, default=30.)
    p.add_argument('--max-speed-rad-s', type=float, default=20.)
    p.add_argument('--no-all-trajectories', action='store_true')
    return p.parse_args()


def numbers(raw, integer=False):
    vals = sorted(set((int if integer else float)(part) for part in raw.split(',')))
    if not vals or any(not np.isfinite(v) or v <= 0 for v in vals):
        raise ValueError('Search parameters must be positive and finite')
    return vals


def load_reference(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 121:
        raise ValueError(f'Expected 121 samples; found {len(rows)}')
    t = np.array([float(row['t_s']) for row in rows])
    def arr(prefix, suffix):
        return np.array([[float(row[f'{prefix}{j}{suffix}']) for j in range(1, 7)]
                         for row in rows])
    q = arr('q', '_rad')
    qd = arr('qdot', '_rad_s')
    qdd = arr('qddot', '_rad_s2')
    if abs(t[0]) > 1e-8 or abs(t[-1]-3.) > 1e-8 or not np.all(np.diff(t)>0):
        raise ValueError('Reference must run from t=0 to t=3 seconds')
    if not all(np.isfinite(v).all() for v in (t, q, qd, qdd)):
        raise ValueError('Nonfinite reference')
    return t, q, qd, qdd


class Model:
    def __init__(self, mass, module_dir):
        sys.path.insert(0, str(module_dir))
        from dynamics.payload_dynamics import (mass_matrix_payload,
            coriolis_centrifugal_vector_payload, gravity_vector_payload)
        from kinematics.forward_kinematics import forward_kinematics
        self.mass = mass
        self._M = mass_matrix_payload
        self._C = coriolis_centrifugal_vector_payload
        self._G = gravity_vector_payload
        self._FK = forward_kinematics

    def M(self, q):
        return self._M(q, payload_mass=self.mass, payload_com_tool=COM_TOOL)

    def C(self, q, qd):
        return self._C(q, qd, payload_mass=self.mass, payload_com_tool=COM_TOOL)

    def G(self, q):
        return self._G(q, payload_mass=self.mass, payload_com_tool=COM_TOOL)

    def acceleration(self, q, qd, torque):
        return np.linalg.solve(self.M(q), torque - self.C(q, qd) - self.G(q))

    def tool(self, q):
        return BASE_TO_WORLD @ self._FK(q)[:3, 3]


class Reference:
    def __init__(self, t, q, qd, qdd, ff, dt, count, end_hold):
        self.t, self.q, self.qd, self.qdd, self.ff = t, q, qd, qdd, ff
        self._end_hold = end_hold
        # Control grid extends past 3s so the horizon predicts final-position HOLD.
        self.grid = np.arange(count + 1, dtype=float) * dt
        self.x = np.zeros((count + 1, 12))
        self.u = np.zeros((count + 1, NJ))
        for k, tk in enumerate(self.grid):
            self.x[k, :6] = self.interp(q, tk)
            if tk < 3. - 1e-10:
                self.x[k, 6:] = self.interp(qd, tk)
                self.u[k] = self.interp(ff, tk)
            else:
                # HOLD acceleration=0, velocity=0 at and beyond the endpoint.
                self.x[k, 6:] = 0.
                self.u[k] = self.end_hold

    def interp(self, arr, t):
        tk = min(max(float(t), 0.), 3.)
        return np.array([np.interp(tk, self.t, arr[:, j]) for j in range(6)])

    @property
    def end_hold(self):
        return self._end_hold

    @end_hold.setter
    def end_hold(self, value):
        self._end_hold = value


def predict_linearization(model, xref, ff, dt, eps=1e-5):
    q = xref[:6]
    qd = xref[6:]
    a0 = model.acceleration(q, qd, ff)
    dq = np.zeros((NJ, NJ))
    dv = np.zeros((NJ, NJ))
    for j in range(NJ):
        perturb = np.zeros(NJ)
        perturb[j] = eps
        dq[:, j] = (model.acceleration(q+perturb, qd, ff)
                     - model.acceleration(q-perturb, qd, ff)) / (2*eps)
        dv[:, j] = (model.acceleration(q, qd+perturb, ff)
                     - model.acceleration(q, qd-perturb, ff)) / (2*eps)
    inv_mass = np.linalg.solve(model.M(q), np.eye(NJ))
    # Semi-implicit Euler linearization to match the offline plant integrator.
    A = np.eye(12)
    A[:6, :6] += dt*dt*dq
    A[:6, 6:] = dt*np.eye(NJ) + dt*dt*dv
    A[6:, :6] = dt*dq
    A[6:, 6:] += dt*dv
    B = np.zeros((12, NJ))
    B[:6, :] = dt*dt*inv_mass
    B[6:, :] = dt*inv_mass
    phi = np.r_[q + dt*(qd+dt*a0), qd + dt*a0]
    return A, B, phi


def prediction_bank(model, reference, control_dt, total_steps, max_h):
    bank = []
    print(f'Building {total_steps+max_h} full-dynamics linearizations...', flush=True)
    for k in range(total_steps + max_h):
        A, B, nominal_next = predict_linearization(
            model, reference.x[k], reference.u[k], control_dt)
        defect = nominal_next - reference.x[k+1]
        bank.append((A, B, defect))
        if (k+1) % 25 == 0:
            print(f'  linearized {k+1}/{total_steps+max_h}', flush=True)
    return bank


def build_prediction(bank, ref, step, horizon, initial_error):
    f = initial_error.copy()
    influence = np.zeros((12, 6*horizon))
    free = np.empty(12*horizon)
    S = np.empty((12*horizon, 6*horizon))
    feedforward = np.empty(6*horizon)
    for h in range(horizon):
        A, B, defect = bank[step+h]
        f = A@f + defect
        influence = A@influence
        influence[:, 6*h:6*(h+1)] += B
        free[12*h:12*(h+1)] = f
        S[12*h:12*(h+1)] = influence
        feedforward[6*h:6*(h+1)] = ref.u[step+h]
    return free, S, feedforward


def mpc_torque(model, bank, ref, step, x, candidate, terminal_scale):
    h = candidate['horizon']
    error = x - ref.x[step]
    free, S, FF = build_prediction(bank, ref, step, h, error)
    q_diag = BASE_Q * candidate['q_scale']
    # Terminal objective is the final horizon state, not an extra time step.
    weights = np.tile(q_diag, h)
    weights[-12:] *= terminal_scale
    r_diag = np.tile(BASE_R*candidate['r_scale'], h)
    H = S.T @ (weights[:, None] * S)
    H.flat[::H.shape[0]+1] += r_diag
    g = S.T @ (weights * free)
    caps = np.tile(TORQUE_LIMITS, h)
    t0 = time.perf_counter()
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(2*H), q=2*g,
                 A=sparse.eye(6*h, format='csc'),
                 l=-caps-FF, u=caps-FF,
                 verbose=False, polishing=False,
                 eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
    sol = solver.solve()
    millis = (time.perf_counter()-t0)*1000.
    if not sol.info.status.lower().startswith('solved') or sol.x is None:
        raise RuntimeError(f'OSQP {sol.info.status} at t={step*.02:.3f}s')
    correction = np.asarray(sol.x[:6])
    raw = ref.u[step] + correction
    clipped = np.clip(raw, -TORQUE_LIMITS, TORQUE_LIMITS)
    return clipped, ref.u[step].copy(), correction, millis, int(sol.info.iter), raw


def save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def simulate(model, ref, bank, candidate, dt, control_dt, total_steps,
             max_error_deg, max_speed, terminal_scale):
    n_sub = round(control_dt/dt)
    q = ref.x[0, :6].copy() + INITIAL_ERROR
    qd = ref.x[0, 6:].copy()
    tau = np.zeros(6)
    ff = np.zeros(6)
    corr = np.zeros(6)
    raw = np.zeros(6)
    logs = []
    prior_tau = None
    max_delta_tau = np.zeros(6)
    sat_updates = 0
    mpc_ms = []
    failure = ''
    for step in range(total_steps+1):
        t = step*dt
        k = step//n_sub
        qr = ref.x[k, :6] if step%n_sub==0 else ref.interp(ref.q, t)
        qdr = ref.x[k, 6:] if step%n_sub==0 else ref.interp(ref.qd, t)
        if step%n_sub==0 and step<total_steps:
            try:
                tau, ff, corr, ms, iters, raw = mpc_torque(
                    model, bank, ref, k, np.r_[q, qd], candidate, terminal_scale)
            except Exception as exc:
                failure = f'MPC_SOLVER_ERROR: {exc}'
                break
            mpc_ms.append(ms)
            sat_updates += int(np.any(np.abs(raw)>TORQUE_LIMITS+1e-8))
            if prior_tau is not None:
                max_delta_tau = np.maximum(max_delta_tau, np.abs(tau-prior_tau))
            prior_tau = tau.copy()
        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            failure = 'NONFINITE_STATE'
            break
        qdd_actual = model.acceleration(q, qd, tau)
        # Interpolate reference at exact physics step, as in PD gain search.
        qr = ref.interp(ref.q, t)
        qdr = ref.interp(ref.qd, t)
        terr = 1000.*np.linalg.norm(model.tool(q)-model.tool(qr))
        eq = np.rad2deg(q-qr)
        ev = np.rad2deg(qd-qdr)
        rec = {'t_s':t, 'tool_error_mm':terr,
               'saturated':int(np.any(np.abs(raw)>TORQUE_LIMITS+1e-8))}
        for j in range(NJ):
            i=j+1
            rec.update({f'q{i}_rad':q[j],f'qd{i}_rad_s':qd[j],
                        f'qdd{i}_rad_s2':qdd_actual[j],f'qref{i}_rad':qr[j],
                        f'qdref{i}_rad_s':qdr[j],f'joint_error{i}_deg':eq[j],
                        f'tau{i}_Nm':tau[j],f'tau_ff{i}_Nm':ff[j],
                        f'tau_correction{i}_Nm':corr[j],f'tau_raw{i}_Nm':raw[j]})
        for j,c in enumerate('xyz'):
            rec[f'tool_{c}_m'] = model.tool(q)[j]
            rec[f'toolref_{c}_m'] = model.tool(qr)[j]
        logs.append(rec)
        if np.max(np.abs(eq))>max_error_deg:
            failure='JOINT_ERROR_LIMIT'
            break
        if np.max(np.abs(qd))>max_speed:
            failure='JOINT_VELOCITY_LIMIT'
            break
        if step==total_steps:
            break
        qd = qd+dt*qdd_actual
        q = q+dt*qd
    if not logs:
        return {'**error':'no-records','completed':0,'failure_reason':failure}, []
    e = np.array([[r[f'joint_error{i}_deg'] for i in range(1,7)] for r in logs])
    e_v = np.array([[np.rad2deg(r[f'qd{i}_rad_s']-r[f'qdref{i}_rad_s']) for i in range(1,7)] for r in logs])
    torques = np.array([[r[f'tau{i}_Nm'] for i in range(1,7)] for r in logs])
    tool = np.array([r['tool_error_mm'] for r in logs])
    final_qd = np.array([logs[-1][f'qd{i}_rad_s'] for i in range(1,7)])
    complete = int(not failure and len(logs)==total_steps+1)
    summary = {**candidate, 'completed':complete, 'failure_reason':failure,
               'last_time_s':logs[-1]['t_s'], 'samples':len(logs),
               'tool_rmse_mm':float(np.sqrt(np.mean(tool**2))),
               'joint_rmse_deg':float(np.sqrt(np.mean(e**2))),
               'velocity_rmse_deg_s':float(np.sqrt(np.mean(e_v**2))),
               'final_tool_error_mm':float(tool[-1]),
               'final_joint_velocity_norm_deg_s':float(np.linalg.norm(np.rad2deg(final_qd))),
               'max_joint_error_deg':float(np.max(abs(e))),
               'max_torque_Nm':float(np.max(abs(torques))),
               'saturation_control_updates':sat_updates,
               'mpc_compute_avg_ms':float(np.mean(mpc_ms)) if mpc_ms else np.nan,
               'mpc_compute_max_ms':float(np.max(mpc_ms)) if mpc_ms else np.nan,
               'mpc_updates_over_20ms':int(np.sum(np.array(mpc_ms)>20.)),
               'control_updates':len(mpc_ms)}
    for j in range(6):
        i=j+1
        summary[f'joint{i}_peak_error_deg']=float(np.max(abs(e[:,j])))
        summary[f'joint{i}_peak_torque_Nm']=float(np.max(abs(torques[:,j])))
        summary[f'joint{i}_torque_rms_Nm']=float(np.sqrt(np.mean(torques[:,j]**2)))
        summary[f'joint{i}_max_delta_tau_per_control_Nm']=float(max_delta_tau[j])
    return summary, logs


def main():
    a=parser()
    if not a.reference.is_file() or not a.model_dir.is_dir():
        raise FileNotFoundError('Reference or model directory not found')
    if any(x<=0 for x in (a.mass,a.dt,a.control_dt,a.top_k,a.terminal_scale)):
        raise ValueError('Mass, time steps, top-k and terminal scale must be positive')
    if abs(3/a.dt-round(3/a.dt))>1e-8 or abs(a.control_dt/a.dt-round(a.control_dt/a.dt))>1e-8:
        raise ValueError('3s must be divisible by dt; control-dt must be a multiple of dt')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f'Nonempty output dir: {a.output_dir}')
    horizons=numbers(a.horizons,integer=True)
    q_scales=numbers(a.q_scales)
    r_scales=numbers(a.r_scales)
    nsteps=round(3/a.dt)
    nctrl=round(3/a.control_dt)
    model=Model(a.mass,a.model_dir)
    t,q,qd,qdd=load_reference(a.reference)
    print('Precomputing 3kg reference inverse dynamics...',flush=True)
    ff=np.array([model.M(q[i])@qdd[i]+model.C(q[i],qd[i])+model.G(q[i])
                 for i in range(len(t))])
    ref=Reference(t,q,qd,qdd,ff,a.control_dt,nctrl+max(horizons),model.G(q[-1]))
    bank=prediction_bank(model,ref,a.control_dt,nctrl,max(horizons))
    candidates=[]
    for horizon in horizons:
        for qs in q_scales:
            for rs in r_scales:
                candidates.append({'candidate_id':f'M{len(candidates)+1:03d}',
                                   'horizon':horizon,'q_scale':qs,'r_scale':rs})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    traj_dir=a.output_dir/'trajectories'
    traj_dir.mkdir()
    results=[]
    kept={}
    start=time.perf_counter()
    for n,candidate in enumerate(candidates,1):
        result, log=simulate(model,ref,bank,candidate,a.dt,a.control_dt,nsteps,
                             a.max_error_deg,a.max_speed_rad_s,a.terminal_scale)
        results.append(result)
        if a.no_all_trajectories:
            kept[candidate['candidate_id']]=log
        elif log:
            save_csv(traj_dir/f'{candidate["candidate_id"]}.csv',log)
        print(f'[{n:02d}/{len(candidates)}] {candidate["candidate_id"]} '
              f'N={candidate["horizon"]} Qx={candidate["q_scale"]:g} '
              f'Rx={candidate["r_scale"]:g} RMSE={result.get("tool_rmse_mm",np.nan):.3f} mm '
              f'Completed={result["completed"]}',flush=True)
    feasible=[r for r in results if r['completed']
              and r['saturation_control_updates']==0
              and r['tool_rmse_mm']<=a.max_tool_rmse_mm]
    feasible.sort(key=lambda r:(r['tool_rmse_mm'],
                r['final_tool_error_mm'],r['mpc_compute_avg_ms']))
    selected=feasible[:a.top_k]
    selected_ids={r['candidate_id'] for r in selected}
    for r in results:
        r['feasible_screen']=int(r in feasible)
        r['selected_for_review']=int(r['candidate_id'] in selected_ids)
    if a.no_all_trajectories:
        for r in selected:
            log=kept[r['candidate_id']]
            if log:
                save_csv(traj_dir/f'{r["candidate_id"]}.csv',log)
    save_csv(a.output_dir/'mpc_search_summary.csv',results)
    if selected:
        save_csv(a.output_dir/'selected_candidates.csv',selected)
    else:
        # Headers still provided, empty result is explicit.
        with open(a.output_dir/'selected_candidates.csv','w',newline='') as f:
            csv.writer(f).writerow(list(results[0]))
    meta={'created_utc':datetime.now(timezone.utc).isoformat(),
          'algorithm':'offline linear time-varying MPC with OSQP, full nonlinear plant',
          'reference':str(a.reference),'model_dir':str(a.model_dir),
          'mass_kg':a.mass,'dt_physics_s':a.dt,'control_dt_s':a.control_dt,
          'initial_joint_error_deg':np.rad2deg(INITIAL_ERROR).tolist(),
          'torque_limit_Nm':TORQUE_LIMITS.tolist(),
          'linearization':'full numerical Jacobian of nonlinear joint acceleration',
          'prediction_discretization':'semi-implicit Euler, control_dt',
          'plant_discretization':'semi-implicit Euler, physics_dt',
          'prediction_model':'reference-centric, feedforward nominal with affine defect',
          'terminal_hold':'q=q_final, qd=0, gravity feedforward',
          'Q_base_diagonal':BASE_Q.tolist(),'R_base_diagonal':BASE_R.tolist(),
          'terminal_multiplier':a.terminal_scale,'search_horizons':horizons,
          'search_q_scales':q_scales,'search_r_scales':r_scales,
          'n_candidates':len(candidates),'selected_ids':[r['candidate_id'] for r in selected],
          'elapsed_s':time.perf_counter()-start,
          'limits':['No PhysX/Isaac/ROS/contacts','Plant uses nominal model',
                    'OSQP wall time includes Python matrix setup, depends on machine',
                    'Tool position is FK-derived, not independent measurement',
                    'No real hardware safety certification']}
    (a.output_dir/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'\nDONE. Candidates={len(candidates)}, feasible={len(feasible)}, '
          f'selected={len(selected)}, elapsed={meta["elapsed_s"]:.1f}s')
    print(f'CSV: {a.output_dir/"mpc_search_summary.csv"}')


if __name__=='__main__':
    main()
