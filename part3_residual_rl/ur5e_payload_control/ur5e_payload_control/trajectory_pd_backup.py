
#!/usr/bin/env python3

"""UR5e ROS2 G025: PRELOAD -> RETURN -> STABILIZE -> TRACKING -> FINAL_HOLD."""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MODEL_DIR = (
    "/home/sooyong/robotics_ws/src/"
    "ur5e_control_analysis/part1_modeling_control_analysis"
)

REFERENCE = (
    "/home/sooyong/Documents/projects/"
    "12-rl-isaac-ur5e/references/"
    "inverse_dynamics_1kg_cube_3s_retimed_v2.csv"
)

OUTPUT_DIR = (
    "/home/sooyong/Documents/projects/"
    "12-rl-isaac-ur5e/results/ros2_physx_pd"
)

MASS = 3.0
COM = np.array([0.0, 0.0, 0.025])

KP = 256.0
KD = 41.6
DT = 0.02

LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

OFFSET = np.deg2rad([
    0.0, 0.5, -0.5, 0.0, 0.0, 0.0,
])

POS_TOL_DEG = 0.05
VEL_TOL_DEG_S = 0.30
TOOL_Z_TOL_MM = 1.0
TOOL_XY_TOL_MM = 2.0

STABLE_SECONDS = 2.0
STATE_TIMEOUT = 0.15


def load_reference():

    with open(REFERENCE, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 121:
        raise ValueError(
            f"Expected 121 reference points; got {len(rows)}"
        )

    t = np.array([
        float(r["t_s"]) for r in rows
    ], dtype=float)

    def load(prefix, suffix):
        return np.array([
            [
                float(r[f"{prefix}{j}{suffix}"])
                for j in range(1, 7)
            ]
            for r in rows
        ], dtype=float)

    q = load("q", "_rad")
    qd = load("qdot", "_rad_s")
    qdd = load("qddot", "_rad_s2")

    if not (
        np.isclose(t[0], 0.0)
        and np.isclose(t[-1], 3.0)
        and np.all(np.diff(t) > 0.0)
    ):
        raise ValueError(
            "Reference time must increase from 0 to 3 seconds"
        )

    if not all(
        np.isfinite(a).all()
        for a in (t, q, qd, qdd)
    ):
        raise ValueError("Nonfinite reference data")

    return t, q, qd, qdd


class TrajectoryPD(Node):

    def __init__(self):

        super().__init__("ur5e_trajectory_pd")

        sys.path.insert(0, MODEL_DIR)

        from dynamics.payload_dynamics import (
            mass_matrix_payload,
            gravity_vector_payload,
            coriolis_centrifugal_vector_payload,
        )

        from kinematics.forward_kinematics import (
            forward_kinematics,
        )

        self.m_fn = mass_matrix_payload
        self.g_fn = gravity_vector_payload
        self.c_fn = coriolis_centrifugal_vector_payload
        self.fk_fn = forward_kinematics

        (
            self.tref,
            self.qref,
            self.vref,
            self.aref,
        ) = load_reference()

        self.q0 = self.qref[0] + OFFSET
        self.qfinal = self.qref[-1].copy()
        self.duration = float(self.tref[-1])

        self.tool0 = self.tool_xyz(self.q0)

        self.get_logger().info(
            "Computing 3 kg inverse-dynamics feedforward..."
        )

        self.ffref = np.array([
            self.mass(q) @ a
            + self.coriolis(q, v)
            + self.gravity(q)
            for q, v, a in zip(
                self.qref,
                self.vref,
                self.aref,
            )
        ])

        self.preload = self.gravity(self.q0, payload_mass=1.0)

        if not (
            np.isfinite(self.ffref).all()
            and np.isfinite(self.preload).all()
        ):
            raise ValueError("Invalid feedforward torque")

        self.q = None
        self.qd = None
        self.sim_stamp = None
        self.recv_wall = None
        self.prev_stamp = None

        self.phase = "PRELOAD"

        self.armed = False
        self.stable_since = None
        self.start_stamp = None

        self.last_report = 0.0

        self.rows = []
        self.saved = False
        self.last_logged_stamp = None

        self.pub = self.create_publisher(
            JointState,
            "/joint_command",
            10,
        )

        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_state,
            10,
        )

        self.srv = self.create_service(
            Trigger,
            "/ur5e/start_trajectory",
            self.on_start,
        )

        self.timer = self.create_timer(
            DT,
            self.tick,
        )

        self.get_logger().info(
            "G025 | model 3kg, physical 1kg | Kp=256 Kd=41.6 | 50 Hz requested"
        )

        self.get_logger().info(
            "Initial joints (deg): "
            + str(np.round(np.rad2deg(self.q0), 4))
        )

        self.get_logger().info(
            "Expected FK Tool0 (base frame, m): "
            + str(np.round(self.tool0, 5))
        )

        self.get_logger().warning(
            "Start node BEFORE Isaac Sim Play. "
            "START service enables RETURN -> "
            "2s STABILIZE -> TRACKING."
        )

    def mass(self, q):

        return np.asarray(
            self.m_fn(
                q,
                payload_mass=MASS,
                payload_com_tool=COM,
            ),
            dtype=float,
        )

    def gravity(self, q, payload_mass=MASS):

        return np.asarray(
            self.g_fn(
                q,
                payload_mass=payload_mass,
                payload_com_tool=COM,
            ),
            dtype=float,
        )

    def coriolis(self, q, v):

        return np.asarray(
            self.c_fn(
                q,
                v,
                payload_mass=MASS,
                payload_com_tool=COM,
            ),
            dtype=float,
        )

    def tool_xyz(self, q):

        T = np.asarray(
            self.fk_fn(q),
            dtype=float,
        )

        if (
            T.shape != (4, 4)
            or not np.isfinite(T).all()
        ):
            raise ValueError(
                "FK must return a finite 4x4 matrix"
            )

        return T[:3, 3].copy()

    def interp(self, t, data):

        return np.array([
            np.interp(t, self.tref, data[:, j])
            for j in range(6)
        ])

    def on_state(self, msg):

        if not all(j in msg.name for j in JOINTS):
            return

        if (
            len(msg.position) != len(msg.name)
            or len(msg.velocity) != len(msg.name)
        ):
            return

        indices = [
            msg.name.index(j)
            for j in JOINTS
        ]

        q = np.array([
            msg.position[i]
            for i in indices
        ], dtype=float)

        v = np.array([
            msg.velocity[i]
            for i in indices
        ], dtype=float)

        stamp = (
            msg.header.stamp.sec
            + 1e-9 * msg.header.stamp.nanosec
        )

        if (
            stamp <= 0.0
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
        ):
            return

        self.q = q
        self.qd = v
        self.sim_stamp = float(stamp)
        self.recv_wall = time.monotonic()

    def fresh(self):

        return (
            self.q is not None
            and self.qd is not None
            and self.sim_stamp is not None
            and self.recv_wall is not None
            and time.monotonic() - self.recv_wall
            <= STATE_TIMEOUT
        )

    def on_start(self, request, response):

        if (
            self.phase != "RETURN"
            or self.armed
            or not self.fresh()
        ):
            response.success = False
            response.message = (
                f"Not ready: phase={self.phase}, "
                f"armed={self.armed}, "
                f"fresh={self.fresh()}"
            )
            return response

        self.armed = True
        self.stable_since = None

        response.success = True
        response.message = (
            "ARMED: RETURN -> 2s STABILIZE -> TRACKING"
        )

        self.get_logger().warning(response.message)

        return response

    def publish(self, tau):

        tau = np.asarray(tau, dtype=float)

        if (
            tau.shape != (6,)
            or not np.isfinite(tau).all()
        ):
            raise ValueError(
                "Nonfinite or invalid torque"
            )

        msg = JointState()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.name = JOINTS.copy()
        msg.position = []
        msg.velocity = []

        msg.effort = np.clip(
            tau,
            -LIMITS,
            LIMITS,
        ).tolist()

        self.pub.publish(msg)

    def save(self):

        if self.saved or not self.rows:
            return

        folder = Path(OUTPUT_DIR)
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            "G025_model3kg_actual1kg_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        path = folder / filename

        with path.open("w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(self.rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(self.rows)

        self.saved = True

        self.get_logger().info(
            f"CSV SAVED: {len(self.rows)} rows -> {path}"
        )

        self.get_logger().info(
            "Logging stopped. "
            "FINAL_HOLD torque control remains active."
        )

    def record(
        self,
        t,
        stamp,
        phase,
        q,
        v,
        qr,
        vr,
        ff,
        fb,
        raw,
        tool,
    ):

        # Important: never record after CSV was saved.
        if self.saved:
            return

        if (
            self.last_logged_stamp is not None
            and stamp <= self.last_logged_stamp + 1e-9
        ):
            return

        tool_ref = self.tool_xyz(qr)

        row = {
            "t_s": float(t),
            "sim_stamp_s": float(stamp),
            "phase": phase,

            "tool_fk_x_m": float(tool[0]),
            "tool_fk_y_m": float(tool[1]),
            "tool_fk_z_m": float(tool[2]),

            "tool_fk_ref_x_m": float(tool_ref[0]),
            "tool_fk_ref_y_m": float(tool_ref[1]),
            "tool_fk_ref_z_m": float(tool_ref[2]),

            "tool_fk_x_error_mm": float(
                1000.0 * (tool[0] - tool_ref[0])
            ),

            "tool_fk_y_error_mm": float(
                1000.0 * (tool[1] - tool_ref[1])
            ),

            "tool_fk_z_error_mm": float(
                1000.0 * (tool[2] - tool_ref[2])
            ),

            "tool_fk_error_norm_mm": float(
                1000.0 * np.linalg.norm(
                    tool - tool_ref
                )
            ),

            "any_torque_saturated": int(
                np.any(np.abs(raw) > LIMITS)
            ),
        }

        for i in range(6):

            j = i + 1

            row.update({
                f"q{j}_ref_rad":
                    float(qr[i]),

                f"q{j}_actual_rad":
                    float(q[i]),

                f"q{j}_error_rad":
                    float(qr[i] - q[i]),

                f"qdot{j}_ref_rad_s":
                    float(vr[i]),

                f"qdot{j}_actual_rad_s":
                    float(v[i]),

                f"tau{j}_ff_Nm":
                    float(ff[i]),

                f"tau{j}_feedback_Nm":
                    float(fb[i]),

                f"tau{j}_raw_Nm":
                    float(raw[i]),

                f"tau{j}_command_Nm":
                    float(np.clip(
                        raw[i],
                        -LIMITS[i],
                        LIMITS[i],
                    )),
            })

        self.rows.append(row)
        self.last_logged_stamp = stamp

    def tick(self):

        wall = time.monotonic()

        # PRELOAD torque before Isaac Play.
        if not self.fresh():

            if self.phase in (
                "TRACKING",
                "FINAL_HOLD",
            ):

                if self.phase == "TRACKING":
                    self.get_logger().error(
                        "Feedback lost during TRACKING. "
                        "Saving partial data."
                    )
                    self.save()

                else:
                    self.get_logger().warning(
                        "Feedback lost during FINAL_HOLD.",
                        throttle_duration_sec=2.0,
                    )

            self.phase = "PRELOAD"
            self.armed = False
            self.stable_since = None
            self.start_stamp = None

            self.publish(self.preload)
            return

        q = self.q.copy()
        v = self.qd.copy()
        stamp = float(self.sim_stamp)

        # Isaac Stop / restart detection.
        if (
            self.prev_stamp is not None
            and stamp < self.prev_stamp - 0.001
        ):

            self.get_logger().warning(
                "Simulation timestamp reset; disarmed"
            )

            if self.phase == "TRACKING":
                self.save()

            self.phase = "PRELOAD"
            self.armed = False
            self.stable_since = None
            self.start_stamp = None

        self.prev_stamp = stamp

        if self.phase == "PRELOAD":

            self.phase = "RETURN"

            self.get_logger().info(
                "Feedback received: RETURN controller active"
            )

        initial_error = float(
            np.max(np.abs(
                np.rad2deg(q - self.q0)
            ))
        )

        initial_speed = float(
            np.max(np.abs(
                np.rad2deg(v)
            ))
        )

        tool = self.tool_xyz(q)

        dx, dy, dz = (
            1000.0 * (tool - self.tool0)
        )

        xy_mm = float(np.hypot(dx, dy))
        z_mm = float(abs(dz))

        stable = (
            initial_error <= POS_TOL_DEG
            and initial_speed <= VEL_TOL_DEG_S
            and xy_mm <= TOOL_XY_TOL_MM
            and z_mm <= TOOL_Z_TOL_MM
        )

        # Initial pose recovery and stabilization.
        if self.phase in (
            "RETURN",
            "STABILIZE",
        ):

            qr = self.q0
            vr = np.zeros(6)
            ff = self.gravity(self.q0, payload_mass=1.0)
            t = 0.0

            if self.armed:

                if stable:

                    if self.stable_since is None:
                        self.stable_since = stamp

                    self.phase = "STABILIZE"

                    if (
                        stamp - self.stable_since
                        >= STABLE_SECONDS
                    ):

                        self.phase = "TRACKING"
                        self.start_stamp = stamp

                        self.rows = []
                        self.saved = False
                        self.last_logged_stamp = None
                        self.armed = False

                        self.get_logger().warning(
                            "RETURN + 2s STABILIZE passed: "
                            "TRACKING STARTED"
                        )

                else:

                    self.stable_since = None

                    if self.phase == "STABILIZE":
                        self.get_logger().warning(
                            "Stability interrupted; timer reset"
                        )

                    self.phase = "RETURN"

        if self.phase == "TRACKING":

            t = max(
                0.0,
                stamp - self.start_stamp,
            )

            if t >= self.duration:

                # End of experiment.
                t = self.duration
                self.phase = "FINAL_HOLD"

                qr = self.qfinal
                vr = np.zeros(6)
                ff = self.gravity(self.qfinal)

                self.get_logger().info(
                    "3s TRACKING completed. "
                    "FINAL_HOLD active."
                )

            else:

                qr = self.interp(t, self.qref)
                vr = self.interp(t, self.vref)
                ff = self.interp(t, self.ffref)

        elif self.phase == "FINAL_HOLD":

            t = self.duration
            qr = self.qfinal
            vr = np.zeros(6)
            ff = self.gravity(self.qfinal)

        # Original computed-torque PD unchanged.
        fb = self.mass(q) @ (
            KP * (qr - q)
            + KD * (vr - v)
        )

        raw = ff + fb

        try:
            self.publish(raw)

        except ValueError as exc:

            self.get_logger().error(str(exc))

            if self.phase == "TRACKING":
                self.save()

            self.phase = "PRELOAD"
            self.armed = False
            return

        # Record only during tracking.
        if self.phase == "TRACKING":

            self.record(
                t=t,
                stamp=stamp,
                phase="TRACKING",
                q=q,
                v=v,
                qr=qr,
                vr=vr,
                ff=ff,
                fb=fb,
                raw=raw,
                tool=tool,
            )

        # Record final state once, save once, then stop logging.
        elif (
            self.phase == "FINAL_HOLD"
            and not self.saved
        ):

            self.record(
                t=self.duration,
                stamp=stamp,
                phase="FINAL_HOLD",
                q=q,
                v=v,
                qr=qr,
                vr=vr,
                ff=ff,
                fb=fb,
                raw=raw,
                tool=tool,
            )

            self.save()

        # Diagnostic logging continues; CSV logging does not.
        if wall - self.last_report >= 1.0:

            self.last_report = wall

            if self.phase in (
                "RETURN",
                "STABILIZE",
            ):

                stable_sec = (
                    0.0
                    if self.stable_since is None
                    else max(
                        0.0,
                        stamp - self.stable_since,
                    )
                )

                self.get_logger().info(
                    f"phase={self.phase} | "
                    f"armed={self.armed} | "
                    f"stable={stable_sec:.2f}/2s | "
                    f"joint_err={initial_error:.4f}deg | "
                    f"speed={initial_speed:.3f}deg/s | "
                    f"FK dz={dz:+.3f}mm | "
                    f"xy={xy_mm:.3f}mm | "
                    f"max_tau={np.max(np.abs(raw)):.2f}Nm"
                )

            elif self.phase == "TRACKING":

                tracking_error = float(
                    np.max(np.abs(
                        np.rad2deg(qr - q)
                    ))
                )

                self.get_logger().info(
                    f"phase=TRACKING | "
                    f"t={t:.3f}s | "
                    f"max_tracking_error="
                    f"{tracking_error:.4f}deg | "
                    f"max_tau="
                    f"{np.max(np.abs(raw)):.2f}Nm | "
                    f"samples={len(self.rows)}"
                )

            elif self.phase == "FINAL_HOLD":

                final_error = float(
                    np.max(np.abs(
                        np.rad2deg(
                            self.qfinal - q
                        )
                    ))
                )

                self.get_logger().info(
                    f"phase=FINAL_HOLD | "
                    f"final_err={final_error:.4f}deg | "
                    f"max_tau="
                    f"{np.max(np.abs(raw)):.2f}Nm | "
                    f"CSV_saved={self.saved} | "
                    f"samples={len(self.rows)}"
                )


def main(args=None):

    rclpy.init(args=args)
    node = None

    try:
        node = TrajectoryPD()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:
            node.save()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
