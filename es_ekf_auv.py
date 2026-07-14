"""
Error-State Extended Kalman Filter (ES-EKF) for Autonomous Underwater
Vehicle (AUV) Navigation.

Fuses IMU (accel + gyro), depth (pressure) sensor, Doppler Velocity Log
(DVL), GPS (surface-only), and magnetometer measurements into a single
consistent estimate of position, velocity, and orientation.

Frame convention: NED (North-East-Down), body frame FRD (Forward-Right-Down).
Depth is therefore +z (positive down). Quaternion is world-to-body? -- We
define q as the BODY-TO-WORLD rotation, i.e. v_world = R(q) @ v_body.

Author: Autonomous Navigation & Robotics Engineering submission
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

RNG = np.random.default_rng(42)

# --------------------------------------------------------------------------
# 1. QUATERNION UTILITIES  (Hamilton convention, q = [w, x, y, z])
# --------------------------------------------------------------------------

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_from_euler(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) Euler angles -> quaternion."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])

def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

def quat_from_small_angle(dtheta):
    """Small-angle approximation: delta rotation vector -> quaternion."""
    angle = np.linalg.norm(dtheta)
    if angle < 1e-8:
        return np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
    axis = dtheta / angle
    return np.concatenate(([np.cos(angle/2)], axis*np.sin(angle/2)))


# --------------------------------------------------------------------------
# 2. SENSOR / NOISE CONFIGURATION
#    (Representative of a small survey-class AUV, e.g. Bluefin/HUGIN class)
# --------------------------------------------------------------------------

@dataclass
class SensorConfig:
    # IMU (tactical-grade MEMS), typically 100-200 Hz
    imu_rate_hz: float = 100.0
    accel_noise_density: float = 0.02          # m/s^2 / sqrt(Hz)  (white noise)
    gyro_noise_density: float = 0.001          # rad/s / sqrt(Hz)
    accel_bias_stability: float = 0.0005       # m/s^3 / sqrt(Hz) (random walk)
    gyro_bias_stability: float = 1e-6          # rad/s^2 / sqrt(Hz)
    accel_bias_init: np.ndarray = field(default_factory=lambda: np.array([0.08, -0.05, 0.10]))
    gyro_bias_init: np.ndarray = field(default_factory=lambda: np.array([0.004, -0.003, 0.002]))

    # Depth / pressure sensor, ~10-20 Hz, very accurate but has zero-offset drift
    depth_rate_hz: float = 10.0
    depth_noise_std: float = 0.03              # m (very precise: high-res pressure transducer)

    # DVL (Doppler Velocity Log), ~5-10 Hz, requires bottom-lock
    dvl_rate_hz: float = 7.0
    dvl_noise_std: float = 0.02                # m/s per axis
    dvl_max_altitude_m: float = 150.0          # bottom-lock lost above this altitude
    dvl_dropout_prob: float = 0.01             # random bottom-lock loss per epoch

    # GPS, only valid at/near the surface, ~1 Hz
    gps_rate_hz: float = 1.0
    gps_noise_std: float = 2.5                 # m (consumer-grade surface fix)
    gps_surface_depth_threshold_m: float = 0.5

    # Magnetometer, ~20 Hz, heading aiding (subject to hard/soft iron distortion)
    mag_rate_hz: float = 20.0
    mag_noise_std: float = 0.03                # unit-vector noise

    # Visual Odometry (camera), ~10 Hz, body-frame velocity increments,
    # degrades/dropouts with turbidity or low light
    vo_rate_hz: float = 10.0
    vo_noise_std: float = 0.06                 # m/s, noisier & less reliable than DVL
    vo_dropout_prob: float = 0.15              # frequent dropouts (turbidity, feature-poor seafloor)


G_WORLD = np.array([0.0, 0.0, 9.81])          # gravity, NED (+z down)
MAG_WORLD = np.array([0.62, 0.02, 0.78])      # representative local magnetic field, normalized-ish


# --------------------------------------------------------------------------
# 3. GROUND-TRUTH MISSION GENERATION
#    A realistic lawnmower survey: dive -> survey legs with turns -> periodic
#    surfacing for GPS fixes -> final ascent.
# --------------------------------------------------------------------------

def generate_ground_truth(duration_s=600.0, dt=0.01):
    n = int(duration_s / dt)
    t = np.arange(n) * dt

    # Commanded body-frame forward speed and yaw-rate program (lawnmower survey)
    u_cmd = np.full(n, 1.4)         # forward speed, m/s
    r_cmd = np.zeros(n)             # yaw rate, rad/s
    w_cmd = np.zeros(n)             # dive rate (world down), m/s

    leg_time = 70.0
    turn_time = 8.0
    depth_target = 30.0

    # Dive phase (first 40s)
    dive_end = 40.0
    w_cmd[t < dive_end] = depth_target / dive_end

    # Lawnmower legs with 180-degree turns, alternating; brief surfacing every ~3 legs
    cycle = leg_time + turn_time
    leg_idx = 0
    tt = dive_end
    turning = False
    surface_windows = []
    while tt < duration_s - 30:
        leg_start = tt
        leg_stop = min(tt + leg_time, duration_s)
        mask = (t >= leg_start) & (t < leg_stop)
        r_cmd[mask] = 0.0
        tt = leg_stop
        if tt >= duration_s - 30:
            break
        turn_start = tt
        turn_stop = min(tt + turn_time, duration_s)
        mask_t = (t >= turn_start) & (t < turn_stop)
        r_cmd[mask_t] = np.pi / turn_time * (1 if leg_idx % 2 == 0 else -1)
        tt = turn_stop
        leg_idx += 1
        # Every 3rd leg, surface briefly for a GPS fix
        if leg_idx % 3 == 0 and tt < duration_s - 60:
            surf_start = tt
            surf_end = tt + 25.0
            surface_windows.append((surf_start, surf_end))
            mask_s = (t >= surf_start) & (t < surf_end)
            # ascent then descent bracketing a hold-at-surface period
            asc = mask_s & (t < surf_start + 8)
            hold = mask_s & (t >= surf_start + 8) & (t < surf_end - 8)
            desc = mask_s & (t >= surf_end - 8)
            w_cmd[asc] = -depth_target / 8.0
            w_cmd[hold] = 0.0
            w_cmd[desc] = depth_target / 8.0
            tt = surf_end

    # Final ascent to surface
    final_ascent_start = duration_s - 30
    mask_final = t >= final_ascent_start
    w_cmd[mask_final] = -depth_target / 30.0
    surface_windows.append((duration_s - 15, duration_s))

    # Integrate kinematics to build ground-truth pose/velocity/orientation
    pos = np.zeros((n, 3))
    vel_world = np.zeros((n, 3))
    yaw = np.zeros(n)
    roll = np.zeros(n)
    pitch = np.zeros(n)
    depth = np.zeros(n)

    for k in range(1, n):
        yaw[k] = yaw[k-1] + r_cmd[k-1]*dt
        # gentle roll/pitch coupling with turning + speed for realism
        roll[k] = 0.15*np.sin(0.3*t[k]) + 0.4*(r_cmd[k-1])
        pitch[k] = 0.05*np.sin(0.2*t[k]) + 0.02*w_cmd[k-1]
        depth[k] = depth[k-1] + w_cmd[k-1]*dt
        vx_world = u_cmd[k-1]*np.cos(yaw[k-1])
        vy_world = u_cmd[k-1]*np.sin(yaw[k-1])
        vz_world = w_cmd[k-1]
        vel_world[k] = [vx_world, vy_world, vz_world]
        pos[k] = pos[k-1] + vel_world[k-1]*dt
        pos[k, 2] = depth[k]

    quats = np.array([quat_from_euler(roll[k], pitch[k], yaw[k]) for k in range(n)])

    # True world-frame acceleration via finite differencing of velocity
    acc_world = np.zeros((n, 3))
    acc_world[1:] = np.diff(vel_world, axis=0) / dt
    acc_world[0] = acc_world[1]

    # True body-frame angular rate via finite differencing of Euler angles (approx, small dt)
    gyro_true = np.zeros((n, 3))
    gyro_true[:, 2] = r_cmd  # yaw rate ~ body z-rate (small roll/pitch approx)
    gyro_true[1:, 0] = np.diff(roll) / dt
    gyro_true[1:, 1] = np.diff(pitch) / dt

    return {
        "t": t, "dt": dt, "pos": pos, "vel_world": vel_world, "quat": quats,
        "acc_world": acc_world, "gyro_body": gyro_true, "depth": depth,
        "surface_windows": surface_windows,
    }


def is_surfaced(t_query, surface_windows):
    for (a, b) in surface_windows:
        if a <= t_query <= b:
            return True
    return False


# --------------------------------------------------------------------------
# 4. SENSOR SIMULATION  (adds bias + noise + dropouts to ground truth)
# --------------------------------------------------------------------------

def simulate_sensors(truth, cfg: SensorConfig):
    t, dt = truth["t"], truth["dt"]
    n = len(t)
    fs_imu = 1.0 / dt

    # --- IMU: accelerometer (specific force) & gyroscope, at native truth rate ---
    accel_bias = np.zeros((n, 3))
    gyro_bias = np.zeros((n, 3))
    accel_bias[0] = cfg.accel_bias_init
    gyro_bias[0] = cfg.gyro_bias_init
    ab_rw_std = cfg.accel_bias_stability * np.sqrt(dt)
    gb_rw_std = cfg.gyro_bias_stability * np.sqrt(dt)
    for k in range(1, n):
        accel_bias[k] = accel_bias[k-1] + RNG.normal(0, ab_rw_std, 3)
        gyro_bias[k] = gyro_bias[k-1] + RNG.normal(0, gb_rw_std, 3)

    accel_meas = np.zeros((n, 3))
    gyro_meas = np.zeros((n, 3))
    an_std = cfg.accel_noise_density * np.sqrt(fs_imu)
    gn_std = cfg.gyro_noise_density * np.sqrt(fs_imu)
    for k in range(n):
        R_bw = quat_to_rotmat(truth["quat"][k]).T  # world->body
        f_true = R_bw @ (truth["acc_world"][k] - G_WORLD)  # specific force in body frame
        accel_meas[k] = f_true + accel_bias[k] + RNG.normal(0, an_std, 3)
        gyro_meas[k] = truth["gyro_body"][k] + gyro_bias[k] + RNG.normal(0, gn_std, 3)

    def decimate_indices(rate_hz):
        step = max(1, int(round(fs_imu / rate_hz)))
        return np.arange(0, n, step)

    # --- Depth sensor ---
    depth_idx = decimate_indices(cfg.depth_rate_hz)
    depth_meas = truth["depth"][depth_idx] + RNG.normal(0, cfg.depth_noise_std, len(depth_idx))

    # --- DVL (body-frame velocity), with bottom-lock dropouts ---
    dvl_idx = decimate_indices(cfg.dvl_rate_hz)
    dvl_meas, dvl_valid_idx = [], []
    for k in dvl_idx:
        if RNG.uniform() < cfg.dvl_dropout_prob:
            continue
        R_bw = quat_to_rotmat(truth["quat"][k]).T
        v_body = R_bw @ truth["vel_world"][k]
        dvl_meas.append(v_body + RNG.normal(0, cfg.dvl_noise_std, 3))
        dvl_valid_idx.append(k)
    dvl_meas = np.array(dvl_meas)
    dvl_valid_idx = np.array(dvl_valid_idx)

    # --- GPS (surface only) ---
    gps_idx = decimate_indices(cfg.gps_rate_hz)
    gps_meas, gps_valid_idx = [], []
    for k in gps_idx:
        if is_surfaced(t[k], truth["surface_windows"]):
            gps_meas.append(truth["pos"][k, :2] + RNG.normal(0, cfg.gps_noise_std, 2))
            gps_valid_idx.append(k)
    gps_meas = np.array(gps_meas)
    gps_valid_idx = np.array(gps_valid_idx)

    # --- Magnetometer (body-frame field vector) ---
    mag_idx = decimate_indices(cfg.mag_rate_hz)
    mag_meas = []
    for k in mag_idx:
        R_bw = quat_to_rotmat(truth["quat"][k]).T
        m_body = R_bw @ MAG_WORLD
        mag_meas.append(m_body + RNG.normal(0, cfg.mag_noise_std, 3))
    mag_meas = np.array(mag_meas)

    # --- Visual Odometry (body-frame velocity), frequent dropouts ---
    vo_idx = decimate_indices(cfg.vo_rate_hz)
    vo_meas, vo_valid_idx = [], []
    for k in vo_idx:
        if RNG.uniform() < cfg.vo_dropout_prob:
            continue
        R_bw = quat_to_rotmat(truth["quat"][k]).T
        v_body = R_bw @ truth["vel_world"][k]
        vo_meas.append(v_body + RNG.normal(0, cfg.vo_noise_std, 3))
        vo_valid_idx.append(k)
    vo_meas = np.array(vo_meas)
    vo_valid_idx = np.array(vo_valid_idx)

    return {
        "accel_meas": accel_meas, "gyro_meas": gyro_meas,
        "depth_idx": depth_idx, "depth_meas": depth_meas,
        "dvl_idx": dvl_valid_idx, "dvl_meas": dvl_meas,
        "gps_idx": gps_valid_idx, "gps_meas": gps_meas,
        "mag_idx": mag_idx, "mag_meas": mag_meas,
        "vo_idx": vo_valid_idx, "vo_meas": vo_meas,
    }


# --------------------------------------------------------------------------
# 5. ERROR-STATE EXTENDED KALMAN FILTER
#    Nominal state x = [p(3), v(3), q(4), ba(3), bg(3)]   (16,)
#    Error state  dx = [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]  (15,)
# --------------------------------------------------------------------------

class ESEKF:
    IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG = 0, 3, 6, 9, 12  # error-state block offsets

    def __init__(self, p0, v0, q0, ba0, bg0, P0, cfg: SensorConfig):
        self.p = p0.copy()
        self.v = v0.copy()
        self.q = q0.copy()
        self.ba = ba0.copy()
        self.bg = bg0.copy()
        self.P = P0.copy()
        self.cfg = cfg

    # ---- Prediction: propagate nominal state with IMU, propagate error covariance ----
    def predict(self, accel_meas, gyro_meas, dt):
        f = accel_meas - self.ba          # corrected specific force
        w = gyro_meas - self.bg           # corrected angular rate
        R = quat_to_rotmat(self.q)

        # Nominal state propagation (Euler integration; dt is small ~0.01s)
        a_world = R @ f + G_WORLD
        self.p = self.p + self.v*dt + 0.5*a_world*dt**2
        self.v = self.v + a_world*dt
        dq = quat_from_small_angle(w*dt)
        self.q = quat_normalize(quat_mult(self.q, dq))

        # Error-state transition matrix F (15x15), continuous model linearized
        # about the current nominal state, discretized to first order.
        F = np.eye(15)
        F[self.IDX_P:self.IDX_P+3, self.IDX_V:self.IDX_V+3] = np.eye(3)*dt
        F[self.IDX_V:self.IDX_V+3, self.IDX_TH:self.IDX_TH+3] = -R @ skew(f) * dt
        F[self.IDX_V:self.IDX_V+3, self.IDX_BA:self.IDX_BA+3] = -R * dt
        F[self.IDX_TH:self.IDX_TH+3, self.IDX_TH:self.IDX_TH+3] = np.eye(3) - skew(w)*dt
        F[self.IDX_TH:self.IDX_TH+3, self.IDX_BG:self.IDX_BG+3] = -np.eye(3)*dt

        # Process noise covariance Qd (15x15), built from IMU noise PSDs
        c = self.cfg
        fs = 1.0/dt
        var_a = (c.accel_noise_density**2) * fs
        var_g = (c.gyro_noise_density**2) * fs
        var_ba = (c.accel_bias_stability**2) * dt
        var_bg = (c.gyro_bias_stability**2) * dt

        Qc = np.zeros((15, 15))
        Qc[self.IDX_V:self.IDX_V+3, self.IDX_V:self.IDX_V+3] = np.eye(3)*var_a
        Qc[self.IDX_TH:self.IDX_TH+3, self.IDX_TH:self.IDX_TH+3] = np.eye(3)*var_g
        Qc[self.IDX_BA:self.IDX_BA+3, self.IDX_BA:self.IDX_BA+3] = np.eye(3)*var_ba
        Qc[self.IDX_BG:self.IDX_BG+3, self.IDX_BG:self.IDX_BG+3] = np.eye(3)*var_bg
        Qd = Qc * dt

        self.P = F @ self.P @ F.T + Qd

    # ---- Generic measurement update given residual y, Jacobian H, noise R ----
    def _apply_update(self, y, H, R_mat):
        S = H @ self.P @ H.T + R_mat
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        dx = K @ y

        self.p = self.p + dx[self.IDX_P:self.IDX_P+3]
        self.v = self.v + dx[self.IDX_V:self.IDX_V+3]
        dtheta = dx[self.IDX_TH:self.IDX_TH+3]
        dq = quat_from_small_angle(dtheta)
        self.q = quat_normalize(quat_mult(self.q, dq))
        self.ba = self.ba + dx[self.IDX_BA:self.IDX_BA+3]
        self.bg = self.bg + dx[self.IDX_BG:self.IDX_BG+3]

        I15 = np.eye(15)
        IKH = I15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_mat @ K.T   # Joseph form (numerically stable)

        return np.linalg.norm(y)  # for innovation logging

    def update_depth(self, z_depth, noise_std):
        H = np.zeros((1, 15))
        H[0, self.IDX_P+2] = 1.0
        y = np.array([z_depth - self.p[2]])
        return self._apply_update(y, H, np.array([[noise_std**2]]))

    def update_gps(self, z_xy, noise_std):
        H = np.zeros((2, 15))
        H[0, self.IDX_P+0] = 1.0
        H[1, self.IDX_P+1] = 1.0
        y = z_xy - self.p[:2]
        return self._apply_update(y, H, np.eye(2)*noise_std**2)

    def _body_vector_update(self, z_body, world_vec_source, noise_std):
        """Shared form for DVL / VO (velocity) and magnetometer (field vector):
        measurement model h(x) = R(q)^T @ w   where w is either world velocity
        or a known world-frame vector (e.g. magnetic field)."""
        R = quat_to_rotmat(self.q)
        w_vec = world_vec_source
        z_pred = R.T @ w_vec
        y = z_body - z_pred
        H = np.zeros((3, 15))
        if world_vec_source is self.v:
            H[:, self.IDX_V:self.IDX_V+3] = R.T
        H[:, self.IDX_TH:self.IDX_TH+3] = skew(z_pred)  # d(R^T w)/d(theta) = [R^T w]_x
        return y, H

    def update_dvl(self, z_body_vel, noise_std):
        y, H = self._body_vector_update(z_body_vel, self.v, noise_std)
        return self._apply_update(y, H, np.eye(3)*noise_std**2)

    def update_vo(self, z_body_vel, noise_std):
        y, H = self._body_vector_update(z_body_vel, self.v, noise_std)
        return self._apply_update(y, H, np.eye(3)*noise_std**2)

    def update_mag(self, z_body_field, noise_std):
        y, H = self._body_vector_update(z_body_field, MAG_WORLD, noise_std)
        return self._apply_update(y, H, np.eye(3)*noise_std**2)


# --------------------------------------------------------------------------
# 6. MAIN SIMULATION LOOP
# --------------------------------------------------------------------------

def run_filter(truth, sensors, cfg: SensorConfig):
    n = len(truth["t"])
    dt = truth["dt"]

    # Initialize filter with a perturbed guess (imperfect initial alignment,
    # as would be realistic before a diagnostic surface GPS fix)
    p0 = truth["pos"][0] + RNG.normal(0, 1.0, 3)
    v0 = np.zeros(3)
    q0 = quat_from_euler(0.0, 0.0, quat_to_euler(truth["quat"][0])[2] + np.deg2rad(5))
    ba0 = np.zeros(3)
    bg0 = np.zeros(3)

    P0 = np.diag([1.0]*3 + [0.25]*3 + [np.deg2rad(10)**2]*3 + [0.05**2]*3 + [0.002**2]*3)

    ekf = ESEKF(p0, v0, q0, ba0, bg0, P0, cfg)

    est_pos = np.zeros((n, 3))
    est_vel = np.zeros((n, 3))
    est_eul = np.zeros((n, 3))
    est_ba = np.zeros((n, 3))
    est_bg = np.zeros((n, 3))
    P_diag = np.zeros((n, 15))

    depth_ptr = dvl_ptr = gps_ptr = mag_ptr = vo_ptr = 0
    innovations = {"depth": [], "dvl": [], "gps": [], "mag": [], "vo": []}
    innovation_t = {"depth": [], "dvl": [], "gps": [], "mag": [], "vo": []}

    for k in range(n):
        if k > 0:
            ekf.predict(sensors["accel_meas"][k-1], sensors["gyro_meas"][k-1], dt)

        if depth_ptr < len(sensors["depth_idx"]) and sensors["depth_idx"][depth_ptr] == k:
            nrm = ekf.update_depth(sensors["depth_meas"][depth_ptr], cfg.depth_noise_std)
            innovations["depth"].append(nrm); innovation_t["depth"].append(truth["t"][k])
            depth_ptr += 1

        if dvl_ptr < len(sensors["dvl_idx"]) and sensors["dvl_idx"][dvl_ptr] == k:
            nrm = ekf.update_dvl(sensors["dvl_meas"][dvl_ptr], cfg.dvl_noise_std)
            innovations["dvl"].append(nrm); innovation_t["dvl"].append(truth["t"][k])
            dvl_ptr += 1

        if gps_ptr < len(sensors["gps_idx"]) and sensors["gps_idx"][gps_ptr] == k:
            nrm = ekf.update_gps(sensors["gps_meas"][gps_ptr], cfg.gps_noise_std)
            innovations["gps"].append(nrm); innovation_t["gps"].append(truth["t"][k])
            gps_ptr += 1

        if mag_ptr < len(sensors["mag_idx"]) and sensors["mag_idx"][mag_ptr] == k:
            nrm = ekf.update_mag(sensors["mag_meas"][mag_ptr], cfg.mag_noise_std)
            innovations["mag"].append(nrm); innovation_t["mag"].append(truth["t"][k])
            mag_ptr += 1

        if vo_ptr < len(sensors["vo_idx"]) and sensors["vo_idx"][vo_ptr] == k:
            nrm = ekf.update_vo(sensors["vo_meas"][vo_ptr], cfg.vo_noise_std)
            innovations["vo"].append(nrm); innovation_t["vo"].append(truth["t"][k])
            vo_ptr += 1

        est_pos[k] = ekf.p
        est_vel[k] = ekf.v
        est_eul[k] = quat_to_euler(ekf.q)
        est_ba[k] = ekf.ba
        est_bg[k] = ekf.bg
        P_diag[k] = np.diag(ekf.P)

    return {
        "est_pos": est_pos, "est_vel": est_vel, "est_eul": est_eul,
        "est_ba": est_ba, "est_bg": est_bg, "P_diag": P_diag,
        "innovations": innovations, "innovation_t": innovation_t,
    }


if __name__ == "__main__":
    import pickle
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    cfg = SensorConfig()
    truth = generate_ground_truth(duration_s=600.0, dt=0.01)
    sensors = simulate_sensors(truth, cfg)
    result = run_filter(truth, sensors, cfg)
    np.savez(os.path.join(HERE, "sim_output.npz"),
             t=truth["t"], pos=truth["pos"], vel=truth["vel_world"],
             quat=truth["quat"], surface_windows=np.array(truth["surface_windows"]),
             est_pos=result["est_pos"], est_vel=result["est_vel"], est_eul=result["est_eul"],
             est_ba=result["est_ba"], est_bg=result["est_bg"], P_diag=result["P_diag"])
    with open(os.path.join(HERE, "sim_extra.pkl"), "wb") as fh:
        pickle.dump({
            "innovations": result["innovations"],
            "innovation_t": result["innovation_t"],
        }, fh)
    print("Simulation complete. Saved sim_output.npz")
    pos_err = result["est_pos"] - truth["pos"]
    vel_err = result["est_vel"] - truth["vel_world"]
    rmse_p = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
    rmse_v = np.sqrt(np.mean(np.sum(vel_err**2, axis=1)))
    print(f"Position RMSE: {rmse_p:.3f} m | Velocity RMSE: {rmse_v:.3f} m/s")
