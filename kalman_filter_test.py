"""
kalman3d.py

Minimal, streaming-friendly 3D Kalman filter for hand-tracking world coordinates.

Contents:
- Kalman3D: Kalman filter class with state [x,y,z,vx,vy,vz]
- helper functions: estimate_R_from_samples, load_csv_sample
- small demo in __main__ that shows how to run the filter on a CSV and plot results

Requirements: numpy, optionally matplotlib for the demo.

Designed so you can drop this into your hand-tracking pipeline and call step(timestamp_s, (x,y,z)).
"""

import numpy as np
import time

try:
    import matplotlib.pyplot as _plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class Kalman3D:
    """
    3D Kalman filter with constant-velocity state.

    State vector (6x1): [x, y, z, vx, vy, vz]^T
    Measurement: position [x, y, z]

    Usage (streaming):
        kf = Kalman3D(initial_time=time.time(), q=0.02)
        ok, mahal = kf.step(timestamp_s, (x,y,z))   # predict + update
        pos, vel = kf.get_state()

    Methods:
      - step(timestamp_s, z, gating_threshold=16.0): perform predict+update
      - predict_only(timestamp_s): advance state without measurement
      - get_state(): returns (pos (3,), vel (3,))
      - set_R_from_samples(samples): estimate measurement covariance from Nx3 samples
    """

    def __init__(self, initial_time=None, q=0.02, R_diag=None):
        # state vector and covariance
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

        # last timestamp (seconds). Use the same clock for timestamps (e.g., time.time()).
        self.last_time = initial_time

        # measurement matrix (3x6): measures position only
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # measurement noise R
        if R_diag is None:
            # reasonable default; replace with set_R_from_samples for best results
            self.R = np.diag([0.01**2, 0.01**2, 0.02**2])
        else:
            self.R = np.diag(R_diag)

        # process noise scale (scalar). Increase to react faster, decrease to smooth more.
        self.q = float(q)

        # cached matrices for current dt
        self.F = np.eye(6)
        self.Q = np.zeros((6, 6))

    def _set_dt(self, dt):
        """Build F and Q matrices for a given dt."""
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.F = F

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = self.q

        Q = np.zeros((6, 6), dtype=float)
        Q_pos = (dt4 / 4.0) * q
        Q_cross = (dt3 / 2.0) * q
        Q_vel = (dt2) * q
        for i in range(3):
            Q[i, i] = Q_pos
            Q[i, i + 3] = Q_cross
            Q[i + 3, i] = Q_cross
            Q[i + 3, i + 3] = Q_vel
        self.Q = Q

    def predict(self, dt):
        """Predict step: advance state by dt seconds."""
        if dt <= 0:
            dt = 1e-6
        self._set_dt(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, gating_threshold=None):
        """
        Update step with measurement z (3-vector).
        Returns (accepted_bool, mahalanobis_distance).
        If gating_threshold provided and mahalanobis > threshold, the measurement is ignored.
        """
        z = np.reshape(z, (3, 1))
        y = z - (self.H @ self.x)  # residual
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        mahal = float((y.T @ S_inv @ y)[0, 0])
        if gating_threshold is not None and mahal > gating_threshold:
            return False, mahal
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        return True, mahal

    def step(self, timestamp_s, z, gating_threshold=16.0):
        """
        One-step convenience wrapper: compute dt from last_time, run predict+update, update last_time.
        If this is the first measurement the filter initializes position to the measurement and leaves velocity zero.
        """
        if self.last_time is None:
            self.last_time = timestamp_s
            self.x[:3, 0] = np.reshape(z, (3,))
            return True, 0.0
        dt = timestamp_s - self.last_time
        # guard against bad timestamps
        if dt <= 0:
            dt = 1e-3
        self.predict(dt)
        ok, mahal = self.update(z, gating_threshold=gating_threshold)
        self.last_time = timestamp_s
        return ok, mahal

    def predict_only(self, timestamp_s):
        """Advance state by dt without an update (use when no detection available)."""
        if self.last_time is None:
            self.last_time = timestamp_s
            return
        dt = timestamp_s - self.last_time
        if dt <= 0:
            dt = 1e-3
        self.predict(dt)
        self.last_time = timestamp_s

    def get_state(self):
        """Return (pos (3,), vel (3,))."""
        return self.x[:3, 0].copy(), self.x[3:, 0].copy()

    def set_R_from_samples(self, samples):
        """
        Estimate measurement covariance R from Nx3 array of stationary samples.
        Returns the estimated variances.
        """
        samples = np.asarray(samples)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError("samples must be Nx3 array")
        var = np.var(samples, axis=0, ddof=1)
        self.R = np.diag(var)
        return var


# ---------- helpers ----------

def estimate_R_from_samples(samples):
    """Convenience wrapper to compute diagonal R from Nx3 stationary samples."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("samples must be Nx3 array")
    var = np.var(samples, axis=0, ddof=1)
    return np.diag(var)


def load_csv_sample_string(csv_text):
    """Very small CSV loader that expects the same columns you provided.
    Returns timestamps (ms) and positions (Nx3 in meters).
    """
    import io
    rows = []
    f = io.StringIO(csv_text)
    hdr = f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        ts = int(parts[0])
        x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
        rows.append((ts, x, y, z))
    arr = np.array(rows)
    if arr.size == 0:
        return np.array([]), np.array([])
    return arr[:, 0].astype(np.int64), arr[:, 1:4]


# ---------- small demo when run directly ----------
if __name__ == "__main__":
    # Demo: run on a small embedded CSV (replace with your path for real use)
    sample_csv = """timestamp_ms,timestamp_iso_utc,landmark,x_m,y_m,z_m
1758791969174,2025-09-25T09:19:29.174000+00:00,wrist_world,0.002727,0.076788,0.013080
1758791969210,2025-09-25T09:19:29.210000+00:00,wrist_world,0.016238,0.087679,0.036159
1758791969297,2025-09-25T09:19:29.297000+00:00,wrist_world,0.015423,0.088681,0.018416
1758791969317,2025-09-25T09:19:29.317000+00:00,wrist_world,0.013722,0.087132,0.014477
1758791969345,2025-09-25T09:19:29.345000+00:00,wrist_world,0.012377,0.086462,0.017050
1758791969366,2025-09-25T09:19:29.366000+00:00,wrist_world,0.010017,0.087466,0.021035
1758791969379,2025-09-25T09:19:29.379000+00:00,wrist_world,0.009183,0.089577,0.015203
1758791969394,2025-09-25T09:19:29.394000+00:00,wrist_world,0.010357,0.089717,0.010542
1758791969418,2025-09-25T09:19:29.418000+00:00,wrist_world,0.008430,0.090869,0.007208
1758791969471,2025-09-25T09:19:29.471000+00:00,wrist_world,0.006360,0.091413,0.002838
1758791969517,2025-09-25T09:19:29.517000+00:00,wrist_world,0.005392,0.092001,0.007705
1758791969548,2025-09-25T09:19:29.548000+00:00,wrist_world,0.004681,0.092028,0.002201
1758791969595,2025-09-25T09:19:29.595000+00:00,wrist_world,0.004965,0.092786,-0.004934
1758791969624,2025-09-25T09:19:29.624000+00:00,wrist_world,0.007212,0.093359,-0.001330
1758791969650,2025-09-25T09:19:29.650000+00:00,wrist_world,0.010361,0.091868,-0.007242
1758791969683,2025-09-25T09:19:29.683000+00:00,wrist_world,0.011424,0.089781,-0.000359
1758791969694,2025-09-25T09:19:29.694000+00:00,wrist_world,0.010802,0.090888,-0.000239
1758791969728,2025-09-25T09:19:29.728000+00:00,wrist_world,0.014024,0.087965,0.012518
1758791969767,2025-09-25T09:19:29.767000+00:00,wrist_world,0.014295,0.086176,0.016979
1758791969794,2025-09-25T09:19:29.794000+00:00,wrist_world,0.013679,0.088058,0.016298
1758791969841,2025-09-25T09:19:29.841000+00:00,wrist_world,0.012590,0.087954,0.013348
1758791969878,2025-09-25T09:19:29.878000+00:00,wrist_world,0.011889,0.088278,0.013394
1758791969924,2025-09-25T09:19:29.924000+00:00,wrist_world,0.012356,0.089017,0.013623
1758791969935,2025-09-25T09:19:29.935000+00:00,wrist_world,0.011337,0.088375,0.014311
1758791969993,2025-09-25T09:19:29.993000+00:00,wrist_world,0.013665,0.090441,0.009812
1758791970010,2025-09-25T09:19:30.010000+00:00,wrist_world,0.011038,0.091142,0.012285
1758791970026,2025-09-25T09:19:30.026000+00:00,wrist_world,0.011279,0.090778,0.015026
1758791970072,2025-09-25T09:19:30.072000+00:00,wrist_world,0.011773,0.090919,0.012369
1758791970093,2025-09-25T09:19:30.093000+00:00,wrist_world,0.011398,0.090815,0.010404
1758791970139,2025-09-25T09:19:30.139000+00:00,wrist_world,0.010890,0.090077,0.011090
1758791970173,2025-09-25T09:19:30.173000+00:00,wrist_world,0.012185,0.089742,0.012411
1758791970209,2025-09-25T09:19:30.209000+00:00,wrist_world,0.012574,0.090748,0.012495
"""

    ts_ms, pos = load_csv_sample_string(sample_csv)
    if pos.size == 0:
        print("No data in sample CSV")
        raise SystemExit(1)

    timestamps_s = ts_ms.astype(np.float64) / 1000.0
    kf = Kalman3D(initial_time=timestamps_s[0], q=0.02)

    # estimate R from a short stationary window if you have one; here we just use empirical var
    var = np.var(pos, axis=0, ddof=1)
    kf.R = np.diag(var)
    print("Estimated measurement std (m):", np.sqrt(var))

    est_pos = np.zeros_like(pos)
    est_vel = np.zeros_like(pos)
    t0 = time.perf_counter()
    for i in range(pos.shape[0]):
        ok, mahal = kf.step(timestamps_s[i], pos[i], gating_threshold=16.0)
        p, v = kf.get_state()
        est_pos[i] = p
        est_vel[i] = v
    dt = time.perf_counter() - t0
    print(f"Processed {pos.shape[0]} samples in {dt:.4f}s ({dt/pos.shape[0]*1000.0:.3f} ms/frame)")

    if _HAS_MPL:
        t_axis = timestamps_s - timestamps_s[0]
        _plt.figure(figsize=(8,3)); _plt.plot(t_axis, pos[:,0], '.', markersize=4, label='measured x'); _plt.plot(t_axis, est_pos[:,0], label='filtered x'); _plt.legend(); _plt.title('X')
        _plt.figure(figsize=(8,3)); _plt.plot(t_axis, pos[:,1], '.', markersize=4, label='measured y'); _plt.plot(t_axis, est_pos[:,1], label='filtered y'); _plt.legend(); _plt.title('Y')
        _plt.figure(figsize=(8,3)); _plt.plot(t_axis, pos[:,2], '.', markersize=4, label='measured z'); _plt.plot(t_axis, est_pos[:,2], label='filtered z'); _plt.legend(); _plt.title('Z')
        _plt.show()

