"""
kalman3d.py

Minimal, streaming-friendly 3D Kalman filter for hand-tracking world coordinates.

Contents:
- Kalman3D: Kalman filter class with state [x,y,z,vx,vy,vz]
- helper functions: estimate_R_from_samples, load_csv_sample

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

