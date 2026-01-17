import numpy as np

class Kalman_fil:
    """
    The 'New' Simple Static Filter (Understandable Version).
    Wrapped to look like the old class so video_feed_handler works.
    """
    def __init__(self, initial_time=None, q=None): 

        self.estimate = np.zeros(3)
        self.p_uncertainty = 1.0
        
        # 2. Tuning
        self.q_process_noise = 0.001  # How fast you move was 0.02
        self.r_measure_noise = 0.05      # Camera noise was 0.01

    def step(self, timestamp_s, z, gating_threshold=16.0):
        # Input 'z' is (x, y, z) from camera
        measurement = np.array([z[0], z[1], z[2]])
        
        # --- YOUR NEW ALGORITHM ---
        
        # 1. Prediction (Add doubt)
        self.p_uncertainty = self.p_uncertainty + self.q_process_noise
        
        # 2. Gain
        # Avoid division by zero
        denom = self.p_uncertainty + self.r_measure_noise
        if denom == 0: denom = 1e-6
        k_gain = self.p_uncertainty / denom
        
        # 3. Update Position
        self.estimate = self.estimate + k_gain * (measurement - self.estimate)
        
        # 4. Update Uncertainty
        self.p_uncertainty = (1 - k_gain) * self.p_uncertainty
        
        # Return True (accepted) and 0.0 (dummy mahal distance)
        return True

    def get_state(self):
        # Returns: (position, velocity)
        # We return [0,0,0] for velocity because we removed it!
        return self.estimate