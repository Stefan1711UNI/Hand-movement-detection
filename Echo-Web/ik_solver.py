import math

L1 = 12.0  # Length of Link 1 (Servo 2: shoulder-to-elbow) in cm
L2 = 13.0  # Length of Link 2 (Servo 3: elbow-to-gripper) in cm

def calculate_ik(target_dist, target_height):
    # Calculate the direct-line distance (D) from the shoulder to the target
    D = math.sqrt(target_dist**2 + target_height**2)

    # --- Check if the target is reachable ---
    if D > (L1 + L2):
        print("IK_SOLVER: Target unreachable (too far)")
        return None
    if D < abs(L1 - L2):
        print("IK_SOLVER: Target unreachable (too close)")
        return None
    
    # --- Use the Law of Cosines to solve the arm triangle ---
    # 1. Find the internal angle of the elbow (B)
    # D^2 = L1^2 + L2^2 - 2*L1*L2*cos(B)
    B_rad = math.acos((L1**2 + L2**2 - D**2) / (2 * L1 * L2))
    
    # The elbow servo angle is 180 degrees (pi) minus the internal angle
    elbow_angle_rad = math.pi - B_rad
    
    # 2. Find the shoulder angle
    # Find angle A1 (angle L1-D)
    A1_rad = math.acos((L1**2 + D**2 - L2**2) / (2 * L1 * D))
    # Find angle A2 (angle of the target relative to ground)
    A2_rad = math.atan2(target_height, target_dist)
    
    # The final shoulder angle is the sum of these two
    shoulder_angle_rad = A1_rad + A2_rad

    # --- Convert to degrees and return ---
    angles = {
        'shoulder': math.degrees(shoulder_angle_rad),
        'elbow': math.degrees(elbow_angle_rad)
    }
    
    return angles

if __name__ == "__main__":
    # Test a point 12cm forward and 10cm up 
    d = 12.0
    h = 10.0
    print(f"Calculating for target: {d}:{h}")
    angles = calculate_ik(d, h)
    
    if angles:
        print(f"  Shoulder (Servo 2): {angles['shoulder']:.2f}°")
        print(f"  Elbow (Servo 3): {angles['elbow']:.2f}°")