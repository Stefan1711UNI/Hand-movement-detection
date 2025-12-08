import math

#Arm Link Lengths (cm)
L1 = 12.0  #Shoulder to Elbow
L2 = 13.0  #Elbow to Gripper

#Hardware Limits (Degrees)
SHOULDER_MIN = 120
SHOULDER_MAX = 170
ELBOW_MIN = 0
ELBOW_MAX = 160

def apply_limits(value, value_min, value_max):
    checked_value = max(value_min, min(value, value_max))
    return checked_value


def calculate_ik(target_dist, target_height):
    max_reach = L1 + L2 - 0.1 #0.1 is safety value
    min_reach = abs(L1 - L2) + 0.1

    #pythagorus theorum
    D = math.sqrt(target_dist**2 + target_height**2)

    #if past boundries we will not throw out the calculation, just bring it back withing safety margins
    if D > max_reach:
        scale = max_reach / D
        D = D * scale
        target_dist = target_dist * scale
        target_height = target_height * scale

    if D < min_reach:
        scale = min_reach / D
        D = D * scale
        target_dist = target_dist * scale
        target_height = target_height * scale


    #Calc geometry
    cos_angle_b = (L1**2 + L2**2 - D**2) / (2*L1*L2)
    b_angle_rad = math.acos(apply_limits(cos_angle_b, -1.0, 1.0))
    elbow_angle_rad = math.pi - b_angle_rad

    #Shoulder angle (A1 + A2)
    cos_angle_a1 = (L1**2 + D**2 - L2**2) / (2*L1*D)
    a1_rad = math.acos(apply_limits(cos_angle_a1, -1.0, 1.0))
    a2_rad = math.atan2(target_height, target_dist)
    shoulder_angle_rad = a1_rad + a2_rad

    #Convert to degrees
    shoulder_deg = math.degrees(shoulder_angle_rad)
    elbow_deg = math.degrees(elbow_angle_rad)

    #Apply hardware limits
    final_shoulder = apply_limits(shoulder_deg, SHOULDER_MIN, SHOULDER_MAX)
    final_elbow = apply_limits(elbow_deg, ELBOW_MIN, ELBOW_MAX)

    return {
        'shoulder': final_shoulder,
        'elbow': final_elbow
    }