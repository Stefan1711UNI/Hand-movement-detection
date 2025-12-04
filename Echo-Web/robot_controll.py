import math
#Arm client
from arm_client import send_angles_to_arm
#Inverse-kinematics
from ik_solver import calculate_ik

#GLOBAL VALUES (START POSITION) Where hand is
current_base = 90
current_reach = 180
current_height = 100

#Control Sensitivity
SPEED = 6.0
X_DEADZONE = 0.008
Y_DEADZONE = 0.002
Z_DEADZONE = 0.015

#for EMA where robot is
target_base = 90
target_reach = 180
target_height = 100

def ema_calc(target, prev_ang, a):
    smoothed_angle = (target * a) + (prev_ang * (1-a))
    return smoothed_angle

def controll_arm(pos_f, thumb_point, index_point):
    #CENTER POINT
    X_CENTER = 0.018
    Y_CENTER = 0.092
    Z_CENTER = 0.005

    global current_reach, current_base, current_height
    global target_base, target_reach, target_height

    alpha = 0.1     #steping weight

    #X_AXIS CONTROLL
    x_diff = pos_f[0] - X_CENTER

    #Deadzone
    if abs(x_diff) > X_DEADZONE:
        current_base += x_diff * 100 * SPEED

    #SAFETY
    current_base = max(0.0, min(180, current_base))

    #Y_AXIS CONTROLL
    y_diff = pos_f[1] - Y_CENTER

    # if abs(y_diff) > Y_DEADZONE:
    #     current_height -= y_diff * 50 * SPEED
    if abs(y_diff) > Y_DEADZONE:
        current_height -= y_diff * 150 * SPEED

    current_height = max(0, min(160, current_height))
    
    #current_height = max(-5, min(18, current_height)) #IK
    

    #Z_AXIS
    z_diff = pos_f[2] - Z_CENTER

    # if abs(z_diff) > Z_DEADZONE:
    #     current_reach -= z_diff * 50 * SPEED
    if abs(z_diff) > Z_DEADZONE:
        current_reach -= z_diff * 100 * SPEED

    current_reach = max(120, min(180, current_reach))

    #Expotential Smoothing (EMA)
    target_base = ema_calc(current_base, target_base, alpha)
    target_height = ema_calc(current_height, target_height, alpha)
    target_reach = ema_calc(current_reach, target_reach, alpha)

    #current_reach = max(0, min(16, current_reach))
    arm_angles = {'shoulder' : target_reach, 'elbow' : target_height}
    #arm_angles = calculate_ik(current_reach, current_height)

    #---GRIPPER GESTURE CONTROLL
    distance = math.sqrt((thumb_point.x - index_point.x)**2 + (thumb_point.y - index_point.y)**2)
    GRIPPER_THRESHOLD = 0.02
    gripper_angle = 0
    if distance > GRIPPER_THRESHOLD:
        gripper_angle = 60

    #print(f"DIFF -> X: {x_diff:.3f} | Y: {y_diff:.3f} | Z: {z_diff:.3f}")
    #print(f"HEIGHT: {current_height}\nREACH: {current_reach}")
    #print(f"ZAXIS: diff:{z_diff:.3f}\nposf:{pos_f[2]}\nreach:{current_reach}")
    #print(f"reach:{current_reach}")
    print(f"diff Y: {y_diff:.3f} | h: :{current_height}" )

    #SEND TO ARM
    if arm_angles:
        fin_shoulder_angle = round(arm_angles['shoulder'])
        fin_elbow_angle = round(arm_angles['elbow'])
        fin_current_base = round(target_base)

        fin_shoulder_angle = max(120, min(180, fin_shoulder_angle))
        fin_elbow_angle = max(0, min(160, fin_elbow_angle))

        # print(f"Pre Vals: B={round(current_base)}, H={round(current_height)}, R={round(current_reach)}\n"
        #   f"EMA Vals: B={fin_current_base}, H={fin_elbow_angle}, R={fin_shoulder_angle}\n")

        #print(f"limited reach:{shoulder_angle}")
        send_angles_to_arm({
            0: fin_current_base,
            1: fin_shoulder_angle,
            2: fin_elbow_angle,
            5: gripper_angle
        })
    else:
        send_angles_to_arm({
            0: target_base,
            1: target_reach,
            2: current_height,
            5: gripper_angle
        })