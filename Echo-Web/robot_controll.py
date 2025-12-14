import math
#Arm client
from arm_client import send_angles_to_arm
#Inverse-kinematics
from ik_solver import calculate_ik

robot_state = {
    "base": 90,
    "reach": 15.0,
    "height": 10.0,
    "shoulder_angle": 90,
    "elbow_angle": 90,
    "gripper": 0,
    "wrist_angle": 90
}

#GLOBAL VALUES (START POSITION) Where hand is
current_base = 90
current_reach = 180
current_height = 100
gripper_angle = 120

current_reach_cm = 15
current_height_cm = 10


SHOULDER_MIN = 10  #120
SHOULDER_MAX = 170
ELBOW_MIN = 0
ELBOW_MAX = 160

#Control Sensitivity
SPEED = 6.0
X_DEADZONE = 0.0010
Y_DEADZONE = 0.002
Z_DEADZONE = 0.015

#for EMA where robot is
target_base = 90
target_reach = 180
target_height = 100

target_reach_cm = 15.0 
target_height_cm = 10.0

def ema_calc(target, prev_ang, a):
    smoothed_angle = (target * a) + (prev_ang * (1-a))
    return smoothed_angle

def controll_arm(pos_f, thumb_point, index_point):
    #CENTER POINT
    X_CENTER = 0.018
    Y_CENTER = 0.092
    Z_CENTER = 0.005

    global current_reach, current_base, current_height, gripper_angle
    global target_base, target_reach, target_height
    global target_base, target_reach_cm, target_height_cm, current_reach_cm, current_height_cm
    global SHOULDER_MIN, SHOULDER_MAX, ELBOW_MAX, ELBOW_MIN 

    alpha = 0.1     #steping weight

    #X_AXIS CONTROLL
    x_diff = pos_f[0] - X_CENTER

    #Deadzone
    if abs(x_diff) > X_DEADZONE:
        current_base += x_diff * 90 * SPEED

    #SAFETY
    current_base = max(0.0, min(180, current_base))

    #Y_AXIS CONTROLL
    y_diff = pos_f[1] - Y_CENTER

    if abs(y_diff) > Y_DEADZONE:
        #current_height -= y_diff * 50 * SPEED   #150
        current_height_cm -= y_diff * 80 * SPEED 

    #current_height = max(0, min(160, current_height))
    
    current_height_cm = max(0, min(35, current_height_cm)) #IK
    

    #Z_AXIS
    z_diff = pos_f[2] - Z_CENTER

    if abs(z_diff) > Z_DEADZONE:
        current_reach_cm -= z_diff * 50 * SPEED   #100

    #current_reach = max(120, min(180, current_reach))
    current_reach_cm = max(2, min(25, current_reach_cm)) #IK

    #Expotential Smoothing (EMA)
    target_base = ema_calc(current_base, target_base, alpha)
    target_height_cm = ema_calc(current_height_cm, target_height_cm, alpha)
    target_reach_cm = ema_calc(current_reach_cm, target_reach_cm, alpha)

    #current_reach = max(0, min(16, current_reach))
    #arm_angles = {'shoulder' : target_reach, 'elbow' : target_height}
    arm_angles = calculate_ik(current_reach_cm, current_height_cm)

    #---GRIPPER GESTURE CONTROLL
    distance = math.sqrt((thumb_point.x - index_point.x)**2 + (thumb_point.y - index_point.y)**2)
    GRIPPER_THRESHOLD = 0.02
    
    gripper_angle = 120
    if distance < GRIPPER_THRESHOLD:
        gripper_angle = 60

    #print(f"DIFF -> X: {x_diff:.3f} | Y: {y_diff:.3f} | Z: {z_diff:.3f}")
    #print(f"HEIGHT: {current_height}\nREACH: {current_reach}")
    #print(f"ZAXIS: diff:{z_diff:.3f}\nposf:{pos_f[2]}\nreach:{current_reach}")
    #print(f"reach:{current_reach}")
    #print(f"diff Y: {y_diff:.3f} | h: :{current_height}" )

    #SEND TO ARM
    if arm_angles:
        fin_shoulder_angle = round(arm_angles['shoulder'])
        fin_elbow_angle = round(arm_angles['elbow'])
        fin_current_base = round(target_base)
        wrist_angle = round(arm_angles['wrist'])

        fin_shoulder_angle = max(SHOULDER_MIN, min(SHOULDER_MAX, fin_shoulder_angle))
        fin_elbow_angle = max(ELBOW_MIN, min(ELBOW_MAX, fin_elbow_angle))

        #Diagram angles
        dia_shoulder = round(arm_angles['Diagram_shoulder'])

        # print(f"Pre Vals: B={round(current_base)}, H={round(current_height)}, R={round(current_reach)}\n"
        #   f"EMA Vals: B={fin_current_base}, H={fin_elbow_angle}, R={fin_shoulder_angle}\n")

        #print(f"limited reach:{shoulder_angle}")

        # print(f"Coords: ({smooth_reach_cm:.1f}, {smooth_height_cm:.1f}) -> Angles: B{fin_base} S{fin_shoulder} E{fin_elbow}")

        # UPDATE THE SHARED STATE DICTIONARY
        robot_state["base"] = fin_current_base
        robot_state["reach"] = round(target_reach_cm, 2)
        robot_state["height"] = round(target_height_cm, 2)
        robot_state["shoulder_angle"] = dia_shoulder
        robot_state["elbow_angle"] = fin_elbow_angle
        robot_state["gripper"] = gripper_angle
        robot_state["wrist_angle"] = wrist_angle

        send_angles_to_arm({
            0: fin_current_base,
            1: fin_shoulder_angle,
            2: fin_elbow_angle,
            3: wrist_angle,
            5: gripper_angle
        })
    else:
        print(f"Target Unreachable: Reach={target_reach_cm:.1f}cm, Height={target_height_cm:.1f}cm")