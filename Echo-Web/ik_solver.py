import math

#Arm Link Lengths (cm)
L1 = 12.0  #Shoulder to Elbow
L2 = 13.0  #Elbow to Gripper

#Hardware Limits (Degrees)
SHOULDER_MIN = 10   
SHOULDER_MAX = 170
ELBOW_MIN = 0
ELBOW_MAX = 160
WRIST_MIN = 10
WRIST_MAX = 170

#OFF-SETS Shoulder angle is off set from the x-axis by 60 degrees
SHOULDER_OFF = 60

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


    #Calc geometry Law Of Cosines
    cos_angle_b = (L1**2 + L2**2 - D**2) / (2*L1*L2)
    b_angle_rad = math.acos(apply_limits(cos_angle_b, -1.0, 1.0))
    elbow_angle_rad = math.pi - b_angle_rad     #Inner elbow angle

    #Shoulder angle (A1 + A2)
    cos_angle_a1 = (L1**2 + D**2 - L2**2) / (2*L1*D)    #Law Of Cosines
    a1_rad = math.acos(apply_limits(cos_angle_a1, -1.0, 1.0))
    a2_rad = math.atan2(target_height, target_dist)     
    shoulder_angle_rad = a1_rad + a2_rad

    #Calculate c2 (Angle between Line D and Horizontal tH)
    #In the right triangle, this is (90 - a2)
    #90 degrees = 1.571 radians
    # c2 = 1.571 - a2_rad
    # #Calculate f1 (Angle from gripper link to Line D)
    # #90 is added because tH is 90 degrees away from the gripper link (Gripper link is parralal to x-axis)
    # f1 = 1.571 + c2
    # #Calculate f2 (Angle from Line D to Link L2)
    # #Inside the robot triangle: 180 - a1 - b
    # #180 degrees = 3.14159 radians
    # f2 = 3.14159 - a1_rad - elbow_angle_rad
    # #Total Angle (Gripper link to L2)
    # f_rad = f1 + f2
    
    # #Now to get the angle the servo must point to always be parralal to the x-axis
    # #we get the angel from L2 to the vertical line(tH)
    # l2_to_tH_angel = f2 + c2
    # #Now we calculate the difference from the previouse angel to the gripper link 
    # servo_angle_rad = -l2_to_tH_angel
    # wrist_angle = math.degrees(servo_angle_rad)
    # print(f"Wrist angle: {wrist_angle}\nf2+c2{l2_to_tH_angel}")

    #NEW
    e1_rad = shoulder_angle_rad
    #Calculate the Deflection (How much the elbow is bent)
    #The deflection is the supplementary angle to the inner elbow angle.
    deflection = math.pi - elbow_angle_rad

    #e2
    e2_rad = math.pi - e1_rad

    #Calculate Global Angle of Link 2 (d_rad)
    d_rad = math.pi - e2_rad - deflection

    #Calculate Servo Angle (F-Rule)
    servo_target_rad =  -d_rad 

    wrist_angle = math.degrees(servo_target_rad)

    wrist_angle = wrist_angle - 70

    #print(f"d_degree: {math.degrees(d_rad)}")
    print(f"Global D: {math.degrees(d_rad):.1f} | Target Wrist: {wrist_angle:.1f}")


    #Convert to degrees
    shoulder_deg = math.degrees(shoulder_angle_rad)
    elbow_deg = math.degrees(elbow_angle_rad)

    #Diagram angle unaffected by offset
    dia_shoulder = apply_limits(shoulder_deg, SHOULDER_MIN, SHOULDER_MAX)

    #When the arm is flat (Math = 180°), the Servo is physically at 60°.
    #Therefore: Servo_Angle = Math_Angle + 60
    shoulder_deg = shoulder_deg + SHOULDER_OFF

    #Apply hardware limits
    final_shoulder = apply_limits(shoulder_deg, SHOULDER_MIN, SHOULDER_MAX)
    final_elbow = apply_limits(elbow_deg, ELBOW_MIN, ELBOW_MAX)
    wrist_angle = apply_limits(wrist_angle, WRIST_MIN, WRIST_MAX)

    return {
        'shoulder': final_shoulder,
        'elbow': final_elbow,
        'Diagram_shoulder': dia_shoulder,
        'wrist': wrist_angle
    }


def calculate_wrist_angle(e1_rad, elbow_angle_rad, is_elbow_up):
    # 1. Calculate the Elbow Deflection (How much L2 turns relative to L1)
    # If the arm is straight, deflection is 0.
    # If b=90, deflection is 90.
    deflection = math.pi - elbow_angle_rad
    
    # 2. Apply direction based on Up/Down configuration
    # Note: Adjust these signs based on your specific coordinate system!
    if is_elbow_up:
        # If elbow goes UP, the link usually points DOWN relative to the extension
        d_rad = e1_rad - deflection
    else:
        # If elbow goes DOWN, the link points UP relative to the extension
        d_rad = e1_rad + deflection

    # 3. Calculate Servo Target (The "F-Rule")
    # We want the global angle to be 0 (Horizontal).
    # Current global angle is 'd'.
    # We add a 90 degree (1.571 rad) OFFSET because the servo center is 90.
    servo_rad = 1.571 - d_rad 
    
    # 4. Convert to Degrees
    servo_deg = math.degrees(servo_rad)
    
    # 5. Handle the 0-180 Limit (Clamping)
    if servo_deg < 0:
        print(f"WARNING: Angle {servo_deg:.1f} is too low for servo!")
        servo_deg = 0
    elif servo_deg > 180:
        print(f"WARNING: Angle {servo_deg:.1f} is too high for servo!")
        servo_deg = 180
        
    return servo_deg