import time
import socket


# --Configuration--
PICO_IP = "192.168.178.96"  
PICO_PORT = 5000


#UDP METHOD
#crete socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_sent = 0
DELAY = 0.05 # 20 times a second

def send_angles_to_arm(angles_dict):
    global last_sent
    if time.time() - last_sent < DELAY:
        return
    
    params = {f"j{joint}": angle for joint, angle in angles_dict.items()}
    message_str = "&".join([f"{key}={value}" for key, value in params.items()])
    
    try:
        print(f"Attempting to send: {params}")
        sock.sendto(message_str.encode(), (PICO_IP, PICO_PORT))
        last_sent = time.time()
    
    except Exception as e:
        print(f"UDP Error: {e}")