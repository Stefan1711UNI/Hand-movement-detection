import requests
import time

# --Configuration--
PICO_IP = "192.168.178.96"  

BASE_URL = f"http://{PICO_IP}/set_angles"

#Last time we sent a command
last_request_time = 0
#Lag in between each request
REQUEST_THROTTLE = 0.05  

#Sends a dictionary of angles {0: 90, 1: 45} to the Pico.
def send_angles_to_arm(angles_dict):
    global last_request_time
    
    current_time = time.time()
    
    if (current_time - last_request_time) < REQUEST_THROTTLE:
        return  

    last_request_time = current_time
    
    #Build the URL query string, e.g., "j0=90&j1=45"
    params = {f"j{joint}": angle for joint, angle in angles_dict.items()}
    
    try:
        #Send the GET request
        requests.get(BASE_URL, params=params, timeout=0.1)
        print(f"Sent angles: {params}")
        
    except requests.exceptions.Timeout:
        #If pico did not respond instantly
        pass
    except requests.exceptions.RequestException as e:
        print(f"Error sending to arm: {e}")