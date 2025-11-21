import requests
import time
import threading
import queue

# --Configuration--
PICO_IP = "192.168.178.96"  

BASE_URL = f"http://{PICO_IP}/set_angles"

#Last time we sent a command
last_request_time = 0
#Lag in between each request
REQUEST_THROTTLE = 0.5  #dont put 0.02 as it will lag the system too much

# maxsize=1 means we only keep the LATEST command. 
# If the network is slow, we drop old frames. This is perfect for robots.
command_queue = queue.Queue(maxsize=1)

def network_worker():
    """
    This function runs in the background forever.
    It waits for commands and sends them one by one.
    """
    # Use a Session for faster, persistent connections
    session = requests.Session()
    
    print("Arm Client: Background worker started...")
    
    while True:
        #Wait for a command to appear in the queue
        angles_dict = command_queue.get()
        
        try:
            # Build parameters
            params = {f"j{joint}": angle for joint, angle in angles_dict.items()}
            
            print(f"Sending: {params}...") 
            

            # We use a short timeout so the worker doesn't get stuck forever
            response = session.get(BASE_URL, params=params, timeout=0.5)
            
            if response.status_code != 200:
                print(f"Pico Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("...Pico Timed out (Skipping)")
        except Exception as e:
            print(f"Arm Error: {e}")
        finally:
            # Mark task as done so the queue knows we are ready
            command_queue.task_done()

#Start the Background Thread
# 'daemon=True' means this thread will die automatically when your app closes
worker_thread = threading.Thread(target=network_worker, daemon=True)
worker_thread.start()

def send_angles_to_arm(angles_dict):
    try:
        # Try to put the command in the queue.
        # block=False means: "If the worker is busy, just drop this command."
        # This automatically handles throttling!
        command_queue.put(angles_dict, block=False)
    except queue.Full:
        # This is GOOD. It means we are generating frames faster than 
        # the network can send them. We ignore this frame to prevent lag.
        pass





#Sends a dictionary of angles {0: 90, 1: 45} to the Pico.
def send_angles_to_arm_old(angles_dict):
    global last_request_time
    
    current_time = time.time()
    
    if (current_time - last_request_time) < REQUEST_THROTTLE:
        return  

    last_request_time = current_time
    
    #Build the URL query string, e.g., "j0=90&j1=45"
    params = {f"j{joint}": angle for joint, angle in angles_dict.items()}
    
    try:
        print(f"Attempting to send: {params} to {BASE_URL}")
        #Send the GET request
        requests.get(BASE_URL, params=params, timeout=0.5)
        print(f"   ...Success!")
        
    except requests.exceptions.Timeout:
        #If pico did not respond instantly
        print(f"   ...Failed: TIMEOUT.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending to arm: {e}")