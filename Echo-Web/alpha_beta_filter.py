import numpy as np

class a_b_filter:
    "Alpha Beta filter. Uses velocity to help decide on filtering"
    def __init__(self, initial_time=None): 

        self.position = np.zeros(3)      
        self.velocity = np.zeros(3)

        self.last_time = initial_time

        #Camera VS History TUNING
        #0.0-1.0| Higher(Trust camera readings more) Low(Trust history more)
        self.alpha = 0.4 #was 0.4 jittery 
        
        #0.0-1.0| Higher(Faster change from movement) Low(Slower momentum change)
        self.beta = 0.01 #was 0.1 too much overshoot


    def step(self, timestamp_s, z):
        if self.last_time is None:
            self.last_time = timestamp_s
            self.position = np.array(z) #If first instance set array to current data
            return True, 0.0
        
        dt = timestamp_s - self.last_time
        self.last_time = timestamp_s
        
        # Safety for very small time steps
        if dt <= 0: dt = 0.001

        measurement = np.array(z)

        #Predict
        #Probly close to where i was plus the current speed
        self.position = self.position + (self.velocity * dt)

        #Calculate Error
        #How wrong is the prediction to camera data
        error = measurement - self.position

        #Update Position
        #Update position towards camera data
        self.position = self.position + (self.alpha * error)

        #Update Velocity
        #Change velocity by error rate / time
        self.velocity = self.velocity + (self.beta * error / dt)

        return True


    def get_state(self):
        #Return position and velocity
        return self.position, self.velocity