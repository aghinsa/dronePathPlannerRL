import numpy as np
import airsim
import cv2
from autologging import traced


def logger(func):
    def wrap(*args,**kwargs):
        print(f"calling {func.__name__}")
        func(*args,**kwargs)
        print(f"returning from {func.__name__}")
    return wrap 


def generatePointInSphere(centre,radius):
    """
        Generates random point in sphere
    """

    # [a,b] -> (b-a) * np.random.random_sample() + a
    phi = 2 * np.pi * np.random.random_sample() #[0,2pi]
    costheta = 2*np.random.random_sample() - 1 #[-1,1]
    u = np.random.random_sample() #[0,1]

    theta = np.arccos( costheta )
    r = radius * np.power( u,1/3 )
    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    x+=centre[0]
    y+=centre[1]
    z+=centre[2]
    return (x,y,z)

@logger
def transform_input(responses,img_size=[84,84]):
    # list returned by airsim api
    #
    response = responses[0]
    
    # get numpy array
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    
    img = cv2.resize(img, dsize=tuple(img_size), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img,axis=-1)
    return img

@traced
class AirSimPlayer(object):
    # you must first press "1" in the AirSim view to turn on the depth capture
    # set initial positions before

    def __init__(self):
        #setting airsim connection
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        print("Connected to Airsim")
        self._client.enableApiControl(True)
        self._client.armDisarm(True) #arming drone
        
        # Async methods returns Future. Call join() to wait for task to complete.
        self._client.takeoffAsync().join()
        self._thresh_dist = 10
        self._radius = 3


    def interpretAction(self,action,scaling_factor=0.25):
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)    
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        return quad_offset


    def updateSourceTarget(self,curr_source,curr_target,update_radius=False,step=3):
        cx = (curr_target[0]-curr_source[0])*np.random.random_sample() + curr_source[0]
        cy = (curr_target[1]-curr_source[1])*np.random.random_sample() + curr_source[1]
        cz = (curr_target[2]-curr_source[2])*np.random.random_sample() + curr_source[2]
        if update_radius:
            self._radius += step
        centre=(cx,cy,cz)
        target=AirSimPlayer.generatePointInSphere(centre,self._radius)
        return centre,target

    def initAnEpisode(self):
        # returns [initial_position,target_position]
        quad_state = self._client.getMultirotorState().kinematics_estimated.position
        centre = (quad_state.x_val,quad_state.y_val,quad_state.z_val)
        target_point = AirSimPlayer.generatePointInSphere(centre,self._radius)
        return centre,target_point
    
    
    
    def computeReward(self,init_point,target_point,collision_info,quad_state,thresh_dist):
        done = False
        sta = lambda s: np.array([s.x_val,s.y_val,s.z_val]).reshape(-1,1) #state to array
        
        if collision_info.has_collided:
            reward = -100
            done=True
        elif (np.linalg.norm(sta(quad_state)-np.array(target_point).reshape(-1,1) ) < thresh_dist) :
            reward = +20
            done=True
        else:
            reward = -1
        
        return reward,done
    
   



   



