import numpy as np
import airsim
from dataclasses import dataclass

#TODO
# Move by pitch,yaw,roll

@dataclass
class PlayerConfig:
    """
    scaling_factor : amount by which curr_possition is offset see actions
    speed : speed 
    """
    scaling_factor : float
    speed : float


class AirSimPlayer(object):
    """ 
    you must first press "1" in the AirSim view to turn on the depth capture
     set initial positions before

    
    """

    def __init__(self,config:PlayerConfig):
        self.config=config
        #setting airsim connection
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        print("Connected to Airsim")
        self._client.enableApiControl(True)
        self._client.armDisarm(True) #arming drone
        
        # Async methods returns Future. Call join() to wait for task to complete.
        self._client.takeoffAsync().join()
    
    def interpretAction(self,action):
        scaling_factor=self.config.scaling_factor
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
        return list(quad_offset)

    def moveToPoint(self,target,is_offset=True):
        """
        target=[x,y,z],
        if `is_offset` off set is added to current position
        """
        
        if(is_offset):
            curr_pos = self.getPosition()
            for i in range(3):
                target[i]+=curr_pos[i]
                
        self._client.moveToPositionAsync(*target,self.config.speed).join()
        return self._client.simGetCollisionInfo().has_collided
        
    def getVisualResponse(self):
        """
        Returns visual responses stacked on depth axis
        """
        responses = self._client.simGetImages(
                        [
                            airsim.ImageRequest("0", airsim.ImageType.Infrared,
                                          False, False),
                            airsim.ImageRequest("0", airsim.ImageType.Scene, 
                                          False, False),
                            airsim.ImageRequest("0", airsim.ImageType.Segmentation, 
                                          False, False),
                            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
                            ]
                            )
        im_reshaped = []
        for response in responses[:-1] :
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            im = img1d.reshape(response.height, response.width, 4) 
            im_reshaped.append(im)
        response = responses[-1]
        img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        img = np.expand_dims(img,axis=-1)
        im_reshaped.append(img)
        vis_response = np.vstack(im_reshaped)
        
        return vis_response
    
    def getPosition(self):
        s = self._client.getMultirotorState().kinematics_estimated.position
        return [s.x_val,s.y_val,s.z_val]

