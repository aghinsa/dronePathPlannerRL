from agent import DQNAgent
from utils import AirSimPlayer,transform_input
import airsim
import time
import numpy as np

"""
To-Do
*  Set limit on z for airsim(z is in negative)
*  Check Reward accumulation
*  Checkpointing
"""
player = AirSimPlayer()
# train_after 10000
# target_update 10000 
agent = DQNAgent(num_actions=6,input_shape=[84,84,1],learning_rate=0.001,mini_batch_size=32,
        memory_size=500,train_after=5,train_interval=6,target_update_interval=10,
        history_length=4,epsilon=0.95,epsilon_decay=.995,decay_interval=3000)

ZLIMIT=-3 #to train on low heights ,so more chane of collision
ZLIMIT_VELOCITY=1
initX = -.55265
initY = -31.9786
initZ = -19.0225
initZ = max(initZ,ZLIMIT)
# setting initial position
player._client.moveToPositionAsync(initX, initY, initZ, 5).join()
player._client.moveByVelocityAsync(0, 0, 0, 5).join()
time.sleep(2) #time for airsim to setup 
num_target_achieved=0
num_actions_in_episode=0
episode_next_ratio = 0.3

source,target = player.initAnEpisode()
# see [https://github.com/microsoft/AirSim/blob/master/docs/image_apis.md]
responses = player._client.simGetImages([airsim.ImageRequest(3, 
                        airsim.ImageType.DepthPerspective, True, False)])
current_state = transform_input(responses)
current_state = np.expand_dims(current_state,axis=0)
print("Starting Training...")

while(True):
    action=agent.act(current_state)
    print(f"action : {action}")
    num_actions_in_episode+=1
    quad_offset=player.interpretAction(action,scaling_factor=0.25)
    quad_vel = player._client.getMultirotorState().kinematics_estimated.linear_velocity
    # player._client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], 
    #     quad_vel.z_val+quad_offset[2], 5).join()
    #for initail phase let velocity in z be 0
    player._client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], 
        0, 5).join()

    time.sleep(0.5)

    quad_state = player._client.getMultirotorState().kinematics_estimated.position
   # quad_state.z_val=max(quad_state.z_val,ZLIMIT)
    collision_info = player._client.simGetCollisionInfo()
    reward,done = player.computeReward(source,target,collision_info,quad_state,thresh_dist=.8)
    

    agent.remember(current_state,action,reward,done)
    agent.train()

    if done:
        print("One iteration done")
        if(reward>0):
            num_target_achieved+=1
            if( (num_target_achieved/num_actions_in_episode) > 0.7):
                source,target=player.updateSourceTarget(source,target,update_radius=True,step=3)
            elif(num_target_achieved/num_actions_in_episode > 0.3):
                source,target=player.updateSourceTarget(source,target,update_radius=False,step=3)
        else:
            player._client.moveToPositionAsync(initX, initY, initZ, 5).join()
            player._client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
            time.sleep(0.5)

    
    responses = player._client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
    current_state = transform_input(responses)
    current_state = np.expand_dims(current_state,axis=0)

