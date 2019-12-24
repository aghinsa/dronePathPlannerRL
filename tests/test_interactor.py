from airsim_interactor import AirSimPlayer,PlayerConfig
import numpy as np

player_config  = PlayerConfig(scaling_factor=5,speed=5)
player = AirSimPlayer(player_config)
actions = [np.random.randint(6) for _ in range(100)]
for a in actions:
    target=player.interpretAction(a)
    player.moveToPoint(target=target,is_offset=True)
    print(player.getPosition())
