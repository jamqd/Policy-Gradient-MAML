"""

Runs experiments for VPG and TRPO including baselines

"""

from trust_region import *
from policy_gradient import *
import os

def main(run_vpg=True, run_trpo=True):
    # directory for storing models
    if not os.path.exists("./models"):
        os.mkdir("./models")

    # all have continuous action spaces
    envs = [
        'HalfCheetahForwardBackward-v1',  
        'AntForwardBackward-v1', 
        'AntDirection-v1', 
        'HumanoidForwardBackward-v1', 
        'HumanoidDirection-v1',
        'Particles2D-v1'
    ]
    
    # run vpg experiments

    if run_vpg:
        pass


    # run trpo experiments

    if run_trpo:
        pass






if __name__ == '__main__':
    main()