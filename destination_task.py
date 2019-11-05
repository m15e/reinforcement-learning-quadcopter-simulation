import numpy as np
import math
from physics_sim import PhysicsSim

class Destination_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # original code form: https://github.com/parkitny/rl-project/blob/master/RL-Quadcopter-2/land_task.py
        
        # [angle]_penalty returns 1 if euler angle = 0 and increasingly smaller numbers as angle increases for flight stability
        # theta penalty is commented out to allow for freedom in pitch angle for flight control 
        # for angent to simply fly straight up uncomment theta_penalty row
        penalty = 1
        phi_penalty = (1. - abs(math.sin(self.sim.pose[3])))
        #theta_penalty = (1. - abs(math.sin(self.sim.pose[4])))
        psi_penalty = (1. - abs(math.sin(self.sim.pose[5])))

        penalty *= phi_penalty 
        penalty *= psi_penalty
        
        # calculate r => pythagorean distance from target
        delta = abs(self.sim.pose[:3] - self.target_pos)
        r = math.sqrt(np.dot(delta, delta))
        
        # function to keep reward within positive range
        if (r > 0.01): 
            decay = math.exp(-1/r) # limit range
        else: decay = 0
                           
        reward = 1. - decay
        
        # add stability penalty to reward
        reward *= penalty 
                           
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state