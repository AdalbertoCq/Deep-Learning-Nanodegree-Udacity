import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None, action_repeat=3, debug=False):
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
        self.action_repeat = int(action_repeat)


        self.state_size = self.action_repeat * self.sim.pose.shape[0]
        self.action_low = 1
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime
        self.debug = debug

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        
        # Reward for X,Y axis.
        reward_position = 0.4*(1-0.01*abs(self.sim.pose[:2] - self.target_pos[:2]).sum())
        # Reinforce more movement across z axis.
        reward_z_axis = 1-0.01*abs(self.sim.pose[2] - self.target_pos[2])
        
        
        ## Need to find a way to handle velocities and penalize the ones that move the target away.
        # Take destination as reference, substract point from current position taking destiantion as axis origin reference.
        # Given position and current acceleration, reward it.
        # If negative, below objective, positive acceleration.
        #reference_position_z = self.sim.pose[2] - self.target_pos[2]
        #reward_accel_z = np.clip(a=-reference_position_z*self.sim.linear_accel[2]/5, a_min=-1, a_max=+1)
        #reference_position_h = self.sim.pose[:2] - self.target_pos[:2]
        #reward_accel_h = 0.3*np.clip(a=-reference_position_h*self.sim.linear_accel[:2]/5, a_min=-1, a_max=+1).sum()
        
        # Reward speed on z axis.
        reward_speed_z = 0.2*np.clip(a=self.sim.v[2]/5, a_min=-1, a_max=+1)
        #reward_accel_z = np.clip(a=self.sim.linear_accel[2]/5, a_min=-1, a_max=+1)
        
        reward = (reward_position + reward_z_axis)/1.4
        
        # Reward getting to the requested hight.
        if self.sim.pose[2] > self.target_pos[2]:
            reward += 5
            done = True
        # Penalize crash.
        #elif done and self.sim.time < self.runtime:
        #    reward += -10
        #    done = True
        # Penalize moving but not getting to the requested hight.
        #elif self.sim.time > self.runtime:
        #    reward += -5
        #    done = True
        
        if self.debug:
            print('------ Step --------')
            print('Position', self.sim.pose[:3], 'Z Speed', self.sim.v[2])
            print()
            print('Position reward', reward_position)
            print('Position Z reward', reward_z_axis)
            #print('Speed Z reward', reward_speed_z)
            print('Final Reward', reward)
        return reward, done

    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for i in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            
            reward_step, done = self.get_reward(done) 
            reward += reward_step
            pose_all.append(self.sim.pose)
            if done:
                while len(pose_all)<self.action_repeat:
                    pose_all.append(self.sim.pose)
                break
        
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # self.sim.pose[:3] = [0., 0. , np.random.normal(0.5, 0.1)]
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state