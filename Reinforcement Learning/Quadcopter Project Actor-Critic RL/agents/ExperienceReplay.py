from collections import deque
import numpy as np

class ExperienceReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.mem = deque(maxlen=capacity)
        
    def add_env_reaction(self, env_reaction):
        # St, At, Rt1, Dt, St1.
        self.mem.append(env_reaction)
    
    def sample_batch(self, debug=False):
        indexes = np.random.choice(a=np.arange(len(self.mem)), size=self.batch_size, replace=False)
        if debug: print(indexes)
        states = list()
        actions = list()
        rewards = list()
        dones = list()
        next_states = list()
        for index in indexes:
            if self.mem[index] is None:
                print(self.mem[index])
            st, at, rt, dt, st_1 = self.mem[index]
            states.append(st)
            actions.append(at)
            rewards.append(rt)
            dones.append(dt)
            next_states.append(st_1)      
        return states, actions, rewards, dones, next_states