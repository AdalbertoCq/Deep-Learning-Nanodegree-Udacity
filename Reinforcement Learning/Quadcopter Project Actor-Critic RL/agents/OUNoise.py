import numpy as np
import copy

class OUNoise:
    def __init__(self, action_space, mean, sigma, theta):
        self.mean = mean*np.ones(action_space)
        self.sigma = sigma
        self.theta = theta
        self.restart()
        
    def restart(self):
        self.current = copy.copy(self.mean)
        
    def sample(self):
        x = self.current
        dx = self.theta*(self.mean-x) + self.sigma*np.random.randn(len(x))
        
        self.current = x+dx
        return x+dx