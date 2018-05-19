import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.20
        self.gamma = 1

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # Implementation for e-greedy policy.
        action_max = np.argmax(self.Q[state])
        probabilities = np.ones(self.nA) * epsilon / self.nA
        probabilities[action_max] = 1 - epsilon + epsilon / self.nA
        return np.random.choice(a=np.arange(self.nA), p=probabilities)

    def expected_reward(self, next_state, epsilon):
        probabilities = np.ones(self.nA) * epsilon / self.nA
        action_max = np.argmax(self.Q[next_state])
        probabilities[action_max] = 1 - epsilon + epsilon / self.nA
        expected = 0
        for action, prob in enumerate(probabilities):
            expected += prob * self.Q[next_state][action]
        return expected

    def step(self, state, action, reward, next_state, done, epsilon):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q-Leaning.
        #         self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action])

        # Sarsa expected.
        expected = self.expected_reward(next_state, epsilon)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * expected - self.Q[state][action])

        # Sarsa
#         next_action = self.select_action(next_state, epsilon)
#         self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action])