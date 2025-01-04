from abc import ABC, abstractmethod

class Node(object):
    def __init__(self, observation, action, proba,reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock
        self.proba=proba
class Algorithm(ABC):
    def __init__(self, reward_giver):
        self.reward_giver = reward_giver
        self.current_trajectory = []
        self.agent=None


    def log_trajectory(self, clock,observation=None,action=None,proba=None):


        self.current_trajectory.append(Node(observation, action, proba,self.reward_giver.get_reward(), clock))

    @abstractmethod
    def __call__(self, cluster, clock):
        pass
