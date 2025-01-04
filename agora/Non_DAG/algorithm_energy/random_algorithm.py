import numpy as np
from cogito.alogrithm import Algorithm


class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RandomAlgorithm(Algorithm):
    def __init__(self, reward_giver, threshold=0.8):
        self.threshold = threshold
        self.current_trajectory = []
        self.reward_giver = reward_giver
        self.always_reward=False
        self.agent=None

    def __call__(self, cluster, clock):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        candidate_task = None
        candidate_machine = None
        all_candidates = []

        for machine in machines:
            for task in tasks:
                if machine.accommodate(task):
                    all_candidates.append((machine, task))
                    if np.random.rand() > self.threshold:
                        candidate_machine = machine
                        candidate_task = task
                        break
        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            return None, None
        if candidate_task is None:
            pair_index = np.random.randint(0, len(all_candidates))
            node = Node(None, None, 0, clock)
            self.current_trajectory.append(node)
            return all_candidates[pair_index]
        else:
            node = Node(None, None, 0, clock)
            self.current_trajectory.append(node)
            return candidate_machine, candidate_task,node
