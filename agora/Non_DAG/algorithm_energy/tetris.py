import numpy as np
from cogito.alogrithm import Algorithm


class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class Tetris(Algorithm):

    def __init__(self, reward_giver):
        self.current_trajectory = []
        self.reward_giver = reward_giver
        self.ga = False
        self.always_reward=False
        self.agent=None

    @staticmethod
    def calculate_alignment(valid_pairs):
        machine_features = []
        task_features = []
        for index, pair in enumerate(valid_pairs):
            machine = pair[0]
            task = pair[1]
            machine_features.append(machine.feature[:2])
            task_features.append([task.task_config.cpu, task.task_config.memory])
        return np.argmax(np.sum(np.array(machine_features) * np.array(task_features), axis=1), axis=0)

    def __call__(self, cluster, clock):
        machines = [machine for machine in cluster.machines if machine.is_working]
        tasks = cluster.tasks_which_has_waiting_instance
        valid_pairs = []
        for machine in machines:
            for task in tasks:
                if machine.accommodate(task):
                    valid_pairs.append((machine, task))

        if len(valid_pairs) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            return None, None

        self.current_trajectory.append(Node(None, None, 0, clock))
        pair_index = Tetris.calculate_alignment(valid_pairs)
        pair = valid_pairs[pair_index]
        return pair[0], pair[1]
