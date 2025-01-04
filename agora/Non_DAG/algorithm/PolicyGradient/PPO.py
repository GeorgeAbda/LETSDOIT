import torch as torch
import torch.nn.functional as F

import numpy as np


class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            features.append([machine.cpu, machine.memory] + self.features_extract_func(task))
        features = self.features_normalize_func(features)
        return features

    def __call__(self, cluster, clock):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        all_candidates = []

        for machine in machines:
            for task in tasks:
                if machine.accommodate(task):
                    all_candidates.append((machine, task))
        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            return None,None,None,None,None,None,None
        else:
            features = self.extract_features(all_candidates)
            features = torch.tensor(features)
            features = features.to(dtype=torch.float32)
            logits = self.agent.brain(features)
            logits = F.softmax(logits, dim=0)
            if len(logits) != 1:
                logits = logits.squeeze()
                actions = np.arange(len(logits))  # Create an array of actions [0, 1, 2, ...]

                pair_index = np.random.choice(actions, p=logits.detach().numpy())
            else:
                pair_index = 0
            #
            # node = Node(features, pair_index, 0, clock)
            # self.current_trajectory.append(node)

        return all_candidates[pair_index],None,None,None,None,None,None
