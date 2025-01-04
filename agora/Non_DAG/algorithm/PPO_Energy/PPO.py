import torch as torch
import torch.nn.functional as F

import numpy as np


class Node(object):
    def __init__(self, observation, action, reward, clock,prob,valids,gnn_embeddings):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock
        self.prob=prob
        self.valid_candid=valids
        self.gnn_embeddings=gnn_embeddings


class RLAlgorithm(object):
    def __init__(self,epsilon, agent, reward_giver, features_normalize_func, features_extract_func):
        self.always_reward=True
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []
        self.epsilon=epsilon
    def log_trajectory(self, clock,observation,action,logprobs,valids,gnn_embeddings):
        # print(f'clock is {clock}')
        self.current_trajectory.append(Node(observation, action,self.reward_giver.get_reward(),clock,logprobs,valids,gnn_embeddings))

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            features.append([machine.cpu, machine.memory] + self.features_extract_func(task))
        features = self.features_normalize_func(features)
        return features

    def __call__(self, cluster, clock):
        machines = cluster.machines
        tasks = cluster.all_tasks
        valid_candidates = []
        all_candidates=[]
        # Check for valid task-machine pairs
        for machine in machines:
            for task in tasks:
                all_candidates.append((machine, task))
                if machine.accommodate(task) and task in cluster.tasks_which_has_waiting_instance:
                    valid_candidates.append((machine, task))

        if len(valid_candidates) == 0:
            # self.current_trajectory.append(Node(None, None, None, self.reward_giver.get_reward(), clock))
            return None, None,None,None,None,None,None
        else:
            valid_indices = [all_candidates.index(candidate) for candidate in valid_candidates]
            action_mask = torch.zeros(len(all_candidates))
            for candidate in valid_candidates:
                action_mask[all_candidates.index(candidate)] = 1

            # Create a tensor of action masks for valid candidates
            features = self.extract_features(all_candidates)
            features = torch.tensor(features)
            features = features.to(dtype=torch.float32)
            logits = self.agent.brain(features)

            masked_action_scores = logits.squeeze() * action_mask + (1 - action_mask) * -1e10

            # Apply softmax to the masked action scores
            logits = F.softmax(masked_action_scores, dim=0)
            #logits = F.softmax(logits, dim=0)
            logprobs = torch.log(logits)
            if len(logits) != 1:
                #logits = logits.squeeze()
                actions = np.arange(len(logits))  # Create an array of actions [0, 1, 2, ...]

                pair_index = np.random.choice(actions, p=logits.detach().numpy())
                logprob = logprobs[pair_index]
                # chosen_all_index = valid_indices[pair_index]
            else:
                pair_index = 0
                logprob = logprobs[pair_index]

            # node = Node(features, pair_index, logprob, 0, clock)
            # self.current_trajectory.append(node)

        return all_candidates[pair_index][0],all_candidates[pair_index][1],features,pair_index,logprobs,action_mask,None
