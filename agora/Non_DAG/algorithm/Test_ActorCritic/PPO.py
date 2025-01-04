import torch as torch
import torch.nn.functional as F

import numpy as np


class Node(object):
    def __init__(self, observation, action, reward, clock,prob,valids,rlreward):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.rlreward=rlreward
        self.clock = clock
        self.prob=prob
        self.valid_candid=valids


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []
        self.always_reward=False

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
        all_candidates = []
        # Check for valid task-machine pairs
        for machine in machines:
            for task in tasks:
                all_candidates.append((machine, task))
                if machine.accommodate(task) and task in cluster.tasks_which_has_waiting_instance:
                    valid_candidates.append((machine, task))

        if len(valid_candidates) == 0:
            # print('no candidates!!')
            self.current_trajectory.append(Node(None, None,self.reward_giver.get_reward() , clock,None,None,self.reward_giver.give_reward_rl()))
            return None, None
        else:
            valid_indices = [all_candidates.index(candidate) for candidate in valid_candidates]
            action_mask = torch.zeros(len(all_candidates))
            for candidate in valid_candidates:
                action_mask[all_candidates.index(candidate)] = 1

            # Create a tensor of action masks for valid candidates
            features = self.extract_features(all_candidates)
            features = torch.tensor(features)
            features = features.to(dtype=torch.float32)
            # print(f'features {features}')

            logits = self.agent.brain(features)
            # print(f'logits brain {logits}')

            masked_action_scores = logits.squeeze() * action_mask + (1 - action_mask) * -1e10

            # Apply softmax to the masked action scores
            logits = F.softmax(masked_action_scores, dim=0)
            # print(f'logits {logits}')
            # logits = F.softmax(logits, dim=0)
            logprobs = torch.log(logits)
            if len(logits) != 1:
                # logits = logits.squeeze()
                actions = np.arange(len(logits))  # Create an array of actions [0, 1, 2, ...]

                pair_index = np.random.choice(actions, p=logits.detach().numpy())

                logprob = logprobs[pair_index]
                # chosen_all_index = valid_indices[pair_index]
            else:
                pair_index = 0
                logprob = logprobs[pair_index]

            node = Node(features,pair_index,0,clock,logprob,action_mask,0)
            self.current_trajectory.append(node)

        return all_candidates[pair_index]
