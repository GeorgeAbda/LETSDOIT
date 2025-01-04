import torch as torch
import torch.nn.functional as F

import numpy as np
device = torch.device("cpu")
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)


class Node(object):
    def __init__(self, observation, action, reward, clock,prob,valids,rlreward,embed,value=None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.rlreward=rlreward
        self.clock = clock
        self.prob=prob
        self.valid_candid=valids
        self.embed=embed
        self.value=value
np.random.seed(1)

class RLAlgorithm(object):
    def __init__(self, epsilon,agent, reward_giver, features_normalize_func, features_extract_func):
        self.always_reward=False
        self.agent = agent
        self.reward_giver = reward_giver

        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []
        self.epsilon=epsilon

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
        #
        # if len(cluster.tasks_which_has_waiting_instance) == 0:
        #     print('no candidates!!')
        #     self.current_trajectory.append(Node(None, None,self.reward_giver.get_reward() , clock,None,None,None))
        #     return None, None


        if len(all_candidates) == 0:
            return None, None, False
        if len(valid_candidates) == 0 and not  len(all_candidates) == 0:
            probs=torch.zeros((len(all_candidates)))
            probs[0] = float(-1e-20)  # Example for the first index

            node=Node(None, None,self.reward_giver.get_reward() , clock,probs,None,self.reward_giver.give_reward_rl(),None,0)
            self.current_trajectory.append(node)

            return None, None,node


        else:
            valid_indices = [all_candidates.index(candidate) for candidate in valid_candidates]
            action_mask = torch.zeros(len(all_candidates))
            for candidate in valid_candidates:
                action_mask[all_candidates.index(candidate)] = 1

            # Create a tensor of action masks for valid candidates
            features = self.extract_features(all_candidates)
            features = torch.tensor(features)
            features = features.to(dtype=torch.float32).to(device)
            logits = self.agent.actor(features)
            value=self.agent.critic(features)
            masked_action_scores = logits.squeeze() * action_mask + (1 - action_mask) * -1e9
            # print(f'masked_action_scores {masked_action_scores}')

            # Apply softmax to the masked action scores
            logits = F.softmax(masked_action_scores,-1)
            # print(f'softmax probs {logits}')

            # logits = F.softmax(logits, dim=0)
            # logprobs = torch.log(logits)
            p = logits.clone().detach().numpy()
            if len(logits) == 0:
                # If no valid actions, choose randomly
                chosen_index = np.random.choice(len(valid_candidates))
            # if np.random.random() < self.epsilon:
            #     chosen_index = np.random.choice(len(valid_candidates))
            #     # print(f"we will select chosen_index from randomly {chosen_index}")
            #
            #     chosen_all_index = valid_indices[chosen_index]
            else:
                chosen_index = np.random.choice(len(all_candidates), p=p)
                chosen_all_index = chosen_index
            chosen_pair = all_candidates[chosen_all_index]

            node = Node(features, chosen_all_index, 0, clock, logits, action_mask, self.reward_giver.give_reward_rl(), None, value)
            self.current_trajectory.append(node)
            # print('scheduling..')
        return chosen_pair[0], chosen_pair[1], node


def temperature_scaled_softmax(logits, temperature):
    """
    Apply temperature scaling to logits before softmax.

    Args:
        logits (numpy array): The input logits (pre-softmax activations).
        temperature (float): The temperature scaling parameter.

    Returns:
        numpy array: Scaled softmax probabilities.
    """
    scaled_logits = logits / temperature
    exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits, dim=-1, keepdim=True)[0])
    softmax_probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    return softmax_probs

def own_softmax(x):

    maxes = torch.max(x, -1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)

    return x_exp / x_exp_sum