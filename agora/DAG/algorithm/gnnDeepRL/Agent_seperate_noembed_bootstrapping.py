import copy
import numpy as np
import torch as torch
from torch.optim import Adam
from agora.Non_DAG.algorithm.PPO_Energy.Networks import FeedForwardNN
from torch.optim.lr_scheduler import ExponentialLR
import time
import copy
from torch_geometric.data import Batch
from torch_geometric.data import Data
from agora.DAG.algorithm.gnnDeepRL.utils import *

import numpy as np
import torch
from agora.DAG.algorithm.gnnDeepRL.brain import Critic_noembed,Actor_noembed,CloudResourceGNN, train_gnn_ac,Actor,Actorv1,Critic,CloudResourceGNNv1,ActorCritic,ActorCritic_noembed
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
node_feature_dim = 12  # Adjust based on your DAG node features
resource_feature_dim = 7  # Adjust based on your resource features
hidden_dim = 32
gnn_model = CloudResourceGNN(node_feature_dim, resource_feature_dim, hidden_dim)
hidden_dim_critic=hidden_dim
from torch.optim import Adam
import torch
torch.autograd.set_detect_anomaly(True)
device = torch.device("cpu")

class Agent(object):
    def __init__(self, gamma, epsilon, reward_to_go, nn_baseline, normalize_advantages, experience_buffer,entropy_coefficient=0.01,lr=0.001,batch_size=1000,model_save_path=None
                 ):
        super().__init__()
        self.batch_size=batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.name='Bootstrapping'
        self.reward_to_go = reward_to_go
        self.baseline = nn_baseline
        self.normalize_advantages = normalize_advantages
        self.entropy_coefficient=entropy_coefficient
        self.gnn=CloudResourceGNNv1(node_feature_dim, resource_feature_dim, hidden_dim)
        #self.brain = Actorv1(self.gnn, hidden_dim)  # Policy network
        self.experience_buffer=experience_buffer
        self.actor=Actor_noembed(self.gnn, hidden_dim)
        self.critic=Critic_noembed(self.gnn, hidden_dim)
        # self.brain = FeedForwardNN(9)
        for param in self.actor.parameters():
            param.data = param.data.to(dtype=torch.float32)
        self.lr=lr
        # self.neighbor = copy.deepcopy(self.brain)
        # self.neighbor.load_state_dict(self.brain.state_dict())
        # for param in self.neighbor.parameters():
        #     param.requires_grad = False

        self.count = 0

        self.optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=self.lr)

        self.model_save_path = model_save_path


    def update(self,Qval,value,value_,logprobs,action,reward):
        td_error  = reward + self.gamma * Qval - value
        critic_loss = 0.5 * td_error .pow(2)  # Equivalent to (TD error) ^ 2 / 2

        self.optimizer.zero_grad()
        # ac_loss.backward()


        self.optimizer_critic.zero_grad()
        # states = torch.tensor(states, dtype=torch.float).to(device)
        # cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
        # values = value_func(states)
        # values = values.squeeze(dim=1)

        critic_loss.mean().backward()
        self.optimizer_critic.step()

        # Optimize policy loss (Actor)
        advantages = reward + self.gamma * Qval - value_
        #
        # with torch.no_grad():
        #     values = value_func(states)
        # actions = torch.tensor(actions, dtype=torch.int64).to(device)
        # advantages = cum_rewards - values
        # logits = actor_func(states)
        action_prob = logprobs.gather(0, torch.tensor(action).unsqueeze(0))
        # print(f'log probs {logprobs}')
        entropy = -torch.sum(logprobs.exp() * logprobs, dim=-1)
        entropy_coef = 0.015  # Small positive value to control the impact of entropy

        # Compute the policy loss
        policy_loss = -action_prob * advantages
        policy_loss = policy_loss + entropy_coef * entropy

        # pi_loss = log_probs * advantages
        # print(f'advantages {advantages}')
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        policy_loss.backward()
        self.optimizer.step()
        print(f'policy loss is {policy_loss}')
        return policy_loss.item(),critic_loss.item()



    def _sum_of_rewards(self, rewards_n):
        """
            Monte Carlo estimation of the Q function.

            arguments:
                rewards_n: shape(...).

            returns:
                q_n: shape: (...).
            ----------------------------------------------------------------------------------

            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages.

            Recall that the expression for the policy gradient PG is

                  PG = E_{tau} [sum_{cogito=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

            where

                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time cogito, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t.

            You will write code for two cases, controlled by the flag 'reward_to_go':

              Case 1: trajectory-based PG

                  (reward_to_go = False)

                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
                  entire trajectory (regardless of which time step the Q-value should be for).

                  For this case, the policy gradient estimator is

                      E_{tau} [sum_{cogito=0}^T grad log pi(a_t|s_t) * Ret(tau)]

                  where

                      Ret(tau) = sum_{cogito'=0}^T gamma^cogito' r_{cogito'}.

                  Thus, you should compute

                      Q_t = Ret(tau)

              Case 2: reward-to-go PG

                  (reward_to_go = True)

                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step cogito. Thus, you should compute

                      Q_t = sum_{cogito'=cogito}^T gamma^(cogito'-cogito) * r_{cogito'}


            Store the Q-values for all timesteps and all trajectories in a variable 'q_n'.
        """
        q_s = []
        for re in rewards_n:
            q = []
            cur_q = 0
            for reward in reversed(re):
                cur_q = cur_q * self.gamma + reward
                q.append(cur_q)
            q = list(reversed(q))
            q_s.append(q)

        if self.reward_to_go:
            return q_s
        else:
            q_n = []
            for q in q_s:
                q_n.append([q[0]] * len(q))
            return q_n

    def _compute_advantage(self, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            arguments:
                q_n: shape: (...).

            returns:
                adv_n: shape: (...).
        """
        # Computing Baselines
        if self.baseline:
            adv_n = copy.deepcopy(q_n)
            max_length = max([len(adv) for adv in adv_n])
            for adv in adv_n:
                while len(adv) < max_length:
                    adv.append(0)
            adv_n = np.array(adv_n)
            adv_n = adv_n - adv_n.mean(axis=0)

            adv_n__ = []
            for i in range(adv_n.shape[0]):
                original_length = len(q_n[i])
                adv_n__.append(list(adv_n[i][:original_length]))
            return adv_n__
        else:
            adv_n = q_n.copy()
            return adv_n

    def estimate_return(self, rewards_n):
        """
            Estimates the returns over a set of trajectories.


            arguments:
                re_n: shape(...).

            returns:
                q_n: shape: (...).
                adv_n: shape: (...).
        """
        q_n = self._sum_of_rewards(rewards_n)
        adv_n = self._compute_advantage(q_n)

        # Advantage Normalization
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_s = []
            for advantages in adv_n:
                for advantage in advantages:
                    adv_s.append(advantage)
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
            adv_n__ = []
            for advantages in adv_n:
                advantages__ = []
                for advantage in advantages:
                    advantages__.append((advantage - mean) / (std + np.finfo(np.float32).eps))
                adv_n__.append(advantages__)
            adv_n = adv_n__
        return q_n, adv_n

    # Function to compare two state dictionaries


    def _loss(self, X, y, advantage,mask):
        # print(f'X shape : {X.shape}')
        # print(f'Y : {y}')
        with torch.autograd.set_detect_anomaly(True):
            print("X requires_grad update:", X.requires_grad)  # True
            for param in self.brain.parameters():
                print(param.requires_grad)  # Should be True for all parameters

            logits = self.brain(X,resource_features=None,update=True).to(dtype=torch.float32)
            logits_ = logits * mask + (1 - mask) * 1e-2

            logits2 = self.neighbor(X,resource_features=None,update=True).to(dtype=torch.float32).detach()
            logits2_ = logits2 * mask + (1 - mask) * 1e-2
            state_dict1 = self.brain.state_dict()
            state_dict2 = self.neighbor.state_dict()
            # if are_state_dicts_equal(state_dict1, state_dict2):
            #     print("The two networks have identical parameters.")
            # else:
            #     print("The networks have different parameters.")
            # print(f'logits shape : {logits.shape}')
            # print(f'logits2 shape : {logits2.shape}')
            if len(logits) == 1:
                logits.unsqueeze(0)
            else:
                logits = logits.squeeze().unsqueeze(0)
            print(f'logits shape : {logits.shape}')

            if len(logits2) == 1:
                logits2.unsqueeze(0)
            else:
                logits2 = logits.squeeze().unsqueeze(0)
            # print(f'logits shape : {logits.shape}')
            print(f'logits2 shape : {logits2.shape}')
            # logprobs2 = torch.add(logits2, 0.01)
            # Use log_softmax for stability
            logprobs = torch.log_softmax(logits_, dim=-1)
            logprobs2 = torch.log_softmax(logits2_ + 0.01, dim=-1)
            # logprobs = torch.log(logits)
            print(f'logprobs shape : {logprobs.shape}')

            logprob1 = logprobs[y[0]]
            logprob2 = logprobs2[y[0]]

            log_ratio = logprobs - logprobs2
            ratio_ = torch.exp(logprobs - logprobs2)
            # print(f'ratio is {ratio_}')
            # print(f'log_ratio is {log_ratio}')
            ratio = torch.exp(logprob1 - logprob2)

            y = torch.tensor(y)
            y = y.long()
            with torch.no_grad():
                kl_div = torch.sum(torch.exp(logprobs) * (logprobs - logprobs2), dim=-1).mean().item()
                print(f"Exact KL Divergence: {kl_div}")
                approx_kl_div = ((ratio_ - 1) - log_ratio).mean().item()
                # print(f'approx_kl_div is {approx_kl_div}')
            loss = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                entropy = -torch.mean(torch.sum(torch.exp(logprobs) * logprobs, dim=-1))
            entropy_term=self.entropy_coefficient * entropy
            # logprob = - loss(logits, y)
            # surr1 = ratio * logprob * advantage
            surr1 = ratio * advantage
            surr2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            ppo_loss = -torch.min(surr1, surr2)

            # print(f'advantage is {advantage}')
            print(f'entropy_term shape is {entropy_term.shape} , type is {entropy_term.dtype}')

            print(f' PPO loss shape is {ppo_loss.shape} type is {ppo_loss.dtype}')
        return (ppo_loss+entropy_term),kl_div,entropy

    def update_parameters(self, all_observations, all_actions, all_advantages,all_masks):
        """
            Update the parameters of the policy.

            arguments:
                all_observations: shape: (...)
                all_actions: shape: (...).
                all_advantages: shape: (...)

            returns:
                nothing
        """
        loss_values = []
        advantages__ = []
        nb_trajectories = len(all_observations)
        entropy_losses=[]
        approx_kl_divs=[]
        for observations, actions, advantages,masks in zip(all_observations, all_actions, all_advantages,all_masks):
            grads_by_trajectory = []
            cnt = 1
            for observation, action, advantage,mask in zip(observations, actions, advantages,masks):
                if observation is None or action is None:

                    continue
                # print(f' len(observations) {len(observations)}')
                with torch.autograd.detect_anomaly(True):
                    print(f' self.batch_size: {self.batch_size}')
                    print(f' cnt: {cnt}')
                    self.optimizer.zero_grad()  # Clear previous gradients
                    loss_value,approx_kl_div,entropy = self._loss(observation, [action], advantage,mask)
                    print(f'loss is {loss_value}')

                    loss_value.backward()  # Compute gradients using autograd                # entropy_losses += [entropy]
                    approx_kl_divs.append(approx_kl_div)
                    grads = [param.grad for param in self.brain.parameters()]  # Get gradients
                    # print(f'grads {grads}')
                    grads_by_trajectory.append(grads)



                    loss_values.append(loss_value.item())
                    advantages__.append(advantage)
                    print(f' self.batch_size: {self.batch_size}')
                    print(f' cnt: {cnt}')

                if cnt % self.batch_size == 0:
                    # print('doing update')
                    self.count += 1
                    before = copy.deepcopy(self.brain.state_dict())
                    for name, param in self.brain.named_parameters():
                        print(name, param.grad)

                    self.optimize(grads_by_trajectory)
                    after = self.brain.state_dict()

                    grads_by_trajectory = []
                    # print(f'cnt is {cnt}')
                    # print(f'self.count is {self.count}')

                    # if self.count % 5 == 0:
                    #     self.neighbor.load_state_dict(self.brain.state_dict())
                    for name, param in self.brain.named_parameters():
                        print(name, param.requires_grad)

                    #     param.requires_grad = False
                    if self.count % 5 == 0:
                        # print("Before transfer:")
                        # print("Brain state_dict sample:", list(before.items())[:2])
                        # print("Neighbor state_dict sample:", list(self.neighbor.state_dict().items())[:2])

                        self.neighbor.load_state_dict(after)

                        # print("After transfer:")
                        # print("Brain state_dict sample:", list(before.items())[:2])
                        # print("Neighbor state_dict sample:", list(self.neighbor.state_dict().items())[:2])

                cnt += 1

            if len(grads_by_trajectory) > 0:
                self.count += 1
                self.optimize(grads_by_trajectory)

                # if self.count % 5 == 0:
                #     self.neighbor.load_state_dict(self.brain.state_dict())
                # for param in self.neighbor.parameters():
                #     param.data = param.data.to(dtype=torch.float32)
                #     param.requires_grad = False

        mean_loss = np.mean(loss_values)
        mean_advantage = np.mean(advantages__)
        # print(f'loss_values is {loss_values}')
        return mean_loss,np.abs(np.mean(entropy_losses)),np.abs(np.mean(approx_kl_divs))

    def optimize(self, grads_by_trajectory):
        self.optimizer.zero_grad()  # Clear previous gradients
        average_grads = []
        for grads_by_layer in zip(*grads_by_trajectory):
            print(f'gradients by layer {grads_by_layer}')
            average_grads.append(torch.mean(torch.stack(grads_by_layer), dim=0))

        assert len(average_grads) == len(list(self.brain.parameters()))
        for average_grad, parameter in zip(average_grads, self.brain.parameters()):
            parameter.grad = average_grad

        self.optimizer.step()
def are_state_dicts_equal(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key not in dict2:
            return False
        # Use torch.equal to check if tensors are equal
        if not torch.equal(dict1[key], dict2[key]):
            return False

    return True