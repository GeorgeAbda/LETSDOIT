import time
import copy
from torch_geometric.data import Batch
from torch_geometric.data import Data
from agora.DAG.algorithm.gnnDeepRL.utils import *

import numpy as np
import torch
from agora.DAG.algorithm.gnnDeepRL.brain import CloudResourceGNN, train_gnn_ac,Actor,Actorv1,Critic,CloudResourceGNNv1
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
node_feature_dim =12 # Adjust based on your DAG node features
resource_feature_dim = 7  # Adjust based on your resource features
hidden_dim = 64
gnn_model = CloudResourceGNN(node_feature_dim, resource_feature_dim, hidden_dim)
hidden_dim_critic=hidden_dim
from torch.optim import Adam

# Assuming BrainTorch is your PyTorch neural network model

class Agent(object):
    def __init__(self, name, gamma, reward_to_go, nn_baseline, normalize_advantages, experience_buffer,model_save_path=None,summary_path=None):
        super().__init__()
        self.global_step = 0  # Track training steps manually
        self.gnn=CloudResourceGNN(node_feature_dim, resource_feature_dim, hidden_dim)
        self.gamma = gamma
        self.name='MonteCarlo'

        self.reward_to_go = reward_to_go
        self.baseline = nn_baseline
        self.normalize_advantages = normalize_advantages
        self.actor = Actor(self.gnn, hidden_dim)  # Policy network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.002)
        self.model_save_path = model_save_path
        self.summary_path = summary_path if summary_path is not None else './tensorboard/%s--%s' % (
            name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.summary_writer = SummaryWriter(self.summary_path)
        for param in self.actor.parameters():
            param.data = param.data.to(dtype=torch.float32)
        self.experience_buffer=experience_buffer


        self.brain = self.actor
        for param in self.brain.parameters():
            param.data = param.data.to(dtype=torch.float32)

        self.optimizer = Adam(self.brain.parameters(), lr=0.002)

    def restore(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))  # Load actor model



    def save(self):
        torch.save(self.actor.state_dict(), self.model_save_path)  # Save actor model

    def _loss(self, X, y, adv):

        # for name, param in self.brain.named_parameters():
        #     print(f" Name {name} , param {param}")
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.shape}")

        logits = self.brain(*X,update=True)
        #logprob = F.log_softmax(logits, dim=1)
        logprob=torch.log(logits+ 1e-8)
        # print(f'logprob shape {logprob.shape}')


        # Ensure y is a LongTensor and within the valid range
        y = torch.clamp(y.long(), 0, logprob.size(1) - 1)

        if logprob.dim() > 2:
            logprob=logprob.reshape(logprob.size(0), -1)

        #logprob_ = logprob.view(-1)
        # print(f'logprob shape {logprob.shape}')

        # Ensure y has the correct shape
        y = y.view(-1)
        # print(f'y shape {y.shape}')

        selected_logprobs = logprob[torch.arange(logprob.size(0)), y]
        # print(f'selected_logprobs shape {selected_logprobs.shape}')
        # print(f'adv shape {adv.shape}')

        # print(selected_logprobs * adv)

        #chosen_pair=logprob_[y]
        return -(selected_logprobs * adv).mean()  # Negative for gradient ascent

    def _loss_list(self, X, y, adv,valids):
        # Get logits as a list of tensors
        logits_list = self.brain(*X, update=True)
        # print(f'len logits {len(logits_list)}')
        # print(f'len(y) {len(y)}')
        # print(f'len(adv) {len(adv)}')

        # Ensure y and adv are lists with the same length as logits_list
        assert len(y) == len(logits_list) == len(adv), "Mismatch in lengths of y, logits, and adv"

        total_loss = 0.0
        l2_reg = 0.0
        for param in self.brain.parameters():
            l2_reg += torch.norm(param, p=2)
        for logits, y_item, adv_item,valid in zip(logits_list, y, adv,valids):
            # Calculate log probabilities
            logits_=logits * valid + (1 - valid) * -1e10
            #logits_ = F.softmax(logits_.view(-1), dim=0).view(logits_.size())
            #logprob = torch.log(logits_ + 1e-8)
            logits_ = F.softmax(logits_, dim=0)

            # Ensure y_item is a LongTensor and within the valid range
            y_item = torch.clamp(y_item.long(), 0, logits_.size(0) - 1)

            # if logits_.dim() > 2:
            #     logits_ = logits_.reshape(logits_.size(0), -1)
            if logits_.dim() == 1:
                logits_ = logits_.unsqueeze(0)  # Reshape to [1, num_classes]

            # Ensure y_item has the correct shape
            y_item = y_item.view(-1)

            # Select log probabilities corresponding to true labels
            #selected_logprobs = logits_[y_item]
            loss = torch.nn.CrossEntropyLoss()
            # print(f'logits_ shape {logits_.shape}')
            # print(f'y_item shape {y_item.shape}')
            # print(f'logits_  {logits_}')
            # print(f'y_item  {y_item}')

            logprob_new = loss(logits_, y_item)  # it is minus logprob but the gradient descent will compensate the minus

            # Compute loss for the current item
            item_loss = (logprob_new * adv_item)
            # Accumulate total loss
            total_loss += item_loss
        lambda_reg = 0.01
        # Return average loss over all items
        return  ((total_loss / len(logits_list) )+ (lambda_reg * l2_reg))




    def update_parameters_offline(self,job_chunk,itr,batch_size):
        print('-------------------------------- We are UPDATINGGGG--------------')
        # for name, param in self.gnn.named_parameters():
        #     if not param.requires_grad:
        #         print(f"HEEEEEEEEEY: {name} does not require gradients")
        #

        loss_values = []
        critic_loss_values = []
        self.actor_optimizer.zero_grad()
        grads_by_trajectory=[]
        cnt=0
        cnt_lg=0
        batching=0
        ehseb=0
        numb_batches=len(self.experience_buffer.buffer)//batch_size
        self.experience_buffer.batch_size=1
        for batch in self.experience_buffer:
            self.optimizer.zero_grad()

            observations, actions, rewards, next_observations, dones, probs, advs,batch_length,valids = batch
            ehseb+=1
            # if ehseb>numb_batches:
            #     break
            batching+=batch_length
            # print(f"batching {batching}")
            loss_value = self._loss_list(observations, actions, advs,valids)
            #print(f"loss value: {loss_value}")
            with torch.autograd.detect_anomaly():

                loss_value.backward()  # Compute gradients using autograd

            grads = [param.grad for param in self.brain.parameters() if param.grad is not None]
            # Get gradients
            # for name, param in self.brain.named_parameters():
            #     if param.grad is None:
            #         print(f"Parameter {name} has None gradient")
            #     else:
            #         print(f"Parameter {name} gradient: {param.grad.mean()}")
            grads_by_trajectory.append(grads)

            loss_values.append(loss_value.item())
            # advantages__.append(advantage)

            #if cnt % 8000 == 0:
            #self.optimize(grads_by_trajectory)
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)

            if batching % batch_size==0:
                self.optimize(grads_by_trajectory)
                # if cnt_lg >20:
                #     self.log(f'{job_chunk}/{itr}/loss', loss_value, self.global_step)
                #     cnt_lg=0
                #self.actor_optimizer.step()
                cnt_lg+=1
                cnt += 1
                self.global_step += 1
                grads_by_trajectory = []


        if len(grads_by_trajectory) > 0:
            print('optimizing from here ')
            self.optimize(grads_by_trajectory)

        mean_loss = np.mean(loss_values)
        # mean_advantage = np.mean(advantages__)
        self.experience_buffer.clear()
        # self.log(f'{job_chunk}/loss', mean_loss, self.global_step)

        return mean_loss




    def update_parameters_batching(self,job_chunk,itr,batch_size):
        print('-------------------------------- We are UPDATINGGGG--------------')
        loss_values = []
        critic_loss_values = []
        #self.actor_optimizer.zero_grad()
        grads_by_trajectory=[]
        cnt=0
        batching=0
        for batch in self.experience_buffer:

            observations, actions, rewards, next_observations, dones, probs, advs,batch_length = batch
            batching+=batch_length
            loss_value = self._loss_list(observations, actions, advs)
            #print(f"loss value: {loss_value}")
            loss_value.backward()  # Compute gradients using autograd

            grads = [param.grad for param in self.brain.parameters()]  # Get gradients
            grads_by_trajectory.append(grads)

            loss_values.append(loss_value.item())
            # advantages__.append(advantage)

            #if cnt % 8000 == 0:
            #self.optimize(grads_by_trajectory)
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=0.5)

            if batching>=batch_size:
                self.optimize(grads_by_trajectory)
                if cnt >=5:
                    self.log(f'{job_chunk}/{itr}/loss', loss_value, self.global_step)
                    cnt=0
                #self.actor_optimizer.step()
                cnt += 1
                self.global_step += 1
                grads_by_trajectory = []


        # if len(grads_by_trajectory) > 0:
        #     self.optimize(grads_by_trajectory)

        mean_loss = np.mean(loss_values)
        # mean_advantage = np.mean(advantages__)
        self.experience_buffer.clear()
        # self.log(f'{job_chunk}/loss', mean_loss, self.global_step)

        return mean_loss

    def update_parameters(self, all_observations, all_actions, all_advantages,itr,job_chunk):
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


        for ep,(observations, actions, advantages) in enumerate(zip(all_observations, all_actions, all_advantages)):
            grads_by_trajectory = []
            cnt = 1
            for observation, action, advantage in zip(observations, actions, advantages):
                if observation is None or action is None:
                    continue
                self.optimizer.zero_grad()
                action=torch.tensor(action)
                # Clear previous gradients
                loss_value = self._loss(observation, action, advantage)
                loss_value.backward()  # Compute gradients using autograd

                grads = [param.grad for param in self.brain.parameters()]  # Get gradients
                grads_by_trajectory.append(grads)

                loss_values.append(loss_value.item())
                advantages__.append(advantage)


                if cnt % 8000 == 0:
                    self.optimize(grads_by_trajectory)
                    self.log(f'{job_chunk}/{itr}/{ep}/loss', np.mean(loss_values), self.global_step)
                    self.global_step += 1
                    grads_by_trajectory = []
                cnt += 1
            if len(grads_by_trajectory) > 0:
                self.optimize(grads_by_trajectory)

        mean_loss = np.mean(loss_values)
        mean_advantage = np.mean(advantages__)

        return mean_loss
        # Log the mean losses



    def log(self, name, loss_value, step):
        self.summary_writer.add_scalar(name, loss_value, step)

    def clip_returns(self, returns, min_value=-10, max_value=10):
        return np.clip(returns, min_value, max_value)

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

        adv_s = []
        for advantages in rewards_n:
            for advantage in advantages:
                adv_s.append(advantage)
        adv_s = np.array(adv_s)
        mean = adv_s.mean()
        std = adv_s.std()
        rewards__n = []
        for advantages in rewards_n:
            advantages__ = []
            for advantage in advantages:
                advantages__.append((advantage - mean) / (std + np.finfo(np.float32).eps))
            rewards__n.append(advantages__)
        q_s = []
        for re in rewards__n:
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
            if len(adv_n)>1:
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
                adv_n = np.array(adv_n[0])
                adv_n = adv_n - adv_n.mean(axis=0)
                return [adv_n]


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



    def normalize_returns(self, returns):
        returns = np.array(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def _sum_of_rewards(self, rewards_n):
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



    def optimize(self, grads_by_trajectory):
        if len(grads_by_trajectory)<2:
            # print('-----------Optimizing--------')
            self.optimizer.step()
        else:

            self.optimizer.zero_grad()  # Clear previous gradients
            print('-----------Optimizing with Average --------')

            average_grads = []
            for grads_by_layer in zip(*grads_by_trajectory):

                average_grads.append(torch.mean(torch.stack(grads_by_layer), dim=0))

            # assert len(average_grads) == len(list(self.brain.parameters()))
            parms=[parm for parm in self.brain.parameters() if parm.grad!=None]
            for average_grad, parameter in zip(average_grads, parms):
                parameter.grad = average_grad

            self.optimizer.step()



