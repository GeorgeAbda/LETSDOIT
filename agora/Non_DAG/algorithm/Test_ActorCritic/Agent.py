import copy
import numpy as np
import torch as torch
from torch import nn
from torch.optim import Adam
from agora.Non_DAG.algorithm.Test_ActorCritic.Networks import FeedForwardNN
from agora.Non_DAG.algorithm.Test_ActorCritic.Networks import CriticNN
from torch.optim.lr_scheduler import ExponentialLR


class Agent(object):
    def __init__(self, gamma, reward_to_go, nn_baseline, normalize_advantages, model_save_path=None
                 ):
        super().__init__()
        self.normalize_returns=True
        self.gamma = gamma
        self.clip = 0.2
        self.reward_to_go = reward_to_go
        self.baseline = nn_baseline
        self.normalize_advantages = normalize_advantages
        self.batch_size=500
        self.brain = FeedForwardNN(9)
        self.critic = CriticNN(9)
        for param in self.brain.parameters():
            param.data = param.data.to(dtype=torch.float32)

        self.optimizer = Adam(self.brain.parameters(), lr=0.001)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=0.001)
        self.model_save_path = model_save_path

    def _sum_of_rewards(self, rewards_n,indices):
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


        if self.normalize_returns:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_s = []
            for advantages in rewards_n:
                for advantage in advantages:
                    adv_s.append(advantage)
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
            adv_n__ = []
            for advantages in rewards_n:
                advantages__ = []
                for advantage in advantages:
                    advantages__.append((advantage - mean) / (std + np.finfo(np.float32).eps))
                adv_n__.append(advantages__)
            rewards_n = adv_n__

        q_s = []
        for re,ind in zip(rewards_n,indices):

            q = []
            cur_q = 0
            for reward in reversed(re):
                cur_q = cur_q * self.gamma + reward
                q.append(cur_q)
            q = list(reversed(q))
            q_=[q[i] for i in ind]
            q_s.append(q_)
        if self.reward_to_go:
            return q_s
        else:
            q_n = []
            for q in q_s:
                q_n.append([q[0]] * len(q))
            return q_n
    def normalize_returns(self, returns):
        returns = np.array(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)
    def _compute_advantage(self, q_n, all_observations,indices):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            arguments:
                q_n: shape: (...).

            returns:
                adv_n: shape: (...).
        """
        # Computing Baselines
        all_V = []

        V = False
        for observations,ind in zip(all_observations,indices):
            observations=[observations[i] for i in ind]
            i = 0
            V_obs = []
            for observation in observations:
                if observation is not None:
                    i += 1
                    V = torch.argmax(self.critic(observation).squeeze())
                    V_obs.append(V)
                else:
                    if i == 0:
                        V_obs.append(0)
                        i += 1
                        continue
                    # V_obs.append(V_obs[-1])
                    V_obs.append(0)
            # V_obs = torch.Tensor(V_obs)
            all_V.append(V_obs)

        max_length = max([len(V) for V in all_V])
        for idx, V in enumerate(all_V):
            while len(V) < max_length:
                V.append(0)
            all_V[idx] = torch.tensor(V)

        max_length_q = max([len(q) for q in q_n])
        for idx, q in enumerate(q_n):
            while len(q) < max_length_q:
                q.append(0)
            q_n[idx] = torch.tensor(q)
        q_n, all_V = torch.stack(q_n), torch.stack(all_V)
        adv_n = torch.subtract(q_n, all_V)

        return adv_n
        # if self.baseline:
        #     adv_n = copy.deepcopy(q_n)
        #     max_length = max([len(adv) for adv in adv_n])
        #     for adv in adv_n:
        #         while len(adv) < max_length:
        #             adv.append(0)
        #     adv_n = np.array(adv_n)
        #     adv_n = adv_n - adv_n.mean(axis=0)
        #
        #     adv_n__ = []
        #     for i in range(adv_n.shape[0]):
        #         original_length = len(q_n[i])
        #         adv_n__.append(list(adv_n[i][:original_length]))
        #     return adv_n__
        # else:
        #     adv_n = q_n.copy()
        #     return adv_n

    def estimate_return(self, rewards_n, all_observations,real_actions):
        """
            Estimates the returns over a set of trajectories.


            arguments:
                re_n: shape(...).

            returns:
                q_n: shape: (...).
                adv_n: shape: (...).
        """
        q_n = self._sum_of_rewards(rewards_n,real_actions)
        adv_n = self._compute_advantage(q_n, all_observations,real_actions)

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

    def _loss(self, X, y,mask):
        logits = self.brain(X)
        # print(f'logits shape : {logits.shape}')
        # if logits.shape != torch.Size([]):
        #     logits = torch.reshape(logits, [1, logits.shape[0]])
        #     logits = logits.to(dtype=torch.float32)
        # else :
        #     logits = torch.reshape(logits, [1])
        #     logits = torch .reshape(logits, [1, logits.shape[0]])
        if len(logits) == 1:
            logits.unsqueeze(0)

        else:
            logits = logits.squeeze().unsqueeze(0)
        logits_ = logits * mask + (1 - mask) * -1e10

        logits = logits_.to(dtype=torch.float32)

        y = torch.tensor(y)
        y = y.long()

        loss = torch.nn.CrossEntropyLoss()
        logprob = -loss(logits, y)
        return logprob

    def update_parameters(self, all_observations, all_actions, all_advantages, all_log_probs, all_q_s,all_masks):
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
        critic_losses = []
        advantages__ = []
        nb_trajectories = len(all_observations)

        for observations, actions, advantages, log_probs, q_s ,masks in zip(all_observations, all_actions, all_advantages,
                                                                     all_log_probs, all_q_s,all_masks):
            V_obs = []
            grads_by_trajectory = []
            grads_by_trajectory_critic = []

            cnt = 1
            curr_log_probs = []
            critic_obs_loss = []
            for observation, action, advantage, log_prob, q,mask in zip(observations, actions, advantages, log_probs, q_s,masks):
                if observation is None or action is None or log_prob is None:
                    continue
                self.brain.zero_grad()
                # print(f"Raw reward: {q}")
                # print(f"Reward stats - Mean: {np.mean(q_s)}, Std: {np.std(q_s)}")
                curr_log_prob = self._loss(observation, [action],mask)
                curr_log_probs.append(curr_log_prob)

                # ratios = torch.exp(torch.subtract(curr_log_prob, log_prob))
                # # advantages = torch.stack(advantage)
                # surr1 = torch.mul(ratios, advantage)
                # surr2 = torch.mul(torch.clamp(ratios, 1 - self.clip, 1 + self.clip), advantage)
                # loss_value = (-torch.min(surr1, surr2))
                # print(f'loss_value {loss_value}')

                # loss_value.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
                #
                # grads = [param.grad for param in self.brain.parameters()]  # Get gradients
                #
                # grads_by_trajectory.append(grads)

                # print(f'loss value {loss_value}')
                # loss_values.append(loss_value.item())
                # advantages__.append(advantage)




                # Compute V
                with torch.set_grad_enabled(True):
                    #V = torch.argmax(self.critic(observation).squeeze()).to(dtype=torch.float32)
                    V = self.critic(observation).squeeze().to(dtype=torch.float32)
                    V = V.detach().requires_grad_()
                    # Ensure q requires gradients if it doesn't already
                    if not q.requires_grad:
                        q = torch.tensor(q, dtype=torch.float32)  # Ensure q is float32

                        q = q.detach().requires_grad_()

                    # Compute critic loss

                    # Use Huber loss for more robust training
                    critic_loss = nn.SmoothL1Loss()(V, q)

                    # critic_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                    critic_obs_loss.append(critic_loss)
                    grads_critic = [param.grad for param in self.critic.parameters()]  # Get gradients

                    grads_by_trajectory_critic.append(grads_critic)
                    critic_losses.append(critic_loss)

                    # mean_critic_loss = torch.mean(torch.stack(critic_losses))
                    # mean_loss = torch.mean(torch.stack(loss_values))
                    #
                    # # self.count += 1
                    # # self.optimize_actor(grads_by_trajectory)
                    # # self.optimize_critic(grads_by_trajectory_critic)
                    # self.critic.zero_grad()
                    # mean_critic_loss.backward()
                    # self.optimizer_critic.step()
                    #
                    # self.brain.zero_grad()
                    # mean_loss.backward()
                    # self.optimizer.step()
                # Optionally, you can check if the loss requires gradients
                # print(f"critic_loss requires_grad: {critic_loss.requires_grad}")
            curr_log_probs, log_probs = torch.tensor(curr_log_probs, requires_grad=True), torch.tensor(log_probs, requires_grad=True)
            l = []
            for log in log_probs:
                if log is not None:
                    l.append(log)
            # print(f"len of l is {len(l)}")
            log_probs = torch.stack(l).squeeze()
            # print(log_probs.shape)
            curr_log_probs = torch.tensor(curr_log_probs, requires_grad=True)
            # print(curr_log_probs.shape)
            advantages = torch.stack(advantages)
            ratios = torch.exp(torch.subtract(curr_log_probs, log_probs))
            # print(ratios.shape, advantages.shape)
            surr1 = torch.mul(ratios, advantages)
            surr2 = torch.mul(torch.clamp(ratios, 1 - self.clip, 1 + self.clip), advantages)

            loss_value = (-torch.min(surr1, surr2)).mean()
            loss_value.backward(retain_graph=True)
            self.optimizer.step()
            loss_values.append(loss_value)

            critic_loss = torch.mean(torch.stack(critic_obs_loss))
            self.critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.optimizer_critic.step()
            critic_losses.append(critic_loss)

        return loss_values, critic_losses


    def optimize_actor(self, grads_by_trajectory):
        self.optimizer.zero_grad()  # Clear previous gradients
        average_grads = []
        for grads_by_layer in zip(*grads_by_trajectory):
            # print(f'grads_by_layer {grads_by_layer}')
            average_grads.append(torch.mean(torch.stack(grads_by_layer), dim=0))

        assert len(average_grads) == len(list(self.brain.parameters()))
        for average_grad, parameter in zip(average_grads, self.brain.parameters()):
            parameter.grad = average_grad

        self.optimizer.step()



    def optimize_critic(self, grads_by_trajectory):
        self.optimizer_critic.zero_grad()  # Clear previous gradients
        average_grads = []
        for grads_by_layer in zip(*grads_by_trajectory):
            average_grads.append(torch.mean(torch.stack(grads_by_layer), dim=0))

        assert len(average_grads) == len(list(self.brain.parameters()))
        for average_grad, parameter in zip(average_grads, self.brain.parameters()):
            parameter.grad = average_grad

        self.optimizer_critic.step()