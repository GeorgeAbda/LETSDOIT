import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.path.append('..')
from agora.DAG.algorithm.gnnDeepRL.agent import Agent
from cogito.machine import MachineConfig
from agora.DAG.utils.csv_reader import CSVReader
from agora.auxiliary.tools import average_completion, average_slowdown
from agora.DAG.adapter.episode import Episode
from agora.DAG.algorithm.gnnDeepRL.DRL import RLAlgorithm
from agora.DAG.algorithm.gnnDeepRL.reward_giver import EnergyOptimisationRewardGiverV2
from agora.DAG.utils.feature_functions import *
from agora.DAG.algorithm.gnnDeepRL.utils import ExperienceBuffer
# Import the CloudResourceGNN and train_gnn_ac functions
from agora.DAG.algorithm.gnnDeepRL.brain import CloudResourceGNN, train_gnn_ac

os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(41)

# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 30
jobs_csv = '../jobs_files/batch_task.csv'

reward_giver = EnergyOptimisationRewardGiverV2()

name = f'GNN-{machines_number}'
model_dir = f'./agents/{name}'
summary_path = f'./Tensorboard/{name}'

# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

csv_reader = CSVReader(jobs_csv)
machine_configs = [
                   MachineConfig(240, 1, 1, "ThinkSystem-SR850-V3",1900), MachineConfig(8, 1, 1, "HPProLiant-ML30-Gen11",3200),
                   MachineConfig(80, 1, 1, "FALINUX-AnyStor-700EC-NM",3000),MachineConfig(64, 1, 1, "Dell-PowerEdgeHS5610",2100), MachineConfig(120, 1, 1, "FusionServer-1288H-V7",1900)]
from agora.DAG.algorithm.gnnDeepRL.utils import *

# Initialize GNN model
node_feature_dim = 5  # Adjust based on your DAG node features
resource_feature_dim = 35  # Adjust based on your resource features
hidden_dim = 64
action_dim = len(machine_configs)  # Number of possible actions (machines to allocate)
experience_buffer=ExperienceBuffer(100000)
agent = Agent("gnn",0.95, reward_to_go=True, nn_baseline=True, normalize_advantages=True,experience_buffer=experience_buffer)
jobs_len=5

epsilon = 1.0  # Start with full exploration
epsilon_min = 0.01  # Minimum exploration
epsilon_decay = 0.995  # Epsilon decay rate per step
# Main training loop

def update_parameters(agent):
    # if not agent.experience_buffer.is_full():
    #     return

    print('-------------------------------- We are UPDATINGGGG--------------')
    all_observations, all_actions, all_rewards, all_next_observations, all_dones = agent.experience_buffer.get_all()

    combined_data = list(
        zip(all_observations, all_actions, all_rewards, all_next_observations, all_dones))
    batch_size = 100  # Define the batch size
    loss_values = []
    critic_loss_values = []

    for i in range(0, len(combined_data), batch_size):
        batch_data = combined_data[i:i + batch_size]
        observations_batch = [data[0] for data in batch_data]
        actions_batch = torch.tensor([data[1] for data in batch_data], dtype=torch.long)
        rewards_batch = torch.tensor([data[2] for data in batch_data], dtype=torch.float32)
        #rewards_batch=agent.normalize_returns(rewards_batch)

        next_observations_batch = [data[3] for data in batch_data]
        dones_batch = torch.tensor([data[4] for data in batch_data], dtype=torch.float32)

        # Prepare batched inputs
        # batch_dags, batch_resource_features = prepare_batch(observations_batch)
        # next_batch_dags, next_batch_resource_features = prepare_batch(next_observations_batch)

        # Zero gradients
        agent.actor_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()

        for obs, act, rew, nxt, do in zip(observations_batch, actions_batch, rewards_batch, next_observations_batch, dones_batch):
            # Prepare single element inputs from batched inputs
            single_obs_dag, single_obs_resource_features = prepare_batch([obs],1)
            single_next_obs_dag, single_next_obs_resource_features = prepare_batch([nxt],1)



            #single_obs_resource_features = batch_resource_features[batch_dags.batch == batch_dags.batch[obs]]



            #single_next_obs_resource_features = next_batch_resource_features[next_batch_dags.batch == next_batch_dags.batch[nxt]]

            # Compute current Q-values for the single element
            current_q_values = agent.critic(single_obs_dag, single_obs_resource_features).squeeze()
            # print(f'current_q_values shape {current_q_values.shape}')
            flattened_q_values = current_q_values.view(-1)
            #
            # print('act is {act}'.format(act=act))
            # print('rew is {rew}'.format(rew=rew))

            # Get Q-value for the chosen action
            state_action_value = flattened_q_values[act].squeeze()
            # print(f'state_action_value shape {state_action_value.shape}')

            with torch.no_grad():
                # Calculate the next state value
                next_state_value = agent.target_critic(single_next_obs_dag, single_next_obs_resource_features).max()
                # print(f'next_state_value shape {next_state_value.shape}')

                # Ensure reward and done flag are tensors with the right shape
                rew = torch.tensor(rew).float().to(next_state_value.device)
                do = torch.tensor(do).float().to(next_state_value.device)

                # Calculate target Q-value
                target_q_value = rew + agent.gamma * next_state_value * (1 - do)

            # Calculate advantage
            advantages = target_q_value - state_action_value
            # print(f'target_q_value shape {target_q_value.shape}')
            # print(f'target_q_value  {target_q_value}')
            #
            # print(f'advantages shape {advantages.shape}')
            # print(f'advantages  {advantages}')


            # Compute critic loss and backpropagate
            critic_loss = F.smooth_l1_loss(state_action_value.unsqueeze(0), target_q_value.unsqueeze(0))
            critic_loss.backward()

            # Update actor
            logits = agent.actor(single_obs_dag,single_obs_resource_features)
            log_probs = F.log_softmax(logits, dim=1).view(-1)
            # print(f'log_probs shape {log_probs.shape}')

            selected_log_prob = log_probs[act]
            actor_loss = -(selected_log_prob * advantages.detach())
            actor_loss.backward()

            # Append losses
            loss_values.append(actor_loss.item())
            critic_loss_values.append(critic_loss.item())

        # Gradient clipping and parameter update
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
        agent.critic_optimizer.step()
        agent.actor_optimizer.step()

    # Clear buffer after processing all data
    agent.experience_buffer.clear()
    agent.global_step += 1

    # Update target network
    agent.update_target_network()

    # Log the mean losses
    agent.log('loss', np.mean(loss_values), agent.global_step)
    agent.log('loss_critic', np.mean(critic_loss_values), agent.global_step)


for job_chunk in range(0,15):
    jobs_configs = csv_reader.generate(jobs_len * job_chunk, jobs_len)
    algorithm = RLAlgorithm(agent, reward_giver, features_normalize_func=features_normalize_func,
                            features_extract_func=features_extract_func, epsilon=epsilon, epsilon_min=epsilon_min,
                            epsilon_decay=epsilon_decay)

    for itr in range(n_iter):
        print(f"********** Iteration {itr} ************")
        trajectories = []
        makespans = []
        average_completions = []
        average_slowdowns = []

        tic = time.time()
        # Collect trajectories
        for e in range(10):
            print(f"********** Epsiode {e} ************")

            episode = Episode(machine_configs, jobs_configs, algorithm, None)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()
            trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
            makespans.append(episode.simulation.env.now)
            average_completions.append(average_completion(episode))
            average_slowdowns.append(average_slowdown(episode))

        # Extract states, actions, returns, and advantages from trajectories
        all_observations = []
        all_actions = []
        all_rewards = []
        all_next_observations = []

        for trajectory in trajectories:
            observations = []
            actions = []
            rewards = []
            next_observations = []

            for i, node in enumerate(trajectory):
                observations.append(node.observation)
                actions.append(node.action)
                rewards.append(node.reward)

                # Add next observation
                if i < len(trajectory) - 1:
                    next_observations.append(trajectory[i + 1].observation)
                else:
                    # For the last step, use the same observation as next_observation
                    # or a terminal state representation if available
                    next_observations.append(node.observation)

            all_observations.append(observations)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_observations.append(next_observations)

        #all_q_s, all_advantages = agent.estimate_return(all_rewards,all_observations,all_next_observations)
        # Assuming all_observations, all_actions, all_rewards, and all_advantages are lists of lists
        for observations, actions, rewards, next_observations in zip(all_observations, all_actions,
                                                                                 all_rewards,
                                                                                 all_next_observations):
            for i, (obs, act, rew, next_obs) in enumerate(
                    zip(observations, actions, rewards, next_observations)):
                done = 1 if i == len(observations) - 1 else 0  # Assume last step in trajectory is done
                experience = (obs, act, rew, next_obs, done)
                agent.experience_buffer.add(experience)
        #agent.update_parameters(all_observations, all_actions, all_rewards, all_advantages)
        update_parameters(agent)
        toc = time.time()
        print(f"Iteration {itr}, Time: {(toc - tic) / 12:.2f}")
        print(f"Average Makespan: {np.mean(makespans):.2f}, "
              f"Avg Completion: {np.mean(average_completions):.2f}, "
              f"Avg Slowdown: {np.mean(average_slowdowns):.2f}")

        # Evaluate current policy
        # eval_makespans = []
        # for _ in range(5):
        #     eval_algorithm = RLAlgorithm(agent, reward_giver, features_normalize_func=features_normalize_func,
        #                                  features_extract_func=features_extract_func)
        #     eval_episode = Episode(machine_configs, jobs_configs, eval_algorithm, None)
        #     eval_episode.run()
        #     eval_makespans.append(eval_episode.simulation.env.now)
        #
        # print(f"Evaluation Average Makespan: {np.mean(eval_makespans):.2f}")

        # Save the model periodically
        if (itr + 1) % 10 == 0:
            torch.save(agent.actor.state_dict(), f'{model_dir}/actor_model_iter_{itr + 1}.pth')
            torch.save(agent.critic.state_dict(), f'{model_dir}/critic_model_iter_{itr + 1}.pth')

# Save the final trained model
torch.save(agent.actor.state_dict(), f'{model_dir}/actor_model_final.pth')
torch.save(agent.critic.state_dict(), f'{model_dir}/critic_model_final.pth')