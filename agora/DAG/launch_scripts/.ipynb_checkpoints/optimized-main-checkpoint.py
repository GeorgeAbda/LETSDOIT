import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from agora.DAG.algorithm.gnnDeepRL.utils import *
torch.autograd.set_detect_anomaly(True)
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
from agora.DAG.algorithm.gnnDeepRL.brain import CloudResourceGNN, train_gnn_ac

os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(41)

# Parameters
machines_number = 5
jobs_len = 5
n_iter = 30
jobs_csv = '../jobs_files/batch_task.csv'
reward_giver = EnergyOptimisationRewardGiverV2()
name = f'GNN-{machines_number}'
model_dir = f'./agents/{name}'
summary_path = f'./Tensorboard/{name}'

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

csv_reader = CSVReader(jobs_csv)
machine_configs = [
    MachineConfig(240, 1, 1, "ThinkSystem-SR850-V3", 1900),
    MachineConfig(8, 1, 1, "HPProLiant-ML30-Gen11", 3200),
    MachineConfig(80, 1, 1, "FALINUX-AnyStor-700EC-NM", 3000),
    MachineConfig(64, 1, 1, "Dell-PowerEdgeHS5610", 2100),
    MachineConfig(120, 1, 1, "FusionServer-1288H-V7", 1900)
]

# Initialize agent with experience buffer
node_feature_dim = 5
resource_feature_dim = 35
hidden_dim = 64
action_dim = len(machine_configs)
experience_buffer = ExperienceBuffer(1000)  # Increased buffer size for more experiences
agent = Agent("gnn", 0.95, reward_to_go=True, nn_baseline=True,
              normalize_advantages=True, experience_buffer=experience_buffer)

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


def prepare_for_loss_calculation(current_q_values, actions_batch, batch_dags):
    # Get the batch assignment for each node
    batch_assignment = batch_dags.batch

    # Get the number of graphs (jobs) in the batch
    num_graphs = batch_assignment.max().item() + 1

    # Aggregate Q-values for each graph (job) using mean
    aggregated_q_values = torch.zeros(num_graphs, current_q_values.size(1), device=current_q_values.device)
    #print(f"Aggregated Q-values shape {aggregated_q_values.shape}")
    for i in range(num_graphs):
        mask = (batch_assignment == i)
        aggregated_q_values[i] = current_q_values[mask].mean(dim=0)

    # Now aggregated_q_values should have shape [num_graphs, num_actions]
    # which should match the shape of actions_batch

    # Ensure actions_batch is a LongTensor and has the right shape
    actions_batch = actions_batch.long().view(-1)
    print(f"actions_batch shape {actions_batch.shape}")

    # Select the Q-values for the taken actions
    action_q_values = aggregated_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

    return action_q_values, actions_batch
def update_agent(agent):
    if not agent.experience_buffer.is_full():
        return 0, 0

    all_observations, all_actions, all_rewards, all_next_observations, all_dones = agent.experience_buffer.get_all()
    combined_data = list(zip(all_observations, all_actions, all_rewards, all_next_observations, all_dones))

    batch_size = 100  # Define the batch size
    loss_values = []
    critic_loss_values = []

    agent.actor_optimizer.zero_grad()
    agent.critic_optimizer.zero_grad()

    for i in range(0, len(combined_data), batch_size):
        batch_data = combined_data[i:i + batch_size]
        observations_batch = [data[0] for data in batch_data]
        actions_batch = torch.tensor([data[1] for data in batch_data], dtype=torch.long)
        rewards_batch = torch.tensor([data[2] for data in batch_data], dtype=torch.float32)
        next_observations_batch = [data[3] for data in batch_data]
        dones_batch = torch.tensor([data[4] for data in batch_data], dtype=torch.float32)

        # Prepare batched inputs using the new prepare_batch function
        batch, resources_features = prepare_batch(observations_batch, batch_size)
        next_batch, next_resources_features = prepare_batch(next_observations_batch, batch_size)

        # Use batch.x, batch.edge_index, and batch.resource_features in your model
        current_q_values = agent.critic(batch,resources_features).squeeze()
        print(f'current_q_values shape = {current_q_values.shape}')
        print(f'actions_batch shape = {actions_batch.shape}')
        action_q_values, actions = prepare_for_loss_calculation(current_q_values, actions_batch, batch)
        print(f'new current_q_values shape = {action_q_values.shape}')
        print(f'new actions_batch shape = {actions.shape}')
        state_action_values = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_state_values = agent.target_critic(next_batch,next_resources_features).max(1)[0]
            target_q_values = rewards_batch + agent.gamma * next_state_values * (1 - dones_batch)

        # Compute advantages
        advantages = target_q_values - state_action_values

        # Update critic
        critic_loss = F.smooth_l1_loss(state_action_values, target_q_values.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Update actor
        logits = agent.actor(batch,resources_features)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[torch.arange(log_probs.size(0)), actions_batch]
        actor_loss = -(selected_log_probs * advantages.detach()).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

        loss_values.append(actor_loss.item())
        critic_loss_values.append(critic_loss.item())

    # Update target network
    agent.update_target_network()

    # Clear buffer after processing all data
    agent.experience_buffer.clear()

    return np.mean(loss_values), np.mean(critic_loss_values)

def collect_trajectory(episode, agent):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i, node in enumerate(episode.simulation.scheduler.algorithm.current_trajectory):
        obs = node.observation
        action = node.action
        reward = node.reward

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        if i < len(episode.simulation.scheduler.algorithm.current_trajectory) - 1:
            next_obs = episode.simulation.scheduler.algorithm.current_trajectory[i + 1].observation
            done = False
        else:
            next_obs = obs  # or a terminal state representation if available
            done = True

        next_observations.append(next_obs)
        dones.append(done)

        # Add experience to buffer immediately for each step in the trajectory
        agent.experience_buffer.add((obs, action, reward, next_obs, done))

    return observations, actions, rewards, next_observations, dones

#
# def update_agent(agent):
#     if not agent.experience_buffer.is_full():
#         return 0, 0
#
#     all_observations, all_actions, all_rewards, all_next_observations, all_dones = agent.experience_buffer.get_all()
#
#     combined_data = list(zip(all_observations, all_actions, all_rewards, all_next_observations, all_dones))
#     batch_size = 100  # Define the batch size
#     loss_values = []
#     critic_loss_values = []
#     agent.actor_optimizer.zero_grad()
#     agent.critic_optimizer.zero_grad()
#
#     for i in range(0, len(combined_data), batch_size):
#         batch_data = combined_data[i:i + batch_size]
#         observations_batch = [data[0] for data in batch_data]
#         actions_batch = torch.tensor([data[1] for data in batch_data], dtype=torch.long)
#         rewards_batch = torch.tensor([data[2] for data in batch_data], dtype=torch.float32)
#         next_observations_batch = [data[3] for data in batch_data]
#         dones_batch = torch.tensor([data[4] for data in batch_data], dtype=torch.float32)
#         print('len(observations_batch)',len(observations_batch))
#         # Prepare batched inputs
#         batch_dags, batch_resource_features ,_= prepare_batch(observations_batch)
#         next_batch_dags, next_batch_resource_features,_ = prepare_batch(next_observations_batch)
#
#         # Ensure these tensors are of floating-point type
#         #batch_dags = batch_dags.to(dtype=torch.float32)
#         batch_resource_features = batch_resource_features.to(dtype=torch.float32)
#         print("batch_dags",batch_dags)
#         print('batch_resource_features shape ', batch_resource_features.shape)
#         # Compute current state values
#         current_q_values = agent.critic(batch_dags, batch_resource_features).squeeze()
#         state_action_values = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()
#         print('next_observations_batch shape ', len(next_batch_dags))
#         print('next_observations_batch shape ', len(next_batch_resource_features))
#         print('current_q_values  shape ', current_q_values.shape)
#
#         with torch.no_grad():
#             print('state_action_values state shape ', state_action_values.shape)
#             next_state_values = agent.target_critic(next_batch_dags, next_batch_resource_features).max(1)[0]
#             print('next_state_values shape ', next_state_values.shape)
#             print('rewards_batch  shape ', rewards_batch.shape)
#             print('current_q_values  shape ', current_q_values.shape)
#
#             target_q_values = rewards_batch + agent.gamma * next_state_values * (1 - dones_batch)
#
#         # Compute advantages
#         advantages = target_q_values - state_action_values
#
#
#
#         # Update critic
#         critic_loss = F.smooth_l1_loss(state_action_values, target_q_values.detach())
#         critic_loss.backward()
#         agent.critic_optimizer.step()
#
#         # Update actor
#         logits = agent.actor(batch_dags, batch_resource_features)
#         log_probs = F.log_softmax(logits, dim=1)
#         selected_log_probs = log_probs[torch.arange(log_probs.size(0)), actions_batch]
#         actor_loss = -(selected_log_probs * advantages.detach()).mean()
#         actor_loss.backward()
#         agent.actor_optimizer.step()
#
#     # Update target network
#     agent.update_target_network()
#
#     # Clear buffer after processing all data
#     agent.experience_buffer.clear()
#
#     return actor_loss.item(), critic_loss.item()


# Main training loop
for job_chunk in range(0, 15):
    jobs_configs = csv_reader.generate(jobs_len * job_chunk, jobs_len)
    algorithm = RLAlgorithm(agent,
                            reward_giver,
                            features_normalize_func=features_normalize_func,
                            features_extract_func=features_extract_func,
                            epsilon=epsilon,
                            epsilon_min=epsilon_min,
                            epsilon_decay=epsilon_decay)

    for itr in range(n_iter):
        print(f"********** Iteration {itr} ************")
        trajectories = []
        makespans = []
        average_completions = []
        average_slowdowns = []

        tic =time.time()
        # Collect trajectories and add experiences to buffer immediately within collect_trajectory function.
        for e in range(1):
            print(f"********** Episode {e} ************")
            episode =Episode(machine_configs,
                    jobs_configs,
                    algorithm,
                    None)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()
            trajectory_data =collect_trajectory(episode,
                               agent)  # Pass agent to add experiences directly to buffer.
            trajectories.append(trajectory_data)

            makespans.append(episode.simulation.env.now)
            average_completions.append(average_completion(episode))
            average_slowdowns.append(average_slowdown(episode))

        print('we must do update')
        # Update agent using experiences from the buffer.
        actor_loss_value,critic_loss_value =update_agent(agent)

        toc =time.time()
        print(f"Iteration {itr}, Time: {(toc - tic) / 12:.2f}")
        print(f"Average Makespan: {np.mean(makespans):.2f}, "
              f"Avg Completion: {np.mean(average_completions):.2f}, "
              f"Avg Slowdown: {np.mean(average_slowdowns):.2f}")
        print(f"Actor Loss: {actor_loss_value:.4f}, Critic Loss: {critic_loss_value:.4f}")

        # Save the model periodically.
        if (itr +
            1) % 10 == 0:
            torch.save(agent.actor.state_dict(),
                       f'{model_dir}/actor_model_iter_{itr + 1}.pth')
            torch.save(agent.critic.state_dict(),
                       f'{model_dir}/critic_model_iter_{itr + 1}.pth')

# Save the final trained model at the end of training loop.
torch.save(agent.actor.state_dict(),
           f'{model_dir}/actor_model_final.pth')
torch.save(agent.critic.state_dict(),
           f'{model_dir}/critic_model_final.pth')


