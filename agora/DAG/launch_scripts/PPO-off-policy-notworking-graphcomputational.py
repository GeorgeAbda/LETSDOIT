import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import wandb
wandb.login(key='7dea3f024f3b2592c4d66656caf1c039db736d53')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.path.append('..')
from agora.DAG.algorithm.gnnDeepRL.utils import compute_gradient_norms
from agora.DAG.algorithm.gnnDeepRL.Agent_noembed import Agent
from cogito.machine import MachineConfig
from agora.DAG.utils.csv_reader import CSVReader
from agora.auxiliary.tools import average_completion, average_slowdown
from agora.DAG.adapter.episode import Episode,EpisodeTest
from agora.DAG.algorithm.gnnDeepRL.DRL_AccumulatedReward import RLAlgorithm
from agora.DAG.algorithm.gnnDeepRL.reward_giver import EnergyBaseline,CombinedReward,MakespanRewardGiver,Energy,EnergyOptimisationTask, EnergyOptimisationRewardGiverV2,EnergyOptimisationRewardGiver
from agora.DAG.utils.feature_functions import *
from agora.DAG.algorithm.gnnDeepRL.utils import ExperienceBuffer
from agora.DAG.algorithm.gnnDeepRL.viz import plot_resource_utilization,visualize_features
from agora.Non_DAG.utils.tools import multiprocessing_run, average_completion, average_slowdown,transform_to_first_digit

# Import the CloudResourceGNN and train_gnn_ac functions
from agora.DAG.algorithm.gnnDeepRL.brain import CloudResourceGNNv1, train_gnn_ac

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.random.seed(41)

# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 5
n_iter = 1
jobs_csv = '../jobs_files/batch_task.csv'
# energy_giver = Energy()
# makespan_giver = MakespanRewardGiver(reward_per_timestamp=1.0)
# reward_giver = CombinedReward(energy_giver, makespan_giver)

name = f'GNN-REWARD-CUMMULATVE{machines_number}'
model_dir = f'./agents/{name}'
summary_path = f'./Tensorboard/{name}'
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=summary_path)

# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

csv_reader = CSVReader(jobs_csv)

import os
import time
from math import gamma

import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

import wandb

wandb.login(key='7dea3f024f3b2592c4d66656caf1c039db736d53')

sys.path.append('..')
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from cogito.machine import MachineConfig
from agora.Non_DAG.algorithm_energy.random_algorithm import RandomAlgorithm
from agora.Non_DAG.algorithm_energy.tetris import Tetris
from agora.Non_DAG.algorithm_energy.first_fit import FirstFitAlgorithm
# from agora.Non_DAG.algorithm.PPO_Energy.PPO_classic import RLAlgorithm
# from agora.Non_DAG.algorithm.PPO_Energy.Agent import Agent
# from agora.Non_DAG.algorithm.PPO.Networks import brain

from agora.Non_DAG.algorithm.DeepJS.reward_giver import AverageCompletionRewardGiver, AverageSlowDownRewardGiver, \
    MakespanRewardGiver
# from agora.Non_DAG.algorithm.PPO.reward_giver import EnergyOptimisationRewardGiver

from agora.Non_DAG.utils.csv_reader import CSVReader
from agora.Non_DAG.utils.feature_functions import features_extract_func_ac, features_normalize_func_ac
from agora.Non_DAG.utils.tools import multiprocessing_run, average_completion, average_slowdown
# from agora.Non_DAG.utils.episode import Episode
from agora.DAG.algorithm.gnnDeepRL.reward_giver import EnergyBaseline,CombinedReward,MakespanRewardGiver,Energy,EnergyOptimisationTask, EnergyOptimisationRewardGiverV2,EnergyOptimisationRewardGiver

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
np.random.seed(1)
# tf.random.set_random_seed(41)
# ************************ Parameters Setting Start *************w***********
machines_number = 5
jobs_len =1
n_iter = 150
n_episode = 12
jobs_csv = '../jobs_files/jobs.csv'

# brain = brain(9)

energy_giver = Energy()
makespan_giver = MakespanRewardGiver(reward_per_timestamp=1.0)
#reward_giver = CombinedReward(energy_giver, makespan_giver)
reward_giver =energy_giver

# reward_giver = EnergyOptimisationRewardGiver()
features_extract_func = features_extract_func_ac
features_normalize_func = features_normalize_func_ac

# name = '%s-%s-m%d' % (reward_giver.name, brain.name, machines_number)
# model_dir = './agents/%s' % name
# ************************ Parameters Setting End ************************

# if not os.path.isdir(model_dir):
#    os.makedirs(model_dir)

"**********************************************************"


def calculate_mean(list_of_lists):
    total_sum = 0
    total_count = 0

    for sublist in list_of_lists:
        total_sum += sum(sublist)
        total_count += len(sublist)

    m = total_sum / total_count
    return m


"*************************************************************"
gamma=0.99
epsilon=0.2
entropy_coefficient=0.2
lr=0.001
batch_size=3000
machine_configs = [
    MachineConfig(0, 240, 1, 1, "ThinkSystem-SR850-V3", 1900),
    MachineConfig(1, 200, 2, 1, "HPProLiant-ML30-Gen11", 3200),
    MachineConfig(2, 512, 2, 1, "FALINUX-AnyStor-700EC-NM", 3000),
    MachineConfig(4, 120, 1, 1, "FusionServer-1288H-V7", 1900),
    MachineConfig(3, 512, 5, 1, "Dell-PowerEdgeHS5610", 2100),
    # Add other entries as required
] * 1 # Repeat the list multiple times as shown in your example

# Update the first parameter of each MachineConfig iteratively
for i, config in enumerate(machine_configs):
    config.param1 = i  # Assuming param1 is the first attribute in the MachineConfig class

epsilon = 0.99 # Start with full exploration
epsilon_min = 0.01  # Minimum exploration
epsilon_decay = 0.99  # Epsilon decay rate per step
from agora.DAG.algorithm.gnnDeepRL.utils import *

# Initialize GNN model
node_feature_dim = 5  # Adjust based on your DAG node features
resource_feature_dim = 35  # Adjust based on your resource features
hidden_dim = 128
action_dim = len(machine_configs)  # Number of possible actions (machines to allocate)

batch_size=500
experience_buffer=ExperienceBuffer(100000,batch_size=batch_size)
agent = Agent(0.99,0.95, reward_to_go=True, nn_baseline=True, normalize_advantages=True,experience_buffer=experience_buffer,entropy_coefficient=entropy_coefficient)
jobs_len=1

epsilon = 0.99 # Start with full exploration
epsilon_min = 0.01  # Minimum exploration
epsilon_decay = 0.99  # Epsilon decay rate per step
# Main training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_iter=10000
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Device name:", torch.cuda.get_device_name(0))

for job_chunk in range(2,3):
    name = "Test PolicyGradient Reward Once Energy Episodic EnergyBaseline " + str(job_chunk) + " /20"
    run=wandb.init(project="TaskScheduling - DRL", entity="anashatay99", name=name)
    jobs_configs = csv_reader.generate(job_chunk*jobs_len, jobs_len)

    loss = []
    mean_loss = []
    for itr in range(n_iter):
        print(f"********** Iteration {itr} ************")
        trajectories = []
        makespans = []
        average_completions = []
        average_slowdowns = []
        loss_values = []  # List to store losses for this iteration
        episode_rewards = []  # List to store rewards for this iteration

        tic = time.time()
        average_energies=[]
        # Collect trajectories
        for e in range(1):
            all_observations = []
            all_actions = []
            all_rewards = []
            all_next_observations = []
            all_probs = []
            all_valid = []
            trajectories = []


            print(f"********** Epsiode {e} ************")
            agent.experience_buffer.clear()
            algorithm = RLAlgorithm(epsilon,agent, reward_giver, features_normalize_func=features_normalize_func_ac,
                                    features_extract_func=features_extract_func_ac,wandb=run)
            episode = Episode(machine_configs, jobs_configs, algorithm, None,e)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()
            trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
            makespans.append(episode.simulation.env.now)
            average_completions.append(average_completion(episode))
            average_slowdowns.append(average_slowdown(episode))
            rewards = [node.reward for node in episode.simulation.scheduler.algorithm.current_trajectory]
            total_reward = sum(rewards)
            total_energy = algorithm.reward_giver.get_total_energy()
            average_energies.append(-total_reward / 3600000 * episode.simulation.env.now)

            # Log total energy per episode along with previous metrics
            wandb.log({
                "Total Energy Episodes": -total_reward / 3600000 * episode.simulation.env.now
            })

            # Log mean energy for the iteration and additional metrics
        wandb.log({
            "Mean Total Energy Iterations": np.mean(average_energies)
        })

        toc = time.time()
        print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))

        # Prepare for parameter update
        all_observations, all_actions, all_rewards, all_masks = [], [], [], []

        # Initialize tracking lists outside the main loop
        policy_entropies = []
        action_probs_over_time = []

        for trajectory in trajectories:
            observations, actions, rewards, masks, no_nones = [], [], [], [], []
            for node in trajectory:
                observations.append(node.embed)
                actions.append(node.action)
                if node.rlreward != None:
                    # print('yess')
                    rewards.append(node.rlreward)
                masks.append(node.valid_candid)
                if node.observation != None:
                    # print('yess')
                    no_nones.append(node.observation)
                # Calculate entropy using logits
                if node.prob != None:
                    # Convert logits to probabilities using softmax
                    logits = node.prob.detach().numpy()
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    probs_ = probs.tolist()
                    # Store action probabilities
                    action_probs_over_time.append(probs_)

                    # Calculate entropy
                    policy_entropy = -np.sum(probs * np.log(probs + 1e-10))
                    policy_entropies.append(policy_entropy)

                    wandb.log({
                        "policy_entropy": policy_entropy,
                    })
            print(f'len of no_nones observations : {len(no_nones)}')
            agent.batch_size = transform_to_first_digit(len(no_nones))
            all_observations.append(observations)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_masks.append(masks)

        # Visualize results if we have collected any data
        if policy_entropies:
            # Plot policy entropy
            plt.figure(figsize=(12, 6))
            plt.plot(policy_entropies)
            plt.title('Policy Entropy Over Time')
            plt.xlabel('Step')
            plt.ylabel('Entropy')
            plt.savefig('policy_entropy.png')
            wandb.log({"policy_entropy_plot": wandb.Image('policy_entropy.png')})
            plt.close()

            # Plot action probabilities
            if action_probs_over_time:
                max_length = max(len(action) for action in action_probs_over_time)

                # Pad the lists with NaN (or any other value) to make them uniform
                padded_action_probs = [action + [np.nan] * (max_length - len(action)) for action in
                                       action_probs_over_time]
                action_probs_array = np.array(padded_action_probs)
                plt.figure(figsize=(12, 6))
                plt.imshow(action_probs_array.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Probability')
                plt.title('Action Probability Distribution Over Time')
                plt.xlabel('Step')
                plt.ylabel('Action')
                plt.savefig('action_probs.png')
                wandb.log({"action_probs_plot": wandb.Image('action_probs.png')})
                plt.close()

        print(f'len observations {len(observations)}')

        all_q_s, all_advantages = agent.estimate_return(all_rewards)
        mean_reward = calculate_mean(all_rewards)
        print("Reward mean is ", mean_reward)

        current_loss, entropy, kl = agent.update_parameters(all_observations, all_actions, all_advantages, all_masks)
        loss.append(current_loss)
        mean_loss.append(current_loss)

        wandb.log({
            "loss": current_loss,
            "makespan": np.mean(makespans),
            "average_completion": np.mean(average_completions),
            "average_slowdowns": np.mean(average_slowdowns),
            "Mean_reward": mean_reward,
            "approx kl div": kl,
            "entropy": entropy
        })

    file_path = 'REINFORCEMakespanModels-total50/model' + str(job_chunk) + '.pt'
    torch.save(agent.brain.state_dict(), file_path)
    wandb.finish()
