import os
import time
import numpy as np
import torch
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import wandb

# wandb.login(key='d6c2052b0b861ed85cd3336fe7e480bbcf118805')

sys.path.append('..')

from cogito.machine import MachineConfig
from agora.Non_DAG.algorithm.PPO.save import RLAlgorithm
from agora.Non_DAG.algorithm.PPO.Agent import Agent

from agora.Non_DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver

from agora.Non_DAG.utils.csv_reader import CSVReader
from agora.Non_DAG.utils.feature_functions import features_extract_func_ac, features_normalize_func_ac
from agora.Non_DAG.utils.episode import Episode

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
machines_number = 5
jobs_len = 10
n_iter = 10
n_episode = 12
jobs_csv = '../jobs_files/jobs.csv'

reward_giver = MakespanRewardGiver(-1)
features_extract_func = features_extract_func_ac
features_normalize_func = features_normalize_func_ac

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
agent = Agent(0.9, reward_to_go=True, nn_baseline=True, normalize_advantages=True)

name = "Test Gradient Policy Makespan Total 50"

jobs_configs = csv_reader.generate(jobs_len * 2, jobs_len)

tic = time.time()

algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                        features_normalize_func=features_normalize_func)
episode = Episode(machine_configs, jobs_configs, algorithm, None)
algorithm.reward_giver.attach(episode.simulation)
episode.run()
trajectory = episode.simulation.scheduler.algorithm.current_trajectory

features = []
for node in trajectory:
    features.append(node.features)
features = [feature for feature in features if feature is not None]

feature = features[len(features) // 2 - 1]
print(feature)
fitness_scores = []

# makespan_chunk_path = "REINFORCEMakespanModels-chunk100/model" + str(chunk) + ".pt"
makespan_chunk_path = "REINFORCEMakespanModels-total50/model" + str(49) + ".pt"
agent = Agent(0.9, reward_to_go=True, nn_baseline=True, normalize_advantages=True)
agent.brain.load_state_dict(torch.load(makespan_chunk_path))
fitness = agent.brain(feature).squeeze()
fitness_scores.append(fitness)

# for chunk in range(30):
#     makespan_chunk_path = "REINFORCEMakespanModels-chunk100/model" + str(chunk) + ".pt"
#     agent = Agent(0.9, reward_to_go=True, nn_baseline=True, normalize_advantages=True)
#     agent.brain.load_state_dict(torch.load(makespan_chunk_path))
#     fitness = agent.brain(feature).squeeze()
#     fitness_scores.append(fitness)

for chunk in range(20):
    makespan_chunk_path = "REINFORCEMakespanModels-chunk100/model" + str(chunk) + ".pt"
    agent = Agent(0.9, reward_to_go=True, nn_baseline=True, normalize_advantages=True)
    agent.brain.load_state_dict(torch.load(makespan_chunk_path))
    fitness = agent.brain(feature).squeeze()
    fitness_scores.append(fitness)

fitness_scores = torch.stack(fitness_scores)
correlation_matrix = torch.corrcoef(fitness_scores)
matrix = correlation_matrix.detach().numpy()
np.save('correlation_matrix.npy', matrix)
# Set up the plot
plt.figure(figsize=(70, 50))
# sns.set(font_scale=1.2)
# sns.set_style("whitegrid")

# Create a heatmap with a color map that indicates the correlation strength
sns.heatmap(np.round(matrix, 2), annot=True, cmap="coolwarm", center=0, linewidths=0.5)

# Customize labels, title, and other properties
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()

# Show the plot
plt.show()

# file_path = '/EnergyModels/totalModel.pt'
# agent.brain.model.load_state_dict(torch.load(file_path))
