

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import os

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_resource_utilization(cluster, n_resources, timesteps, itr, ep):
    """
    Plot resource utilization over time and calculate total allocations for each resource.
    """
    model_dir = f'./XAI/Resource_Utilization/{itr}'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    decisions = cluster.get_decisions
    timesteps = len(decisions)

    episode_actions = [decision[0] for decision in decisions]
    resource_usage = np.zeros((n_resources, timesteps))

    for t, (resource, _) in enumerate(decisions):
        resource_usage[resource.index, t] = 1

    # Calculate total allocations for each resource
    total_allocations = np.sum(resource_usage, axis=1)

    print("Total allocations for each resource:", total_allocations)

    plt.figure(figsize=(12, 6))

    # Define a custom color cycle
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for r in range(n_resources):
        plt.plot(range(timesteps), resource_usage[r],
                 label=f'Resource {r}', color=colors[r % len(colors)], linewidth=0.5, alpha=0.7)

    plt.xlabel('Timestep')
    plt.ylabel('Utilization')
    plt.title('Resource Utilization Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_dir}/RU_ep{ep}.png')
    plt.close()


def visualize_features(features, itr, ep,j):
    # Create directory if it does not exist
    features_=features.clone().detach().numpy()
    model_dir = f'./XAI/Features/{itr}/{ep}'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Assuming `features` is a 3D tensor with shape (x, y, z)
    num_tasks, num_resources, combined_feature_dim = features.shape

    # Extract the task embeddings (first part of the combined embeddings)
    task_embeddings = features_[:, :, :combined_feature_dim // 2]  # Adjust slicing if needed

    # Plotting task_embeddings as a heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(task_embeddings.reshape(num_tasks, -1), aspect='auto', cmap='viridis')
    plt.colorbar(label='Task Embedding Value')
    plt.title(f'Task Embeddings Visualization - Episode {ep}')
    plt.xlabel('Feature Index (across all tasks and resources)')
    plt.ylabel('Task Index')
    # Save the plot
    plt.savefig(f'{model_dir}/Features_tra{j}.png')
    plt.close()

    # Loop over each task (slice along the x axis) and plot a heatmap for each
    for i in range(num_tasks):
        plt.figure(figsize=(10, 5))
        plt.imshow(features_[i], aspect='auto', cmap='viridis')
        plt.colorbar(label='Feature Value')
        plt.title(f'Task {i + 1} - Resources vs Features')
        plt.xlabel('Features')
        plt.ylabel('Resources')

        # Save each slice as an image in the model directory
        plt.savefig(f'{model_dir}/Features_tra{j}_task{i + 1}.png')
        plt.close()  # Close the figure to save memory
