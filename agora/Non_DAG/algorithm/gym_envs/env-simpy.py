import csv
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

task_features = []
task_duration = []

with open('job.csv', 'r') as file:
    reader = csv.reader(file)

    # Read the single row containing all the values
    row = next(reader)

    # Split the row by commas and extract the required information
    for value in row:
        values = value.split(',')
        cpu = float(values[3])  # Extract CPU value from index 3
        memory = float(values[4])  # Extract memory value from index 4
        duration = float(values[2])  # Extract duration value from index 2

        task_features.append((cpu, memory))  # Append CPU and memory duo to task_features list
        task_duration.append(duration)  # Append duration to task_duration list

import gym
from gym import spaces


class SimPyGymEnvironment(gym.Env):
    def __init__(self, simpy_env, number_servers, number_tasks, episode):
        super(SimPyGymEnvironment, self).__init__()
        self.number_servers = number_servers
        self.number_tasks = number_tasks
        self.episode = episode

        # Define the observation space and action space
        self.continuous_state_spaces = [spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), shape=(3,)) for
                                        server in range(self.number_servers)]
        self.observation_space = spaces.Tuple(self.continuous_state_spaces)
        self.action_space = spaces.Discrete(self.number_servers * self.number_tasks + 1)

    def reset(self):
        # Step 4: Reset the SimPy environment to its initial state
        self.episode = self.episode

        # Return the initial observation (servers are not used)
        initial_observation = tuple(np.zeros(3) for _ in range(self.number_servers))
        return initial_observation

    def step(self, action):
        if action == 0:
            server, task = None, None
        else:
            server, task = ..., ...

        self.simpy_env.step(action)

        # Obtain the new observation, reward, and done flag from your SimPy environment
        observation = self.simpy_env.get_observation()
        reward = self.simpy_env.get_reward()

        # Determine the termination condition based on the number of remaining events
        done = self.episode.done()

        # Return the observation, reward, done flag, and an empty dictionary (additional information)
        return observation, reward, done, {}

    def render(self):
        # Visualize or display the environment's state

        # Prepare the data for plotting
        servers = range(self.number_servers)
        observation_array = np.array(self.continuous_state_spaces)
        cpu_utilizations = observation_array[:, 0]
        memory_utilizations = observation_array[:, 1]

        # Create the figure and axes
        fig, ax = plt.subplots()

        # Plot the CPU and memory utilizations
        ax.plot(servers, cpu_utilizations, label='CPU')
        ax.plot(servers, memory_utilizations, label='Memory')

        # Set the plot labels and title
        ax.set_xlabel('Server Index')
        ax.set_ylabel('Utilization')
        ax.set_title('CPU and Memory Utilization')

        # Set the y-axis limits
        ax.set_ylim(0, 1)

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()
