import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import csv

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


class CustomEnv(gym.Env):
    def __init__(self, number_servers, tasks, tasks_duration):
        super(CustomEnv, self).__init__()
        # Define your observation space
        self.tasks = tasks
        self.tasks_duration = tasks_duration
        self.number_tasks = len(tasks)
        self.tasks_executed = []
        self.number_servers = number_servers
        self.continuous_state_spaces = [spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), shape=(2,)) for
                                        server in
                                        range(self.number_servers)]
        self.observation_space = spaces.Tuple(self.continuous_state_spaces)
        # Define your action space
        self.action_space = spaces.Discrete(self.number_servers * self.number_tasks + 1)

        # Set up any other necessary attributes

    def reset(self):
        # Reset the environment to its initial state
        # Reset the environment to its initial state

        # Initialize each continuous observation to zeros
        initial_observations = [np.zeros((2,)) for server in range(len(self.continuous_state_spaces))]

        # Combine the initial observations into a tuple
        initial_observation = tuple(initial_observations)

        return initial_observation

    def step(self, action):
        # Execute the chosen action and update the environment state
        task_index = action % len(self.continuous_state_spaces)  # -> find the task associated to the action : action =
        # nb_server*total_tasks + nb_task
        server_index = action // len(self.continuous_state_spaces)

        self.continuous_state_spaces[server_index] -= np.array(self.tasks[task_index])
        self.observation_space = spaces.Tuple(self.continuous_state_spaces)
        # Calculate rewards and check termination conditions
        reward = ...
        done = False
        if len(self.tasks_executed) == len(self.tasks):
            done = True
        else:
            self.tasks_executed.append(server_index)

        return self.observation_space, reward, done

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

    def get_available_actions(self):
        # Determine the available actions based on the current state
        ...

