from abc import ABC


class RewardGiver(ABC):
    def __init__(self):
        self.simulation = None
        self.total_reward=0
        self.total_number_of_tasks=0

    def attach(self, simulation):
        self.simulation = simulation

    def get_reward(self):
        if self.simulation is None:
            raise ValueError('Before calling method get_reward, the reward giver '
                             'must be attach to a simulation using method attach.')



class EnergyOptimisationRewardGiverV2(RewardGiver):
    def get_reward(self):
        super().get_reward()

        power_consumption=0
        total_energy = 0
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        for machine in self.simulation.cluster.machines:
            #tasks = machine.task_instances
            #total_cpu_utilization = np.sum([task.cpu for task in tasks])
            #total_memory_utilization = np.sum([task.memory for task in tasks])
            #otal_duration = np.sum([task.duration for task in tasks])

            #if total_duration == 0:
              #  continue  # Skip if no tasks are running on the machine

           # cpu_utilization_ratio = total_cpu_utilization / machine.cpu_capacity
           #memory_utilization_ratio = total_memory_utilization / machine.memory_capacity
            #[p_idle, p_busy, r] = energy_usage[machine.name]

            # Calculate power consumption for the machine based on CPU and memory utilization
            #print("power of machine",machine.power())
            power_consumption += machine.power()

            #print('ahna hne')
            #print(power_consumption)

            # Normalize power consumption
            #normalized_power = (p_busy - power_consumption) / (p_busy - p_idle)

            # Calculate energy consumption
            #energy_consumption = power_consumption * total_duration

            #total_energy += energy_consumption

        return -power_consumption





class EnergyOptimisationTask(RewardGiver):
    def get_reward(self):
        super().get_reward()

        min_power=sum([machine.min_power for machine in self.simulation.cluster.machines])
        max_power=sum([machine.max_power for machine in self.simulation.cluster.machines])
        power_consumption=0
        total_energy = 0
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        decision=cluster.get_decisions
        if decision[0]!=None:
            power_task=decision[0].get_power_consumed
            #power_task_=decision[1].cpu
        else:
            power_task=0
        # for machine in self.simulation.cluster.machines:
        #
        #     power_consumption += machine.power()
        # normalized_power = (power_consumption - min_power) / (max_power - min_power)
        #
        # self.total_reward+=power_consumption
        # if len(cluster.finished_tasks)!=0 :
        #     self.total_number_of_tasks+=len(cluster.finished_tasks)
        #
        #
        #     return -self.total_reward*2.77778e-7
        # else:
        #     return 0
        return -power_task


class CombinedReward(RewardGiver):
    def __init__(self, energy_reward,makespan_reward,energy_weight=1, makespan_weight=0, reward_per_timestamp=-1):
        super().__init__()
        self.energy_weight = energy_weight
        self.makespan_weight = makespan_weight
        self.reward_per_timestamp = reward_per_timestamp
        self.energy_reward = energy_reward
        self.makespan_reward = makespan_reward

    def get_reward(self):
        super().get_reward()
        self.energy_reward.attach(self.simulation)
        self.makespan_reward.attach(self.simulation)

        # Get Energy Reward
        energy_reward = self.energy_reward.get_reward()
        # print('power', energy_reward)

        # Get Makespan Reward
        # makespan_reward = self.makespan_reward.get_reward()
        # print('makespan_reward', makespan_reward)
        delay_penalty=len(self.simulation.cluster.tasks_which_has_waiting_instance)
        num_running_task_instances=len(self.simulation.cluster.running_task_instances)
        # print('tasks_which_has_waiting_instance', delay_penalty)
        # print('running_task_instances', num_running_task_instances)

        # Combine rewards
        combined_reward = (self.energy_weight * energy_reward +
                           self.makespan_weight * delay_penalty )
        # print('combined_reward', combined_reward)

        return combined_reward









class EnergyBaseline(RewardGiver):
    def get_reward(self):
        super().get_reward()
        self.total_number_of_tasks=self.simulation.cluster.numberofallocation
        power = 0
        min_power=sum([machine.min_power for machine in self.simulation.cluster.machines])
        max_power=sum([machine.max_power for machine in self.simulation.cluster.machines])

        decision=self.simulation.cluster.get_decisions[-1]

        baseline=max([machine.get_power_consumed(decision[0]) for machine in self.simulation.cluster.machines if machine!=decision[0]])
        if decision!=None:
            power_task=decision[0].get_power_consumed(decision[0])
            gain=baseline-power_task

            # print('haamdoulah')

            #power_task_=decision[1].cpu
        else:
            # print('we are here')
            gain=0

        return gain





class Energy(RewardGiver):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.total_energy = 0  # Initialize total energy counter
        self.last_time = 0     # Track last timestamp
        self.cumulative_power = 0  # Track cumulative power
    def get_reward(self):
        super().get_reward()
        self.total_number_of_tasks=self.simulation.cluster.numberofallocation
        power = 0
        min_power=sum([machine.min_power for machine in self.simulation.cluster.machines])
        max_power=sum([machine.max_power for machine in self.simulation.cluster.machines])
        if 'cumulative_power' not in locals():
            cumulative_power = 0
        # if self.total_number_of_tasks!=0 :
            # print('we are here ')
        load=[]
        for machine in self.simulation.cluster.machines:
            power += machine.power()
            # print('power',power)
            # print('self.total_number_of_tasks',self.total_number_of_tasks)

            # print('reward is : ',-power  //  self.total_number_of_tasks)
            # normalized_power = (power - min_power) / (max_power - min_power)
            load.append(len(machine.running_task_instances))
        # Initialize load penalty
        load_penalty = 0.0
        cumulative_power += power

        # Calculate the average load
        # if len(load) > 0:
        #     avg_load = sum(load) / len(load)
        #
        #     # Calculate the load penalty as the sum of absolute deviations from the average load
        #     load_penalty = sum(abs(machine_load - avg_load) for machine_load in load)
        #
        #     # Optionally scale the penalty by a factor beta if needed
        #     beta =-0.1  # Adjust this scaling factor as needed
        #     load_penalty *= beta
        cluster = self.simulation.cluster

        decision_time=cluster.get_times
        current_time = self.simulation.env.now

        delta_t=current_time - decision_time
        self.total_energy =cumulative_power
        # if delta_t > 0:
        #     # Calculate total energy consumed since last decision with non-zero time step
        #     energy_consumed = cumulative_power * delta_t
        #     self.total_energy += energy_consumed
        #
        #     # Reset cumulative power after calculating energy
        #     cumulative_power = 0
        #
        #     # Calculate reward based on total energy (negative reward to minimize energy)
        #     reward = -energy_consumed
        #
        #     # Update last decision time
        #     # last_decision_time = current_time
        # else:
        #     # No reward or energy calculation yet if no time has passed; continue accumulating power.
        #     reward = 0
        # print(f'load_penalty {load_penalty}')
        #power_=-0.125*power/(self.total_number_of_tasks +1e-8)
        #penalty=(power_+load_penalty)
        # print(f'delta_t {delta_t}')

        # power_=-power* delta_t
        return -power # add task duration and features ?

    def give_reward_rl(self):
        super().get_reward()
        power=0
        for machine in self.simulation.cluster.machines:
            power += machine.power()
        cluster = self.simulation.cluster

        decision_time=cluster.get_times
        # print(f'decision_time is {decision_time} ')
        current_time = self.simulation.env.now

        delta_t=current_time - decision_time
        if delta_t>0 :
            penalize=delta_t
        else:
            penalize=0
        power_ = -0.125 * power
        reward=power_ - penalize
        # print(f'power is {power_}')
        # print(f'penalize is {penalize}')

        return reward

    def get_total_energy(self):
        """Return total energy consumption"""
        ep_energy=self.total_energy
        self.total_energy=0
        return ep_energy






class Power(RewardGiver):
    def get_reward(self):
        super().get_reward()
        self.total_number_of_tasks=self.simulation.cluster.numberofallocation
        power = 0
        min_power=sum([machine.min_power for machine in self.simulation.cluster.machines])
        max_power=sum([machine.max_power for machine in self.simulation.cluster.machines])
        # if self.total_number_of_tasks!=0 :
            # print('we are here ')
        load=[]
        for machine in self.simulation.cluster.machines:
            power += machine.power()
            # print('power',power)
            # print('self.total_number_of_tasks',self.total_number_of_tasks)

            # print('reward is : ',-power  //  self.total_number_of_tasks)
            # normalized_power = (power - min_power) / (max_power - min_power)
            load.append(len(machine.running_task_instances))
        # Initialize load penalty
        load_penalty = 0.0

        # Calculate the average load
        if len(load) > 0:
            avg_load = sum(load) / len(load)

            # Calculate the load penalty as the sum of absolute deviations from the average load
            load_penalty = sum(abs(machine_load - avg_load) for machine_load in load)

            # Optionally scale the penalty by a factor beta if needed
            beta =-0.1  # Adjust this scaling factor as needed
            load_penalty *= beta

        # print(f'load_penalty {load_penalty}')
        power_=-0.125*power/(self.total_number_of_tasks +1e-8)
        penalty=(power_+load_penalty)
        power=-power/(sum(load) +1e-8)
        return power # add task duration and features ?






class MakespanRewardGiver(RewardGiver):
    name = 'Makespan'

    def __init__(self, reward_per_timestamp=-1):
        super().__init__()
        self.reward_per_timestamp = reward_per_timestamp

    def get_reward(self):
        super().get_reward()

        cluster = self.simulation.cluster

        decision_time = cluster.get_times
        current_time = self.simulation.env.now

        delta_t = current_time - decision_time

        if delta_t > 0:
            # Calculate total energy consumed since last decision with non-zero time step

            # Reset cumulative power after calculating energy


            # Calculate reward based on total energy (negative reward to minimize energy)
            self.reward_per_timestamp = -1

        else:
            # No reward or energy calculation yet if no time has passed; continue accumulating power.
            self.reward_per_timestamp = 0

        return self.reward_per_timestamp






class EnergyOptimisationRewardGiver(RewardGiver):
    def get_reward(self):
        super().get_reward()

        min_power=sum([machine.min_power for machine in self.simulation.cluster.machines])
        max_power=sum([machine.max_power for machine in self.simulation.cluster.machines])
        power_consumption=0
        total_energy = 0
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        for machine in self.simulation.cluster.machines:

            power_consumption += machine.power()
        normalized_power = (power_consumption - min_power) / (max_power - min_power)


        self.total_reward+=power_consumption
        # if len(cluster.finished_tasks)!=0 :
        #     self.total_number_of_tasks+=len(cluster.finished_tasks)
        #
        #
        #     return -self.total_reward*2.77778e-7
        # else:
        #     return 0
        return -self.total_reward * 2.77778e-7


class MakespanRewardGiver(RewardGiver):
    name = 'Makespan'

    def __init__(self, reward_per_timestamp):
        super().__init__()
        self.reward_per_timestamp = reward_per_timestamp

    def get_reward(self):
        super().get_reward()
        return self.reward_per_timestamp


class AverageSlowDownRewardGiver(RewardGiver):
    name = 'AS'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += (- 1 / task.task_config.duration)
        return reward


class AverageCompletionRewardGiver(RewardGiver):
    name = 'AC'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        return - unfinished_task_len
