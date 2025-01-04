from abc import ABC


class RewardGiver(ABC):
    def __init__(self):
        self.simulation = None

    def attach(self, simulation):
        self.simulation = simulation

    def get_reward(self):
        if self.simulation is None:
            raise ValueError('Before calling method get_reward, the reward giver '
                             'must be attach to a simulation using method attach.')


class EnergyOptimisationRewardGiver(RewardGiver):
    def get_reward(self):
        super().get_reward()
        state = self.simulation.cluster.state
        cpu = state['cpu']
        energy_usage = [0, 194, 254, 303, 345, 386, 427]
        energy_class = int(cpu*100//10)
        if energy_class > 6:
            energy_class = 6
        return -energy_usage[energy_class]
