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
        energy_usage = {"PowerEdgeC5220": [194, 672, 0.9569],
                        "HPProLiantDL2000": [178, 660, 0.7257],
                        "IBMSystemx3630M4": [58.1, 269, 1.5767],
                        "IBMSystemxiDataPlexdx360M3": [92.7, 341, 0.7119],
                        "Systemx3200M3": [45, 119, 1.5324]
                        }
        power = 0
        for machine in self.simulation.cluster.machines:
            [p_idle, p_busy, r] = energy_usage[machine.name]
            power += p_idle + (p_busy-p_idle)*(machine.state["cpu"])**r

        return -power   # add task duration and features ?
