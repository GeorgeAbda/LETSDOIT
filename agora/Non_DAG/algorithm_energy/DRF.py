from cogito.alogrithm import Algorithm


class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class DRF(Algorithm):
    def __init__(self, reward_giver):
        self.reward_giver = reward_giver
        self.current_trajectory = []
        self.always_reward=False
        self.agent=None

    def __call__(self, cluster, clock):
        machines = cluster.machines
        unfinished_tasks = cluster.unfinished_tasks
        candidate_task = None
        candidate_machine = None

        for machine in machines:
            for task in unfinished_tasks:
                if machine.accommodate(task):
                    candidate_machine = machine
                    candidate_task = task
                    break
        if candidate_task is None:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
        else:
            self.current_trajectory.append(Node(None, None, 0, clock))
        return candidate_machine, candidate_task
