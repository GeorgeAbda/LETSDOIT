import simpy
from cogito.cluster import Cluster
from cogito.scheduler import Scheduler
from cogito.broker import Broker
from cogito.simulation import Simulation


class Episode(object):
    broker_cls = Broker

    def __init__(self, machine_configs, task_configs, algorithm, event_file,episode):
        self.env = simpy.Environment()
        self.episode=episode
        cluster = Cluster()
        cluster.add_machines(machine_configs)

        task_broker = Episode.broker_cls(self.env, task_configs,self.episode)

        scheduler = Scheduler(self.env, algorithm)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def run(self):
        self.simulation.run()
        self.env.run()

    def step(self):
        self.simulation.run()
        self.env.step()

    def done(self):
        return self.env.now >= self.env.peek()

class EpisodeTest(object):
    broker_cls = Broker

    def __init__(self, machine_configs, task_configs, algorithm, event_file):
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)

        task_broker = Episode.broker_cls(self.env, task_configs)

        scheduler = Scheduler(self.env, algorithm)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def run(self):
        self.simulation.run()
        self.env.run()

    def step(self):
        self.simulation.run()
        self.env.step()

    def done(self):
        return self.env.now >= self.env.peek()
