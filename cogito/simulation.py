from cogito.monitor import Monitor
from agora.DAG.algorithm.gnnDeepRL.DRL_AccumulatedReward import *
from agora.DAG.utils.feature_synthesize import task_features, father_task_indices
from agora.DAG.utils.feature_synthesize import task_features, father_task_indices
from agora.DAG.utils.DAG import DAG
from agora.DAG.adapter.job import JobBatch
import numpy as np
from agora.DAG.algorithm.gnnDeepRL.utils import *
from agora.DAG.algorithm.gnnDeepRL.viz import *

class Simulation(object):
    def __init__(self, env, cluster, task_broker, scheduler, event_file):
        self.env = env
        self.cluster = cluster
        self.task_broker = task_broker
        self.scheduler = scheduler
        self.event_file = event_file
        if event_file is not None:
            self.monitor = Monitor(self)

        self.task_broker.attach(self)
        self.scheduler.attach(self)

    def run(self):
        # Starting monitor process before task_broker process
        # and scheduler process is necessary for log records integrity.
        if self.event_file is not None:
            self.env.process(self.monitor.run())
        self.env.process(self.task_broker.run())
        self.env.process(self.scheduler.run())

    @property
    def finished(self):
        return self.task_broker.destroyed \
               and len(self.cluster.unfinished_jobs) == 0


    # def step(self):
    #     action=self.cluster.action
    #     cluster1=self.cluster.copy
    #     cluster1.resources.cpu-=
    #
    #     # Extract features for these waiting tasks
    #     job_batch_features = self.scheduler.algorithm.extract_features(cluster1, cluster1.all_tasks)
    #     observation_batch = (job_batch_features.dag, job_batch_features.resource_features)
    #     if len(job_batch_features.dag) > 1:
    #         observation_batch = (job_batch_features.dag, job_batch_features.resource_features)
    #         # print(f'obsrvation batch {obervation_batch}')
    #         #
    #         # print('Here we have an observation with')
    #         # print(f' --------------------------------Here we we will pass it in prepare_batch {len(job_batch_features)} !! ')
    #
    #     batch_dags, resources_features = prepare_batch([observation_batch], 1)
    #     new_state=(batch_dags, resources_features)
    #
    #     reward=
    #
    #     return new_state, reward, done


# def reward_simulation(cluster):
#
#     power = 0
#     for machine in cluster.machines:
#         power += machine.power()
#
#     decision_time = cluster.get_times
#     # print(f'decision_time is {decision_time} ')
#     current_time = self.simulation.env.now
#
#     delta_t = current_time - decision_time
#     if delta_t > 0:
#         penalize = delta_t
#     else:
#         penalize = 0
#     power_ = -0.125 * power
#     reward = power_ - penalize
#     return reward
