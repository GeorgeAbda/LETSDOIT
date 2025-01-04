import torch
from agora.DAG.utils.feature_synthesize import task_features, father_task_indices
from agora.DAG.utils.DAG import DAG
from agora.DAG.adapter.job import JobBatch
import numpy as np
from agora.DAG.algorithm.gnnDeepRL.utils import *
from agora.DAG.algorithm.gnnDeepRL.viz import *

from agora.Non_DAG.algorithm_energy.tetris import *
import os
class RLAlgorithm:
    def __init__(self, epsilon,agent, reward_giver, features_normalize_func, features_extract_func,dag_directory=None,wandb=None):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.epsilon = epsilon
        self.always_reward=False
        self.wandb=wandb
        self.job_batches = []  # Initialize job_batches
        self.current_trajectory = []
        self.epsilon = epsilon

        self.embeddings_log = []  # To store embeddings for visualization
        self.count=0

        if dag_directory is None:
            dag_directory = os.path.join(os.getcwd(),'dags')
        self.dag_directory=dag_directory
        if not os.path.exists(self.dag_directory):
            os.makedirs(self.dag_directory)

    def log_trajectory(self, clock,observation,action,logprobs,valids,gnn_embeddings):
        self.current_trajectory.append(Node(observation, action,self.reward_giver.get_reward(),clock,logprobs,valids,gnn_embeddings))

    def extract_features(self, cluster, tasks):
        # Group tasks by job
        jobs = {}
        for task in tasks:
            if task.job not in jobs:
                jobs[task.job] = []
            jobs[task.job].append(task)
        # Extract resource features
        resource_features = self.extract_resource_features(cluster)
        # for job,_ in jobs.items():
        #     job.update_tasks_map()
            # Create DAGs for each job
        dags = [DAG(job, resource_features) for job, tasks in jobs.items()]
        # if len(dags) > 1:
        #     # print(f' --------------------------------Here we have an observation with {len(dags)} !! ')

        for job_id,dag in enumerate(dags):
            #print(f'DAG directory {self.dag_directory}')
            dag.visualize(self.dag_directory, f"dag_{job_id}_{len(cluster.tasks_which_has_waiting_instance)}_{len(cluster.machines)}")
            #print(f" end visualisaing dag_{job_id}")

        return JobBatch(dags, resource_features)

    def extract_resource_features(self, cluster):
        resource_features = []
        for machine in cluster.machines:
            resource_features.append([
                machine.cpu,
                machine.memory,
                machine.disk,
                machine.cpu_capacity,
                machine.memory_capacity,
                machine.disk_capacity,
                machine.processing
            ])
        return torch.tensor(resource_features, dtype=torch.float32)

    def update_gnn_embeddings(self, cluster):
        # Recompute GNN embeddings for all tasks across all job batches
        #all_tasks = [task for job_batch in self.job_batches for task in job_batch.tasks]
        #job_batch_features = self.extract_features(cluster, all_tasks)

        waiting_tasks = cluster.all_tasks
        with torch.autograd.detect_anomaly(True):

            # Extract features for these waiting tasks
            job_batch_features = self.extract_features(cluster, waiting_tasks)
            obervation_batch=(job_batch_features.dag,job_batch_features.resource_features)
            if len(job_batch_features.dag)>1:
                obervation_batch=(job_batch_features.dag,job_batch_features.resource_features)
                # print(f'obsrvation batch {obervation_batch}')
                #
                # print('Here we have an observation with')
                # print(f' --------------------------------Here we we will pass it in prepare_batch {len(job_batch_features)} !! ')

            batch_dags,resources_features = prepare_batch([obervation_batch],1)

            features =(job_batch_features.dag, job_batch_features.resource_features)

            features=(batch_dags,resources_features)
            print(f'batch_dags: {batch_dags}, resources_features: {resources_features.shape}')
           # with torch.no_grad():
            #updated_gnn_embeddings = self.agent.gnn(batch_dags,resources_features)
            action_probs,value= self.agent.actor_critic(batch_dags,resources_features)
            gnn_embeddings = self.agent.actor_critic.get_embeddings()
            print("gnn_embeddings tensor requires_grad:", gnn_embeddings.requires_grad)  # True


        # print(f'gnn_embeddings shape {gnn_embeddings.shape}')
            #self.embeddings_log.append(gnn_embeddings)  #
           # print(f'action_probs shape: {action_probs.shape}')
        return action_probs,features,gnn_embeddings,value

    def get_embeddings_log(self):
        return self.embeddings_log
    def __call__(self, cluster, clock):
        tasks = cluster.all_tasks
        machines = cluster.machines
        # print(f'we have {len(tasks)} tasks waiting')
        # Initialize a list to store valid (machine, task) pairs
        valid_candidates = []
        for index,job in enumerate(self.job_batches):
            if len(job.tasks)==0:
                self.job_batches.pop(index)
               # print('done with updating job_batches')

        no_valid_candidates=[]
        all_candidates=[]
        # Check for valid task-machine pairs
        for machine in machines:
            for task in tasks:
                all_candidates.append((machine, task))
                if machine.accommodate(task) and task.ready and task in cluster.tasks_which_has_waiting_instance:
                    valid_candidates.append((machine, task))
                else:
                    no_valid_candidates.append((machine.cpu,task.task_config.cpu,machine.memory,task.task_config.memory,
               machine.disk,task.task_config.disk))

        # print(f'we have {len(valid_candidates)} valid candidates we can schedule')

        if len(all_candidates)>0:
            self.count=0
        if len(all_candidates) == 0:
            self.count += 1
            return None, None, False
        if len(valid_candidates) == 0 and not  len(all_candidates) == 0:
            probs=torch.zeros((len(all_candidates)))
            probs[0] = float(-1e-20)  # Example for the first index

            node=Node(None, None,self.reward_giver.get_reward() , clock,probs,None,self.reward_giver.give_reward_rl(),None,0)
            self.current_trajectory.append(node)

            return None, None,node
        # if not valid_candidates:
        #     #print(no_valid_candidates)
        #     probs=torch.zeros((len(all_candidates)))
        #     probs[0] = float('nan')  # Example for the first index
        #
        #     node=Node(None, None,self.reward_giver.get_reward() , clock,probs,None,self.reward_giver.give_reward_rl(),None,0)
        #     self.current_trajectory.append(node)
        #
        #     return None, None,node
        #print(f'len of tasks {len(tasks)}')
        #print(f' len of self.job_batches {len(self.job_batches)}')
       # print(f'Number of tasks in the job {[len(job.tasks) for job in cluster.unfinished_jobs]}')
        # Check for new job arrivals and update the job batches
        new_jobs = [job for job in cluster.jobs if job not in [batch.dag.job for batch in self.job_batches]]


        if len(new_jobs)>0:
            new_job_tasks = [task for job in new_jobs for task in job.tasks]
            self.add_new_jobs(cluster, new_jobs)
            #print(f' len of self.job_batches after update {len(self.job_batches)}')

        # if len(tasks)==1:
        #     print('only one task in the waiting list, scheduling using Heuristic ')
        #
        #     machine, task, _, _, _ = self.heuristic_schedule(cluster, clock)
        #     return  machine, task, None, None, None

        # Update GNN embeddings for all tasks across all job batches
        action_scores, features,gnn_embeddings,value = self.update_gnn_embeddings(cluster)
        #print('features are {features}'.format(features=features))
        # Reshape action_probs to match the number of possible actions
        #action_scores = action_scores.view(-1)

        # print(f'len of valid_candidates {len(valid_candidates)}')
        # print(f'len of all_candidates {len(all_candidates)}')
        # print(f'action_probs shape: {action_scores.shape}')

        valid_indices = [all_candidates.index(candidate) for candidate in valid_candidates]
        #
        # action_scores_masked=action_scores[valid_indices]
        # action_probs= F.softmax(action_scores_masked, dim=0).view(action_scores_masked.size())
        action_mask = torch.zeros(len(all_candidates))
        for candidate in valid_candidates:
            action_mask[all_candidates.index(candidate)] = 1

        # Mask the action scores, setting invalid actions to a very low value
        masked_action_scores = action_scores * action_mask + (1 - action_mask) * -1e10

        # Apply softmax to the masked action scores
        action_probs = F.softmax(masked_action_scores, dim=0)
        logprobs = torch.log(action_probs)

        # Select an action from valid candidates

        # Filter action_probs to only include valid candidates
        #valid_indices = [i for i, (machine, task) in enumerate(valid_candidates)]

        # print('valid_indices:', len(valid_candidates))
        # valid_probs = action_probs.clone()[valid_indices]
        # print('action_probs shape', len(action_probs))
        p=action_probs.clone().detach().numpy()
        if len(action_probs) == 0:
            # If no valid actions, choose randomly
            chosen_index = np.random.choice(len(valid_candidates))
        # if np.random.random() < self.epsilon:
        #     chosen_index = np.random.choice(len(valid_candidates))
        #     # print(f"we will select chosen_index from randomly {chosen_index}")
        #
        #     chosen_all_index = valid_indices[chosen_index]
        else:
            chosen_index = np.random.choice(len(all_candidates),p=p)
            chosen_all_index=chosen_index

        #chosen_pair = valid_candidates[chosen_index]
        # chosen_all_index = valid_indices[chosen_index]
        chosen_pair = all_candidates[chosen_all_index]

        # print('Decision Made ')
        # print(f'chosen_pair is {chosen_pair}')
        # print(f'chosen_all_index is {chosen_all_index}')
        # print(f'action_probs is {action_probs}')


        node = Node(features, chosen_all_index, 0, clock, action_probs, action_mask, 0,gnn_embeddings,value)
        self.current_trajectory.append(node)
        return chosen_pair[0], chosen_pair[1],node

    def add_new_jobs(self, cluster, new_jobs):
        #print('We will add new jobs')
        for new_job in new_jobs:
        # Create a new job batch for the new job
            new_job_batch = JobBatch(DAG(new_job,cluster.machines), self.extract_resource_features(cluster))
        # Add the new job batch to the existing job batches
            self.job_batches.append(new_job_batch)
            #print('done adding new job batch')
        # Remove the new job from the cluster
            # Remove the new job from the clusterAdd the new job batch to the existing job batches

    def decay_epsilon(self):
        """Decays the epsilon value with a set decay rate, ensuring it doesn't fall below epsilon_min."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print('Epsilon: ', self.epsilon, end='\r')
        else:
            self.epsilon = self.epsilon_min

            print('Epsilon min: ', self.epsilon, end='\r')

class Node(object):
    def __init__(self, observation, action, reward, clock,prob,valids,rlreward,embed,value=None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.rlreward=rlreward
        self.clock = clock
        self.prob=prob
        self.valid_candid=valids
        self.embed=embed
        self.value=value
    def to_tuple(self):
        return (self.observation, self.action, self.reward,self.prob,self.valid_candid)