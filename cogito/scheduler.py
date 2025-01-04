from agora.DAG.algorithm.gnnDeepRL.utils import *
import wandb

wandb.login(key='7dea3f024f3b2592c4d66656caf1c039db736d53')
device = torch.device("cpu")
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from cogito.global_context import global_context
import numpy as np
import torch
class Scheduler(object):
    def __init__(self, env, algorithm):
        self.env = env
        self.j=0
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        if self.algorithm.agent:
            self.agent = self.algorithm.agent  # Agent to interact with the scheduler
        self.valid_pairs = {}
        global_context.algorithm = algorithm  # Set the algorithm in the global context
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions=[]
        self.i=0
        self.values_policy=[]
        self.entropy_term=0

        self.batch_rewards = []
        self.batch_values = []
        self.batch_values_policy = []
        self.batch_log_probs = []
        self.batch_actions = []
        self.batch_entropy_terms = []
        self.bol=True
    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def make_decision(self):
        batch_size =len(self.cluster.all_tasks_instances)//6
        # print(batch_size)
        while True:
            if self.algorithm.always_reward:
                machine, task, features, action, probs, valids, gnn_embeddings = self.algorithm(self.cluster,
                                                                                                self.env.now)
            else:
                machine, task, node = self.algorithm(self.cluster, self.env.now)

            if node == False:
                break

            if machine is None or task is None and node != False:
                # print("Handle no valid machine-task pair")
                # Handle no valid machine-task pair
                reward = node.rlreward
                value = torch.tensor(node.value).unsqueeze(0)
                self.batch_values.append(value)
                self.batch_values_policy.append(value.numpy())
                self.batch_log_probs.append(node.prob)
                self.batch_rewards.append(reward)
                self.bol=False
                break
            else:
                self.cluster.add_decision((machine, task))
                self.cluster.add_times(self.simulation.env.now)

                if self.algorithm.always_reward:
                    task.start_task_instance(machine, features, action, probs, valids)
                else:
                    # print("start_task_instance")

                    task.start_task_instance(machine)
                    probs = node.prob
                    mask = node.valid_candid
                    probs_ = probs * mask + (1 - mask) * 1e-3
                    reward = node.rlreward
                    self.batch_rewards.append(reward)
                    # print(f'len(batch_rewards) {len(self.batch_rewards)}')
                    self.bol=True
                    value = node.value
                    self.batch_values.append(value)
                    self.batch_values_policy.append(value.detach().numpy())

                    if probs[0] == float(-1e-20):
                        continue

                    entropy = -torch.sum(torch.mean(probs) * torch.log(probs + 1e-8))
                    self.batch_entropy_terms.append(entropy)
                    self.batch_actions.append(node.action)
                    self.batch_log_probs.append(probs)

            # Process the batch when the size is reached
            if len(self.batch_rewards) >= batch_size:
                print('batch size reached')
                self.process_batch(self.batch_rewards, self.batch_values, self.batch_values_policy, self.batch_log_probs, self.batch_actions,
                                   self.batch_entropy_terms)
                self.batch_rewards, self.batch_values, self.batch_values_policy, self.batch_log_probs, self.batch_actions, self.batch_entropy_terms = [], [], [], [], [], []

        # Process any remaining data in the batch
        if self.batch_rewards and self.bol:
            print('processing remaining batch')
            self.process_batch(self.batch_rewards, self.batch_values, self.batch_values_policy, self.batch_log_probs, self.batch_actions,
                               self.batch_entropy_terms)

    def process_batch(self, rewards, values, values_policy, log_probs, actions, entropy_terms):
        # Normalize rewards and values
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        values = torch.FloatTensor(values)
        values = (values - values.mean()) / (values.std() + 1e-8)
        values_policy = torch.FloatTensor(values_policy)
        values_policy = (values_policy - values_policy.mean()) / (values_policy.std() + 1e-8)

        # Calculate loss, backpropagate, and optimize
        Qval=self.get_new_state_sepernoembed()
        Qval = Qval.detach().numpy()

        ac_loss,critic_loss=self.agent.update(values,rewards,Qval,log_probs,entropy_terms,values_policy,actions)
        if ac_loss!= None:
            self.log_loss(ac_loss,critic_loss)

    #
    # def make_decision(self):
    #     while True:
    #         if self.algorithm.always_reward:
    #             machine, task,features ,action,probs,valids,gnn_embeddings= self.algorithm(self.cluster, self.env.now)
    #         else:
    #             machine, task,node= self.algorithm(self.cluster, self.env.now)
    #         if node==False:
    #
    #             break
    #         if machine is None or task is None and node!=False :
    #             # print('No machine or no task')
    #             reward = node.rlreward
    #             value = torch.tensor(node.value).unsqueeze(0)
    #             self.values.append(value)
    #
    #             value_ =value.numpy()
    #             # print(type(value), value.shape if isinstance(value, np.ndarray) else None)
    #
    #             self.values_policy.append(value_)
    #
    #             self.log_probs.append(node.prob)
    #             self.rewards.append(reward)
    #             break
    #         else:
    #             self.cluster.add_decision((machine, task))
    #             self.cluster.add_times(self.simulation.env.now)
    #             if self.algorithm.always_reward:
    #
    #                 task.start_task_instance(machine,features,action,probs,valids)
    #             else:
    #                 task.start_task_instance(machine)
    #
    #                 probs=node.prob
    #                 mask=node.valid_candid
    #                 probs_ = probs * mask + (1 - mask) * 1e-3
    #                 reward = node.rlreward
    #                 self.rewards.append(reward)
    #                 value=node.value
    #                 self.values.append(value)
    #                 value_ = value.detach().numpy()
    #                 # print(type(value), value.shape if isinstance(value, np.ndarray) else None)
    #                 # print("Grad enabled for detached tensor:", value_.requires_grad)
    #                 # print("Grad enabled for saved tensor:", self.values[0].requires_grad)
    #                 epsilon = 1e-8  # A small value to avoid log(0)
    #                 # print(f' probs are {probs}')
    #                 # logprobs = torch.log(probs+epsilon)
    #                 if probs[0] == float(-1e-20):
    #                     continue
    #                 entropy = -torch.sum(torch.mean(probs) * torch.log(probs+1e-8))
    #                 # print(f' entropy are {entropy}')
    #                 self.actions.append(node.action)
    #
    #                 self.values_policy.append(value_)
    #
    #                 self.log_probs.append(probs)
    #                 self.entropy_term = entropy
    #                 self.i=0
    #                 if self.algorithm.agent.name=='Bootstrapping':
    #
    #                     if probs[0] == float(-1e-20):
    #                         continue
    #                     Qval = self.get_new_state_sepernoembed()
    #                     policy_loss,critic_loss=self.algorithm.agent.update(Qval,value,value_,probs,node.action,reward)
    #                     if self.i%10:
    #                         self.log_loss(policy_loss,critic_loss)
    #                     self.i+=1



                # self.j+=1
                # if self.j%30==0:
                #     if self.algorithm.agent.name != 'Bootstrapping':
                #
                #         Qval = self.get_new_state_sepernoembed()
                #         Qval = Qval.detach().numpy()
                #         values=self.values.copy()
                #         rewards=self.rewards.copy()
                #         actions=self.actions.copy()
                #         log_probs=self.log_probs.copy()
                #         entropy_term=self.entropy_term
                #         values_policy=self.values_policy.copy()
                #         ac_loss, critic_loss = self.agent.update(values, rewards, Qval, log_probs,
                #                                                  entropy_term, values_policy, actions)
                #         self.log_loss(ac_loss, critic_loss)
                #         self.log_probs = []
                #         self.values = []
                #         self.rewards = []
                #         self.actions = []
                #         self.i = 0
                #         self.values_policy = []
                #         self.entropy_term = 0
                    # print('Scheduling..')

    def run(self):
        while not self.simulation.finished:
            self.make_decision()
            yield self.env.timeout(1)
        self.destroyed = True

        # final_state=self.get_new_state()
        # if self.algorithm.agent.name != 'Bootstrapping' and (len(self.values)!=0):
        #
        #     Qval=self.get_new_state_sepernoembed()
        #     Qval = Qval.detach().numpy()
        #
        #     ac_loss,critic_loss=self.agent.update(self.values,self.rewards,Qval,self.log_probs,self.entropy_term,self.values_policy,self.actions)
        #     if ac_loss!= None:
        #         self.log_loss(ac_loss,critic_loss)

    def get_new_state_sepernoembed(self):
        machines = self.cluster.machines
        tasks = self.cluster.all_tasks
        valid_candidates = []
        all_candidates = []
        # Check for valid task-machine pairs
        for machine in machines:
            for task in tasks:
                all_candidates.append((machine, task))
        features = self.algorithm.extract_features(all_candidates)
        features = torch.tensor(features)
        features = features.to(dtype=torch.float32)

        with torch.no_grad():
                Qval = self.agent.critic(features)

        return Qval
    def get_new_state(self):
        waiting_tasks = self.cluster.all_tasks
        with torch.autograd.detect_anomaly(True):
            # Extract features for these waiting tasks
            job_batch_features = self.algorithm.extract_features(self.cluster, waiting_tasks)
            obervation_batch = (job_batch_features.dag, job_batch_features.resource_features)
            if len(job_batch_features.dag) > 1:
                obervation_batch = (job_batch_features.dag, job_batch_features.resource_features)
                # print(f'obsrvation batch {obervation_batch}')
                #
                # print('Here we have an observation with')
                # print(f' --------------------------------Here we we will pass it in prepare_batch {len(job_batch_features)} !! ')

            batch_dags, resources_features = prepare_batch([obervation_batch], 1)

            features = (job_batch_features.dag, job_batch_features.resource_features)

            features = (batch_dags, resources_features)
            print(f'batch_dags: {batch_dags}, resources_features: {resources_features.shape}')
            with torch.no_grad():
                _,Qval = self.agent.actor_critic(batch_dags,resources_features)

        return Qval

    def get_new_state_noembed(self):
        machines = self.cluster.machines
        tasks = self.cluster.all_tasks
        valid_candidates = []
        all_candidates = []
        # Check for valid task-machine pairs
        for machine in machines:
            for task in tasks:
                all_candidates.append((machine, task))
        features = self.algorithm.extract_features(all_candidates)
        features = torch.tensor(features)
        features = features.to(dtype=torch.float32)

        with torch.no_grad():
                _, Qval = self.agent.actor_critic(features)

        return Qval

    def log_loss(self,loss_ac,loss_critic):
        print(f'-------------------- loss actor {loss_ac} ,loss critic {loss_critic}-----------------')
        print(f'loss actor {loss_ac} ,loss critic {loss_critic}')
        wandb.log({

            "Actor loss": loss_ac,
            "Critic loss": loss_critic

        })