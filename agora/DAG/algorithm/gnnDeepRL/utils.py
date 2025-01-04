
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random

from torch_geometric.data import Data, Batch

class DAGDataset(Dataset):
    def __init__(self, observations):
        super().__init__()
        self.observations = observations

    def len(self):
        return len(self.observations)

    def get(self, idx):
        obs = self.observations[idx]
        dags = obs[0]  # Assuming obs[0] is a list of DAGs
        resource_features = obs[1]

        data_list = []
        for dag in dags:
            node_features = dag.get_node_features()
            edge_index = dag.get_adjacency_matrix().nonzero().t().contiguous()
            data_list.append(Data(x=node_features, edge_index=edge_index))

        return data_list, resource_features
from collections import defaultdict, deque


class ExperienceBuffer_group:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.groups = defaultdict(list)
        self.current_group_keys = []
        self.current_group_index = 0
        self.current_batch_index = 0

    def add(self, experience):
        # Assuming experience is a tuple where the first element is the observation
        observation = experience[0]
        obs_shape = observation.shape

        # Add experience to the buffer
        self.buffer.append(experience)

        # Group by observation shape
        self.groups[obs_shape].append(experience)

    def __iter__(self):
        # Prepare for iteration by resetting indices
        self.current_group_keys = list(self.groups.keys())
        self.current_group_index = 0
        self.current_batch_index = 0
        return self

    def __next__(self):
        while self.current_group_index < len(self.current_group_keys):
            current_key = self.current_group_keys[self.current_group_index]
            experiences = self.groups[current_key]

            start_index = self.current_batch_index * self.batch_size
            end_index = start_index + self.batch_size

            if start_index >= len(experiences):
                # Move to the next group if current is exhausted
                self.current_group_index += 1
                self.current_batch_index = 0
                continue

            # Adjust end_index if batch_size is greater than available experiences
            end_index = min(end_index, len(experiences))

            batch = experiences[start_index:end_index]
            print(f'len of batch = {len(batch)}')
            # Increment batch index for the next call within the same group
            if end_index == len(experiences):
                # Move to next group for the next call if this was the last batch of current group
                self.current_group_index += 1
                self.current_batch_index = 0
            else:
                self.current_batch_index += 1

            # Unpack and convert the batch into tensors
            observations, actions, rewards, next_observations, dones, probs, advs = zip(*batch)

            # Convert lists to tensors
            observations_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
            advs = torch.tensor(np.array(advs), dtype=torch.float32)

            return observations_tensor, actions_tensor, rewards_tensor, next_observations, dones, probs, advs

        raise StopIteration

    def clear(self):
        self.buffer.clear()
        self.groups.clear()

    def is_full(self):
        return len(self.buffer) >= self.capacity














class ExperienceBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.buffer = []
        self.batch_size = batch_size
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def get_all(self):
        observations, actions, rewards, next_observations, dones, probs, advs,valids = zip(*self.buffer)
        return observations, actions, rewards, next_observations, dones, probs, advs,valids

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def is_full(self):
        return len(self.buffer) >= self.capacity

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.buffer):
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, len(self.buffer))

        # Retrieve the batch in a sequential manner
        batch = self.buffer[self.current_index:end_index]

        # Update the current index for the next batch
        self.current_index = end_index
        #self.current_index += self.batch_size

        observations, actions, rewards, next_observations, dones, probs, advs,valids= zip(*batch)
        batch_size_=min(self.batch_size, len(self.buffer))
        # Prepare batch
        #batch_dags, batch_resource_features = prepare_batch(observations, batch_size=batch_size_,update=True)

        # Convert lists to tensors
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
        advs_tensor = torch.tensor(np.array(advs), dtype=torch.float32)

        return observations, actions_tensor, rewards_tensor, next_observations, dones, probs, advs_tensor,self.batch_size,valids

        #get data to feed to gnn
        """3/"""

import torch
torch.autograd.set_detect_anomaly(True)


    # Log the mean losses



def create_data_object(dag, batch_idx):
    """
    Creates a PyTorch Geometric Data object with batch information.

    Args:
      dag: The DAG object.
      batch_idx: The index of the batch this DAG belongs to.

    Returns:
      A PyTorch Geometric Data object.
    """
    if isinstance(dag, list):
        dag = dag[0]
    elif isinstance(dag, tuple):
        dag = dag[0][0]
    #resource_features = obs[1]

    node_features = dag.get_node_features()
    edge_index = dag.get_adjacency_matrix().nonzero().t().contiguous()
    num_nodes = node_features.size(0)

    # Create a batch tensor that assigns each node to its corresponding graph
    batch = torch.full((num_nodes,), batch_idx, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, batch=batch)
#
# class DAGDataset(Dataset):
#     def __init__(self, observations):
#         super().__init__()
#         self.observations = observations
#
#     def len(self):
#         return len(self.observations)
#
#     def get(self, idx):
#         obs = self.observations[idx]
#         dags = obs[0]
#         print(f'dags are {dags}')# Assuming obs[0] is a list of DAGs
#         resource_features = obs[1]
#
#         data_list = []
#         for dag in dags:
#             node_features = dag.get_node_features()
#             print(node_features.shape)
#             edge_index = dag.get_adjacency_matrix().nonzero().t().contiguous()
#             data_list.append(Data(x=node_features, edge_index=edge_index, batch=torch.full((node_features.size(0),), idx, dtype=torch.long)))
#
#         return data_list, resource_features


class DAGDataset(Dataset):
    def __init__(self, observations):
        super().__init__()
        self.observations = observations

    def len(self):
        return len(self.observations)

    def get(self, idx):
        obs = self.observations[idx]
        dags = obs[0]  # Assuming obs[0] is a list of DAGs
        resource_features = obs[1]  # Assuming this is a tensor or similar structure
        data_list = []
        batch_offset = 0  # Keep track of the batch offset for node indices
        pointers = []  # List to store batch indices for each node
        dag_info = []  # List to store information about each DAG
        nodes=[]

        # Ensure resource_features is structured correctly for multiple DAGs
        if isinstance(resource_features, list):
            # print('itst a list')
            # If resource_features is a list, we need to convert it to a tensor or appropriate format
            resource_features = torch.tensor(resource_features)

        num_dags = len(dags)
        num_resource_features = resource_features.size(1) if resource_features.dim() > 1 else 1

        # Create a tensor that reflects the resource features for each DAG
        extended_resource_features = torch.zeros(num_dags, 5,num_resource_features)
        # print(f'extended_resource_features shape {extended_resource_features.shape} ')

        for i, dag in enumerate(dags):
            # print(f'Processing DAG {i + 1} of {len(dags)}')
            node_features = dag.get_node_features()
            edge_index = dag.get_adjacency_matrix().nonzero().t().contiguous()
            edge_index_=edge_index.clone()
            # print(f'Edge index {edge_index}')
            # Ensure node_features is a 2D tensor of shape (num_tasks, num_features)
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(0)
                # print(f'node_features {node_features}')
            # print(f'node_features {node_features}')

            # Adjust edge_index to account for batch offset
            edge_index_ += batch_offset

            # Create pointers for each node in this DAG
            num_nodes = node_features.size(0)
            dag_pointers = [idx] * num_nodes  # This will create a list of 'idx' repeated num_nodes times
            pointers.extend(dag_pointers)
            nodes.extend(node_features)

            # Store information about this DAG
            # dag_info.append({
            #     'observation_idx': idx,
            #     'dag_idx': i,
            #     'num_nodes': num_nodes,
            #     'start_index': batch_offset,
            #     'end_index': batch_offset + num_nodes
            # })
            # Create a list of dictionaries, one for each node in the DAG
            dag_info_entries = [
                {
                    'observation_idx': idx,
                    'dag_idx': i,
                    'num_nodes': num_nodes,
                    'start_index': batch_offset,
                    'end_index': batch_offset + num_nodes
                } for _ in range(num_nodes)
            ]

            # Use extend instead of append to add all entries at once
            dag_info.extend(dag_info_entries)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                complete_pointers=torch.tensor(dag_pointers, dtype=torch.long)
            )

            # Add complete_pointers and dag_info to each Data object
            # data.pointers = torch.tensor(pointers, dtype=torch.long)
            data.dag_info = dag_info_entries.copy()  # Use copy to avoid reference issues

            data_list.append(data)
            batch_offset += num_nodes

            # Assign resource features for this DAG
            # print(f'resource_features.size(0) {resource_features.size(0)}')
            # if i < resource_features.size(0):
            #     print('yes')
            #     extended_resource_features[i] = resource_features
        # # Attach the extended resource features to the first Data object (or all if needed)
        if data_list:
            data_list[0].complete_pointers = torch.tensor(pointers, dtype=torch.long)
            data_list[0].dag_info = dag_info.copy()
            #data_list[0].x = torch.tensor(nodes)

            #to remove grad remove requires
            data_list[0].x = torch.cat([node.unsqueeze(0) for node in nodes], dim=0).requires_grad_(True)

        #     data_list[0].extended_resource_features = extended_resource_features

        return data_list, resource_features

def prepare_batch(observations_batch, batch_size=100,update=False):
    dataset = DAGDataset(observations_batch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    num_nodes_per_graph = []

    all_data_list = []
    all_resource_features = []

    for data_list, resource_features in loader:
        all_data_list.extend(data_list)
        all_resource_features.extend(resource_features)
    batch_dags = Batch.from_data_list(all_data_list)
    # print(f'len of all all_data_list {len(all_data_list)}')
    # print(f' batch_dags {batch_dags}')
    num_nodes_per_graph = [data.x.size(0) for data in all_data_list]
    # print(f'num_nodes_per_graph {len(num_nodes_per_graph)} ')
    # Get the complete pointers from the first Data object
    batch_dags.pointers = all_data_list[0].complete_pointers
    # batch_dags.x = all_data_list[0].x
    # print(f'batch_dags.x {batch_dags.x.shape}')

    # Add num_nodes_per_graph as an attribute to the batched graph
    batch_dags.num_nodes_per_graph = num_nodes_per_graph
    all_dag_info = all_data_list[0].dag_info
    batch_dags.dag_info=all_dag_info
    # print(f'dag info {batch_dags.dag_info}')

    # Resize resource features
    expanded_resource_features = []
    start_idx = 0
    total_num_nodes=sum(num_nodes_per_graph)
    # for num_nodes in num_nodes_per_graph:
    #     print(f'num nodes {num_nodes}')
    #     expanded_resource_features.append(all_resource_features[start_idx].repeat(num_nodes, 1, 1))
    #     print(f'expanded_resource_features shape {expanded_resource_features[start_idx].shape}')
    #     start_idx += 1
    expanded_resource_features.append(all_resource_features[0].repeat(total_num_nodes, 1, 1))
    print(f'expanded_resource_features shape {expanded_resource_features[start_idx].shape}')

    batch_resource_features = torch.cat(expanded_resource_features, dim=0)
    print(f'expanded_resource_features shape {batch_resource_features.shape}')

    # print(f"batch_dags.x shape: {batch_dags.x.shape}")
    # print(f"batch_dags.edge_index shape: {batch_dags.edge_index.shape}")
    # print(f"batch_resource_features shape: {batch_resource_features.shape}")
    # print(f"num_nodes_per_graph: {batch_dags.num_nodes_per_graph}")

    return batch_dags, batch_resource_features











def update_parameters_V(agent,job_chunk,iter,writer):
    print('-------------------------------- We are UPDATINGGGG--------------')
    all_observations, all_actions, all_rewards, all_next_observations, all_dones,all_logits,all_advantages = agent.experience_buffer.get_all()

    combined_data = list(zip(all_observations, all_actions, all_rewards, all_next_observations, all_dones,all_logits,all_advantages))
    batch_size = 1000  # Define the batch size
    loss_values = []
    critic_loss_values = []
    agent.critic_optimizer.zero_grad()
    agent.actor_optimizer.zero_grad()

    for i in range(0, len(combined_data), batch_size):
        batch_data = combined_data[i:i + batch_size]
        actions_batch =  torch.tensor([data[1] for data in batch_data], dtype=torch.long)

        logits =[data[5] for data in batch_data]

        advantages=torch.tensor([data[6] for data in batch_data], dtype=torch.float32)
        batch_size=len(logits)



        # Zero gradients

        #TO DO GET FROM THE BATCH_DAGS ONLY THE ELEMNTS OF THE OBSERVATION


        # Update actor
        log_probs_list = logits

        selected_log_probs = []
        selected_probs = []
        for i, probs in enumerate(log_probs_list):
            probs_ = probs.view(-1)
            selected_prob = probs_.gather(0, actions_batch[i].long().unsqueeze(0)).squeeze()
            selected_probs.append(selected_prob)

        selected_log_probs = torch.stack(selected_probs)


        actor_loss = -(torch.log(selected_log_probs + 1e-8) * advantages).mean()
        # print(f'actor_loss = {actor_loss}')

        actor_loss = torch.tensor(actor_loss, requires_grad=True)  # Or calculate it using requires_grad=True

        # print(f'actor_loss = {actor_loss}')

        actor_loss.backward()

        # Append losses
        loss_values.append(actor_loss.item())

        # Gradient clipping and parameter update
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)

        agent.actor_optimizer.step()

    # Clear buffer after processing all data
    agent.experience_buffer.clear()
    agent.global_step += 1
    mean_loss = np.mean(loss_values)
    agent.log(f'{job_chunk}/loss', mean_loss, agent.global_step)
    # Add markers for job chunk and iteration
    if job_chunk is not None:
        writer.add_scalar('Job_Chunk_Start', mean_loss, agent.global_step)
        writer.add_text('Job_Chunk', f'Start of Job Chunk {job_chunk}', agent.global_step)

    writer.add_scalar('Iteration', mean_loss, agent.global_step)
    writer.add_text('Iteration', f'Iteration {iter}', agent.global_step)
    return mean_loss


def compute_gradient_norms(model):
    total_norm = 0
    parameter_norms = {}
    for name, param in model.named_parameters():

        if param.grad is None:
            print(f"Parameter {name} has None gradient")

        if param.grad is not None:
            print(f"Parameter {name} gradient: {param.grad.mean()}")

            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            parameter_norms[name] = param_norm.item()
    total_norm = total_norm ** 0.5
    return total_norm, parameter_norms