from distutils.core import setup_keywords

import torch
import torch.nn as nn
import torch
device = torch.device("cpu")

from torch_geometric.nn import GATConv, global_mean_pool

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
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

set_seed(1)


# class GraphCNN(nn.Module):
#     def __init__(self, node_feature_dim, resource_feature_dim, hidden_dim, num_heads=2, dropout=0.5):
#         super(GraphCNN, self).__init__()


class Actor_noembed(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(Actor_noembed, self).__init__()
        #self.gnn = gnn.to(device)
        # self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
        self.embeddings = 0

        # Linear layers followed by Batch Normalization
        self.layer1 = nn.Linear(9, 3).to(device)
        self.bn1 = nn.BatchNorm1d(3).to(device)  # BatchNorm for layer1

        self.layer2 = nn.Linear(3, 9).to(device)
        self.bn2 = nn.BatchNorm1d(9).to(device)  # BatchNorm for layer2

        self.layer3 = nn.Linear(9, 18).to(device)
        self.bn3 = nn.BatchNorm1d(18).to(device)  # BatchNorm for layer3

        self.layer4 = nn.Linear(18, 9).to(device)
        self.bn4 = nn.BatchNorm1d(9).to(device)   # BatchNorm for layer4

        self.layer5 = nn.Linear(9, 1).to(device)
        self.bn5 = nn.BatchNorm1d(1).to(device)   # BatchNorm for layer4



        # self.init_actor_weights()  # For Actor class


    def forward(self, features):
        # print('-------------------------------- Actor working--------------')
        self.embeddings=features.to(device)

        # print(f'self.embeddings shape{self.embeddings.shape} ')

        # Reshape embeddings to (batch_size * num_tasks * num_resources, combined_feature_dim)
        flat_embeddings = self.embeddings.view(-1, self.embeddings.size(-1))

        # print("Reshaped tensor requires_grad:", flat_embeddings.requires_grad)  # True

        # Pass through the layers with Batch Normalization
        state = F.leaky_relu(self.bn1(self.layer1(flat_embeddings)))
        state = F.leaky_relu(self.bn2(self.layer2(state)))
        state = F.leaky_relu(self.bn3(self.layer3(state)))
        state = F.leaky_relu(self.bn4(self.layer4(state)))
        # print(self.layer5(state).squeeze(-1))
        scores = self.layer5(state)
        scores=scores.squeeze(-1)
        # print("scores tensor requires_grad:", scores.requires_grad)  # True
        # print(scores)
        # print(f' after normalization : {self.bn5(self.layer5(state)).squeeze(-1)}')

        # print(scores.shape)
        # print('-------------------------------- Actor Done--------------')

        return scores




class Critic_noembed(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(Critic_noembed, self).__init__()
        #self.gnn = gnn.to(device)
        # self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
        self.embeddings = 0


        # Critic network (separate branch)
        self.critic_layer1 = nn.Linear(9, 64).to(device)
        self.critic_layer2 = nn.Linear(64, 32).to(device)
        self.critic_output = nn.Linear(32, 1).to(device)  # Outputs a single value


        # self.init_actor_weights()  # For Actor class


    def forward(self, features):
        # print('-------------------------------- Critic working--------------')
        self.embeddings=features.to(device)

        # print(f'self.embeddings shape{self.embeddings.shape} ')

        # Reshape embeddings to (batch_size * num_tasks * num_resources, combined_feature_dim)
        flat_embeddings = self.embeddings.view(-1, self.embeddings.size(-1))
        # print("Reshaped tensor requires_grad:", flat_embeddings.requires_grad)  # True


        pooled_embeddings = torch.mean(flat_embeddings, dim=0, keepdim=True)  # Example: Mean pooling

        critic_state = F.relu(self.critic_layer1(pooled_embeddings))
        critic_state = F.relu(self.critic_layer2(critic_state))
        state_value = self.critic_output(critic_state).squeeze(-1)  # Single scalar value for the state
        # print("state_value tensor requires_grad:", state_value.requires_grad)  # True

        # print('-------------------------------- Critic Done--------------')

        return state_value















class ActorCritic_noembed(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(ActorCritic_noembed, self).__init__()
        #self.gnn = gnn.to(device)
        # self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
        self.embeddings = 0

        # Linear layers followed by Batch Normalization
        self.layer1 = nn.Linear(9, 3).to(device)
        self.bn1 = nn.BatchNorm1d(3).to(device)  # BatchNorm for layer1

        self.layer2 = nn.Linear(3, 9).to(device)
        self.bn2 = nn.BatchNorm1d(9).to(device)  # BatchNorm for layer2

        self.layer3 = nn.Linear(9, 18).to(device)
        self.bn3 = nn.BatchNorm1d(18).to(device)  # BatchNorm for layer3

        self.layer4 = nn.Linear(18, 9).to(device)
        self.bn4 = nn.BatchNorm1d(9).to(device)   # BatchNorm for layer4

        self.layer5 = nn.Linear(9, 1).to(device)



        # Critic network (separate branch)
        self.critic_layer1 = nn.Linear(9, 64).to(device)
        self.critic_layer2 = nn.Linear(64, 32).to(device)
        self.critic_output = nn.Linear(32, 1).to(device)  # Outputs a single value


        # self.init_actor_weights()  # For Actor class


    def forward(self, features):
        # print('-------------------------------- Actor working--------------')
        self.embeddings=features

        print(f'self.embeddings shape{self.embeddings.shape} ')

        # Reshape embeddings to (batch_size * num_tasks * num_resources, combined_feature_dim)
        flat_embeddings = self.embeddings.view(-1, self.embeddings.size(-1))
        print("Reshaped tensor requires_grad:", flat_embeddings.requires_grad)  # True

        # Pass through the layers with Batch Normalization
        state = F.leaky_relu(self.bn1(self.layer1(flat_embeddings)))
        state = F.leaky_relu(self.bn2(self.layer2(state)))
        state = F.leaky_relu(self.bn3(self.layer3(state)))
        state = F.leaky_relu(self.bn4(self.layer4(state)))
        scores = self.layer5(state).squeeze(-1)

        # Apply softmax over all task-machine pairs for each batch
        probabilities = F.softmax(scores.view(scores.size(0), -1), dim=1).view(scores.size())
        pooled_embeddings = torch.mean(flat_embeddings, dim=0, keepdim=True)  # Example: Mean pooling

        critic_state = F.relu(self.critic_layer1(pooled_embeddings))
        critic_state = F.relu(self.critic_layer2(critic_state))
        state_value = self.critic_output(critic_state).squeeze(-1)  # Single scalar value for the state

        print(scores.shape)
        print('-------------------------------- Actor Done--------------')

        return scores,state_value







class ActorCritic(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(ActorCritic, self).__init__()
        self.gnn = gnn.to(device)
        # self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
        self.embeddings = 0

        # Linear layers followed by Batch Normalization
        self.layer1 = nn.Linear(2 * hidden_dim, 3).to(device)
        self.bn1 = nn.BatchNorm1d(3).to(device)  # BatchNorm for layer1

        self.layer2 = nn.Linear(3, 9).to(device)
        self.bn2 = nn.BatchNorm1d(9).to(device)  # BatchNorm for layer2

        self.layer3 = nn.Linear(9, 18).to(device)
        self.bn3 = nn.BatchNorm1d(18).to(device)  # BatchNorm for layer3

        self.layer4 = nn.Linear(18, 9).to(device)
        self.bn4 = nn.BatchNorm1d(9).to(device)   # BatchNorm for layer4

        self.layer5 = nn.Linear(9, 1).to(device)



        # Critic network (separate branch)
        self.critic_layer1 = nn.Linear(2 * hidden_dim, 64).to(device)
        self.critic_layer2 = nn.Linear(64, 32).to(device)
        self.critic_output = nn.Linear(32, 1).to(device)  # Outputs a single value


        # self.init_actor_weights()  # For Actor class


    def init_actor_weights(self):
        # Initialize linear layers with He initialization (for ReLU)
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer4.weight, nonlinearity='leaky_relu')

        # Last layer with Xavier (output doesn't use ReLU)
        nn.init.xavier_uniform_(self.layer5.weight)

        # Initialize batch norm layers
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)
        nn.init.ones_(self.bn3.weight)
        nn.init.zeros_(self.bn3.bias)
        nn.init.ones_(self.bn4.weight)
        nn.init.zeros_(self.bn4.bias)
    def forward(self, batch_dags, resource_features=None, update=False):
        # print('-------------------------------- Actor working--------------')

        if update:
            self.embeddings = batch_dags
        else:
            self.embeddings = self.gnn(batch_dags, resource_features)

        print(f'self.embeddings shape{self.embeddings.shape} ')

        # Reshape embeddings to (batch_size * num_tasks * num_resources, combined_feature_dim)
        flat_embeddings = self.embeddings.view(-1, self.embeddings.size(-1))
        print("Reshaped tensor requires_grad:", flat_embeddings.requires_grad)  # True

        # Pass through the layers with Batch Normalization
        state = F.leaky_relu(self.bn1(self.layer1(flat_embeddings)))
        state = F.leaky_relu(self.bn2(self.layer2(state)))
        state = F.leaky_relu(self.bn3(self.layer3(state)))
        state = F.leaky_relu(self.bn4(self.layer4(state)))
        scores = self.layer5(state).squeeze(-1)

        # Apply softmax over all task-machine pairs for each batch
        probabilities = F.softmax(scores.view(scores.size(0), -1), dim=1).view(scores.size())
        pooled_embeddings = torch.mean(flat_embeddings, dim=0, keepdim=True)  # Example: Mean pooling

        critic_state = F.relu(self.critic_layer1(pooled_embeddings))
        critic_state = F.relu(self.critic_layer2(critic_state))
        state_value = self.critic_output(critic_state).squeeze(-1)  # Single scalar value for the state

        print(scores.shape)
        # print('-------------------------------- Actor Done--------------')

        return scores,state_value

    def get_embeddings(self):
        return self.embeddings






















def separate_embeddings(embeddings, num_nodes_per_graph):
    separated_embeddings = []
    start_idx = 0
    for num_nodes in num_nodes_per_graph:
        end_idx = start_idx + num_nodes
        separated_embeddings.append(embeddings[start_idx:end_idx])
        start_idx = end_idx
    return separated_embeddings





class CloudResourceGNNv1(nn.Module):

    def __init__(self, node_feature_dim, resource_feature_dim, hidden_dim, num_heads=2, dropout=0.5):
        super(CloudResourceGNNv1, self).__init__()

        self.task_conv1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.task_conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.resource_linear = nn.Linear(resource_feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.embeddings=0
        # self.init_gnn_weights()  # For GNN class

        # for param in self.parameters():
        #     param.requires_grad = True

    def init_gnn_weights(self):
        # GAT layers
        nn.init.xavier_uniform_(self.task_conv1.lin.weight)
        nn.init.xavier_uniform_(self.task_conv2.lin.weight)

        # Attention weights
        nn.init.xavier_uniform_(self.task_conv1.att_src)
        nn.init.xavier_uniform_(self.task_conv1.att_dst)
        nn.init.xavier_uniform_(self.task_conv2.att_src)
        nn.init.xavier_uniform_(self.task_conv2.att_dst)

        # Linear layer
        nn.init.xavier_uniform_(self.resource_linear.weight)

        # Layer norm
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
    def forward(self, batch_dags, resource_features):
        # Process task features through GAT layers
        x = F.elu(self.dropout(self.task_conv1(batch_dags.x, batch_dags.edge_index.to(device))))
        x = self.task_conv2(x, batch_dags.edge_index.to(device))
        x = self.layer_norm(x)
        # x_expanded = x.unsqueeze(1).repeat(1, resource_features.size(1), 1).to(device)
        x = x.to(device)# Ensure x is on the correct device
        print("x:", x.requires_grad)

        x_expanded = x.unsqueeze(1).repeat(1, resource_features.size(1), 1)  # Do not call .to(device) again
        print("x_expanded requires_grad (after repeat):", x_expanded.requires_grad)

        batch_resource_features = resource_features.to(device)
        # Process resource features through a linear layer
        resource_embedding = F.elu(self.dropout(self.resource_linear(batch_resource_features)))

        # Pooling to get graph-level embeddings
        pooled_x = global_mean_pool(x, batch_dags.batch)

        # Reshape pooled_x to (batch_size, num_tasks, task_feature_dim)
        batch_size = batch_dags.num_graphs
        num_tasks = pooled_x.size(0) // batch_size
        task_embeddings = pooled_x.view(batch_size, num_tasks, -1)

        # Reshape resource_embedding to (batch_size, num_resources, resource_feature_dim)
        num_resources = resource_embedding.size(0) // batch_size
        # resource_embeddings = resource_embedding.view(1, num_resources, -1)
        print(f'task_embeddings shape {task_embeddings.shape}')
        print(f'x_expanded shape {x_expanded.shape}')

        print(f'resource_embedding shape {resource_embedding.shape}')

        # Combine embeddings
        combined_features = torch.cat((x_expanded, resource_embedding), dim=-1)  # Shape: [3, 5, 64]
        self.embeddings = combined_features
        print("combined_features tensor requires_grad:", combined_features.requires_grad)  # True
        for param in self.parameters():
            print(param.requires_grad)  # Should be True for all parameters

        #print(f'self.embeddings shape {self.embeddings.shape}')
        return combined_features

    def get_embeddings(self):
        return self.embeddings


    def combine_embeddings(self, resource_embeddings, task_embeddings):
        """
           Combines resource and task embeddings to create a feature map for each
           task-resource pair.

           Args:
             resource_embeddings: Tensor of shape (batch_size, num_resources, resource_feature_dim).
             task_embeddings: Tensor of shape (batch_size, num_tasks, task_feature_dim).

           Returns:
             A tensor of shape (batch_size, num_tasks, num_resources, combined_feature_dim)
             representing the combined features for each task-resource pair.
           """
        batch_size, num_tasks, task_feature_dim = task_embeddings.shape
        _, num_resources, resource_feature_dim = resource_embeddings.shape

        # # Expand dimensions for broadcasting:
        # # Expand resource embeddings to (batch_size, num_tasks, num_resources, resource_feature_dim)
        # expanded_resource_embeddings = resource_embeddings.unsqueeze(1).expand(-1, num_tasks, -1, -1)
        # print(f'expanded_resource_embeddings shape {expanded_resource_embeddings.shape}')
        # # Expand task embeddings to (batch_size, num_tasks, num_resources, task_feature_dim)
        # expanded_task_embeddings = task_embeddings.unsqueeze(2).expand(-1, -1, num_resources, -1)
        # print(f'expanded_task_embeddings shape {expanded_task_embeddings.shape}')

        # Concatenate along the feature dimension
        combined_embeddings = torch.cat((task_embeddings, resource_embeddings), dim=-1)
        print(f'combined_embeddings shape {combined_embeddings.shape}')
        print("combined_embeddings tensor requires_grad:", combined_embeddings.requires_grad)  # True

        return combined_embeddings



import torch.nn as nn
import torch.nn.functional as F

class Actorv1(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(Actorv1, self).__init__()
        self.gnn = gnn.to(device)
        # self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
        self.embeddings = 0

        # Linear layers followed by Batch Normalization
        self.layer1 = nn.Linear(2 * hidden_dim, 3).to(device)
        self.bn1 = nn.BatchNorm1d(3).to(device)  # BatchNorm for layer1

        self.layer2 = nn.Linear(3, 9).to(device)
        self.bn2 = nn.BatchNorm1d(9).to(device)  # BatchNorm for layer2

        self.layer3 = nn.Linear(9, 18).to(device)
        self.bn3 = nn.BatchNorm1d(18).to(device)  # BatchNorm for layer3

        self.layer4 = nn.Linear(18, 9).to(device)
        self.bn4 = nn.BatchNorm1d(9).to(device)   # BatchNorm for layer4

        self.layer5 = nn.Linear(9, 1).to(device)
        # self.init_actor_weights()  # For Actor class


    def init_actor_weights(self):
        # Initialize linear layers with He initialization (for ReLU)
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.layer4.weight, nonlinearity='leaky_relu')

        # Last layer with Xavier (output doesn't use ReLU)
        nn.init.xavier_uniform_(self.layer5.weight)

        # Initialize batch norm layers
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)
        nn.init.ones_(self.bn3.weight)
        nn.init.zeros_(self.bn3.bias)
        nn.init.ones_(self.bn4.weight)
        nn.init.zeros_(self.bn4.bias)
    def forward(self, batch_dags, resource_features=None, update=False):
        # print('-------------------------------- Actor working--------------')

        if update:
            self.embeddings = batch_dags
        else:
            self.embeddings = self.gnn(batch_dags, resource_features)

        print(f'self.embeddings shape{self.embeddings.shape} ')

        # Reshape embeddings to (batch_size * num_tasks * num_resources, combined_feature_dim)
        flat_embeddings = self.embeddings.view(-1, self.embeddings.size(-1))
        print("Reshaped tensor requires_grad:", flat_embeddings.requires_grad)  # True

        # Pass through the layers with Batch Normalization
        state = F.leaky_relu(self.bn1(self.layer1(flat_embeddings)))
        state = F.leaky_relu(self.bn2(self.layer2(state)))
        state = F.leaky_relu(self.bn3(self.layer3(state)))
        state = F.leaky_relu(self.bn4(self.layer4(state)))
        scores = self.layer5(state).squeeze(-1)

        # Apply softmax over all task-machine pairs for each batch
        probabilities = F.softmax(scores.view(scores.size(0), -1), dim=1).view(scores.size())

        print(scores.shape)
        print('-------------------------------- Actor Done--------------')

        return scores

    def get_embeddings(self):
        return self.embeddings




class CloudResourceGNN(nn.Module):

    def __init__(self, node_feature_dim, resource_feature_dim, hidden_dim, num_heads=2, dropout=0.5):
        super(CloudResourceGNN, self).__init__()
        self.task_conv1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.task_conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.resource_linear = nn.Linear(resource_feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.current_embeddings=0
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, batch_dags,resource_features):
        # Process task features through GAT layers
        #print(f'Batch dags for task {batch_dags}')
        with torch.set_grad_enabled(False):

            self.pointers=batch_dags.pointers
            # batch_dags.x.requires_grad_(True)

            x = F.elu(self.dropout(self.task_conv1(batch_dags.x, batch_dags.edge_index.to(device))))

            x = self.task_conv2(x, batch_dags.edge_index.to(device))
            x = self.layer_norm(x)
            batch_resource_features=resource_features.to(device)
            # Process resource features through a linear layer
            resource_embedding = F.elu(self.dropout(self.resource_linear(batch_resource_features)))


            is_same = torch.all(batch_dags.batch == batch_dags.batch[0])
            if is_same:
                # If all tasks have the same number of nodes, combine their embeddings
                combined = self.combine_embeddings(resource_embedding,x)
                self.current_embeddings = combined

                #self.embeddings = combined.clone()
                #print(f'self.embeddings shape {self.embeddings.shape}')
                return combined

            # Pooling to get graph-level embeddings
            #pooled_x = global_mean_pool(x, batch_dags.batch)

            # print(f'batch_dags.batch {batch_dags.batch}')
            # print(f'resource_embedding shape is {resource_embedding.shape}')
            task_embeddings = x
            # print(f'pooled_x shape is {pooled_x.shape}')

            # Adjust embeddings to match each task with resources
            combined = self.combine_embeddings(resource_embedding,x)
            self.current_embeddings = combined
            #print(f'self.embeddings differeeeenttt shape {self.embeddings.shape}')
            # print(f'combined embedding shape is {combined.shape}')
        return combined

    def get_embeddings(self):
        return self.embeddings



    def get_separated_embeddings(self):
        if self.embeddings is None or self.num_nodes_per_graph is None:
            return None
        return separate_embeddings_ptr(self.embeddings, self.num_nodes_per_graph)

    def combine_embeddings(self,resource_embeddings, task_embeddings):
        """Combines resource and task embeddings to create a feature map for each
        task-resource pair.

        Args:
          resource_embeddings: Tensor of shape (num_resources, resource_feature_dim).
          task_embeddings: Tensor of shape (num_tasks, task_feature_dim).

        Returns:
          A tensor of shape (num_tasks, num_resources, combined_feature_dim)
          representing the combined features for each task-resource pair.
        """

        resource_embeddings = resource_embeddings.to(device)
        task_embeddings = task_embeddings.to(device)


        num_tasks = task_embeddings.shape[0]
        num_resources = resource_embeddings.shape[0]
        # print(f'task_embeddings shape is {task_embeddings.shape}')
        # print(f'num_resources shape is {resource_embeddings.shape}')

        # 1. Expand dimensions for broadcasting:
        # Expand resource embeddings to (num_tasks, num_resources, resource_feature_dim)
        if len(resource_embeddings.shape)<3 :

            expanded_resource_embeddings = resource_embeddings.unsqueeze(0).expand(num_tasks, -1, -1)
            # print(f'expanded_resource_embeddings shape is {expanded_resource_embeddings.shape}')

            # Expand task embeddings to (num_tasks, num_resources, task_feature_dim)
            expanded_task_embeddings = task_embeddings.unsqueeze(1).expand(-1, num_resources, -1)
            # print(f'expanded_task_embeddings shape is {expanded_task_embeddings.shape}')
            combined_embeddings = torch.cat((expanded_task_embeddings, expanded_resource_embeddings),
                                            dim=-1)  # Changed order here

        else:
            list_of_embeddings =[]
            # 2. Concatenate along the feature dimension:
            expanded_task_embeddings = task_embeddings.unsqueeze(1).expand(-1, resource_embeddings.shape[1], -1)
            # print(f'expanded_task_embeddings shape is {expanded_task_embeddings.shape}')
            if expanded_task_embeddings.shape[0]!=1:
                resource_embeddings=resource_embeddings.expand(expanded_task_embeddings.shape[0], -1, -1)
                # print(f'resource_embeddings shape is {resource_embeddings.shape}')

            combined_embeddings = torch.cat((expanded_task_embeddings, resource_embeddings),
                                                dim=-1)# Changed order here
        # combined_embeddings.requires_grad_(True)

        return combined_embeddings


def separate_embeddings_ptr(embeddings, pointers,dag_info):
    """
    Separates the embeddings into a list of tensors for each element of the batch.

    Args:
        embeddings (torch.Tensor): The embeddings for the entire batch with shape (num_nodes, ..., feature_dim).
        pointers (torch.Tensor): A tensor indicating the index of each node in the original graphs.

    Returns:
        List[torch.Tensor]: A list where each element is a tensor of embeddings for one graph in the batch.
    """
    # Get unique graph indices from pointers
    unique_graph_indices = torch.unique(pointers)
    # Flatten the nested list of dictionaries in dag_info
    flattened_dag_info = [info for sublist in dag_info for info in sublist]

    # Now flattened_dag_info contains all dictionaries in a single list
    # print(len(flattened_dag_info))
    # print('len(unique_graph_indices)',len(unique_graph_indices))
    observation_indices = torch.tensor([info['observation_idx'] for info in flattened_dag_info])
    # print(observation_indices)
    # print(pointers)
    # Initialize a list to store separated embeddings
    separated_embeddings = []
    # Iterate over each unique graph index
    for graph_index in unique_graph_indices:
        # Find nodes that belong to the current graph
        node_indices = (observation_indices == graph_index).nonzero(as_tuple=True)[0]
        # Extract embeddings for these nodes
        graph_embeddings = embeddings[node_indices]

        # Append to the list
        separated_embeddings.append(graph_embeddings)

    return separated_embeddings


def pad_and_mask_tensors(tensor_list):
    # Find maximum size in first dimension
    max_size = max([tensor.size(0) for tensor in tensor_list])
    batch_size = len(tensor_list)
    feature_dim = tensor_list[0].size(-1)  # Assuming all tensors have same feature dimension
    middle_dim = tensor_list[0].size(1)  # Assuming all tensors have same middle dimension

    # Initialize padded tensor and mask
    padded_tensor = torch.zeros((batch_size, max_size, middle_dim, feature_dim), device=tensor_list[0].device)
    mask = torch.zeros((batch_size, max_size), dtype=torch.bool, device=tensor_list[0].device)

    # Fill padded tensor and create mask
    for i, tensor in enumerate(tensor_list):
        actual_size = tensor.size(0)
        padded_tensor[i, :actual_size] = tensor
        mask[i, :actual_size] = True

    return padded_tensor, mask



def unpad_tensor(padded_output, mask):
    """
    Separates the padded tensor into a list of tensors, preserving gradients.

    Args:
        padded_output (torch.Tensor): The padded tensor.
        mask (torch.Tensor): The mask indicating valid elements.

    Returns:
        List[torch.Tensor]: A list of tensors, each representing a single unpadded tensor.
    """

    unpadded_list = []
    for i in range(padded_output.size(0)):
        real_elements = padded_output[i][mask[i]]
        unpadded_list.append(real_elements)

    return unpadded_list
#
# class Actorv1(nn.Module):
#     def __init__(self, gnn, hidden_dim):
#         super(Actorv1, self).__init__()
#         self.gnn = gnn.to(device)
#
#
#
#         self.task_machine_head = nn.Linear(2 * hidden_dim, 1).to(device)
#         # Ensure all parameters require gradients
#         for param in self.gnn.parameters():
#             param.requires_grad = True
#
#         # Initialize embeddings as a parameter
#         # self.register_buffer('embeddings', torch.zeros(1))
#
#     def forward(self, batch_dags,resource_features=None,update=False):
#         # print('-------------------------------- Actor working--------------')
#         # print(f'batch_dags {batch_dags}')
#
#         if update:
#             batch_dags.x.requires_grad_(True)
#             # print(f'edge_index {batch_dags.edge_index}')
#
#             embeddings = self.gnn(batch_dags, resource_features)
#             assert embeddings.requires_grad, "GNN output doesn't require gradients"
#             new_embeddings=separate_embeddings_ptr(embeddings, batch_dags.pointers,batch_dags.dag_info)
#             # print(f'new_embeddings {new_embeddings.shape}')
#
#             padded_tensor, mask=pad_and_mask_tensors(new_embeddings)
#
#             assert padded_tensor.requires_grad, "padded_tensor doesn't require gradients"
#
#             # print(f'padded_tensor {padded_tensor.shape}')
#
#
#             scores = self.task_machine_head(padded_tensor).squeeze(-1)
#             list_scores=unpad_tensor(scores,mask)
#             list_scores_=[t.view(-1) for t in list_scores]
#             # print(f'scores {scores.shape}')
#
#             #probabilities=[F.softmax(scores.view(-1), dim=0) for scores in list_scores]
#             return padded_tensor,mask
#         with torch.set_grad_enabled(False):
#             embeddings = self.gnn(batch_dags,resource_features)
#             self.embeddings = embeddings
#             scores = self.task_machine_head(self.embeddings).squeeze(-1).view(-1)
#         # Apply softmax over all task-machine pairs
#         #probabilities = F.softmax(scores.view(-1), dim=0).view(scores.size())
#             # print('-------------------------------- Actor Done--------------')
#
#             return scores



class Actor(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(Actor, self).__init__()
        self.gnn = gnn
        self.layer2 = nn.Linear(2 * hidden_dim, 32).to(device)
        self.layer3 = nn.Linear(32, 16).to(device)
        self.layer4 = nn.Linear(16, 8).to(device)
        self.elu = nn.ELU()
        self.embeddings=0

        self.task_machine_head = nn.Linear(8, 1).to(device)
        # Ensure all parameters require gradients
        for param in self.gnn.parameters():
            param.requires_grad = False

        # Initialize embeddings as a parameter
        # self.register_buffer('embeddings', torch.zeros(1))

    def forward(self, batch_dags,resource_features=None,update=False):
        # print('-------------------------------- Actor working--------------')
        # print(f'batch_dags {batch_dags}')

        if update:
            #batch_dags.x.requires_grad_(True)
            # print(f'edge_index {batch_dags.edge_index}')

            embeddings = self.gnn(batch_dags,resource_features)
            #assert embeddings.requires_grad, "GNN output doesn't require gradients"
            print(f'embeddings shape {embeddings.shape}')
            new_embeddings=separate_embeddings_ptr(embeddings, batch_dags.pointers,batch_dags.dag_info)
            self.embeddings=new_embeddings
            # print(f'new_embeddings {new_embeddings.shape}')

            padded_tensor, mask=pad_and_mask_tensors(new_embeddings)

            #assert padded_tensor.requires_grad, "padded_tensor doesn't require gradients"

            # print(f'padded_tensor {padded_tensor.shape}')

            x = self.elu(self.layer2(padded_tensor))
            x = self.elu(self.layer3(x))
            x = self.elu(self.layer4(x))
            scores = self.task_machine_head(x).squeeze(-1)

            #scores = self.task_machine_head(padded_tensor).squeeze(-1)
            list_scores=unpad_tensor(scores,mask)
            list_scores_=[t.view(-1) for t in list_scores]
            # print(f'scores {scores.shape}')

            #probabilities=[F.softmax(scores.view(-1), dim=0) for scores in list_scores]
            return list_scores_
        # with torch.set_grad_enabled(False):
        embeddings = self.gnn(batch_dags,resource_features)
        self.embeddings = embeddings
        # print(f'self . embeddings shape {self.embeddings.shape}')
        x = self.elu(self.layer2(embeddings))
        x = self.elu(self.layer3(x))
        x = self.elu(self.layer4(x))

        scores = self.task_machine_head(x).squeeze(-1).view(-1)
    # Apply softmax over all task-machine pairs
    #probabilities = F.softmax(scores.view(-1), dim=0).view(scores.size())
        # print('-------------------------------- Actor Done--------------')

        return scores

    def get_embeddings(self):
        return self.embeddings



class Critic(nn.Module):
    def __init__(self, gnn, hidden_dim):
        super(Critic, self).__init__()
        self.gnn = gnn.to(device)
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim).to(device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1).to(device)
        )

    def forward(self, batch_dags, resource_features, batch_size):
        embeddings = self.gnn(batch_dags, resource_features).clone()
        print(embeddings.shape)

        # Assuming embeddings has shape [num_total_nodes, num_resources, features]
        # Reshape to [batch_size, num_nodes_per_graph, num_resources, features]
        num_nodes_per_graph = embeddings.shape[0] // batch_size
        embeddings = embeddings.view(batch_size, num_nodes_per_graph, -1, embeddings.shape[2])

        # Pool across nodes and resources
        pooled_embeddings = torch.mean(embeddings, dim=(1, 2))  # Shape: [batch_size, features]

        output = self.value_head(pooled_embeddings).clone().squeeze(-1)

        return output



#
# class Actor(nn.Module):
#     def __init__(self, gnn,node_feature_dim, resource_feature_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.gnn =gnn
#         self.task_machine_head = nn.Linear(2*hidden_dim, 1)
#
#     def forward(self, grouped_embeddings):
#         #batch_size = len(dags)
#         #num_machines = resource_features.size(0)
#
#         # Get embeddings for all tasks across all DAGs
#         #grouped_embeddings = self.gnn(dags, resource_features)
#         # Compute scores for all task-machine pairs
#         scores = self.task_machine_head(grouped_embeddings).squeeze(-1)
#         # Apply softmax over all task-machine pairs
#         probabilities = F.softmax(scores.view(-1), dim=0).view(scores.size())
#         return probabilities
#
#
#
#
#
#
# class Critic(nn.Module):
#     def __init__(self, gnn,node_feature_dim, resource_feature_dim, hidden_dim):
#         super(Critic, self).__init__()
#         self.gnn = gnn
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, features):
#         #batch_size = len(dags)
#         batch_size=len(features)
#         features = self.gnn(dags, resource_features)
#         # Reshape features to match all possible task-resource combinations
#         features = features.view(batch_size, -1, features.size(-1))  # [batch_size, num_tasks, hidden_dim]
#
#         return self.value_head(features)


def check_items_not_none(states, actions, rewards):
    return all(state is not None and state != -1 for state in states) and \
           all(action is not None and action != -1 for action in actions) and \
           all(reward is not None and reward != -1 for reward in rewards)

def train_gnn_ac(model, optimizer, trajectories, gamma=0.99):
    model.train()
    total_loss = 0

    for trajectory in trajectories:
        trajectory_tuples = [node.to_tuple() for node in trajectory]

        states, actions, rewards = zip(*trajectory_tuples)
        if check_items_not_none(states, actions, rewards):


            # Convert to tensors
            state_tensor = torch.stack([dag.get_node_features() for dag in states])
            action_tensor = torch.tensor(actions)
            reward_tensor = torch.tensor(rewards)

            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)

            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Forward pass
            action_probs, state_values = model(state_tensor)

            # Compute losses
            advantage = returns - state_values.squeeze()
            actor_loss = -(
                        torch.log(action_probs.gather(1, action_tensor.unsqueeze(1))).squeeze() * advantage.detach()).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            entropy_loss = -(action_probs * torch.log(action_probs)).sum(dim=1).mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            print("Skipping trajectory with None or -1 values")
            continue

    return total_loss / len(trajectories)