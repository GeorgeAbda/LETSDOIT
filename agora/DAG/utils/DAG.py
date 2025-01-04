import torch
from typing import Dict, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from agora.DAG.utils.feature_synthesize import task_features
import os
from agora.DAG.utils.feature_functions import *

class DAG:
    def __init__(self, job, resources):
        self._nodes: List[str] = []
        self._edges: List[Tuple[str, str]] = []
        self._node_features: Dict[str, Dict] = {}
        self._resource_features: Dict[str, List[float]] = resources
        self._tasks = job.tasks
        self.dag_object= nx.DiGraph()
        self._build_dag(job)


    def __str__(self):

        print(f'Nodes of the DAG: {self._nodes}')


    def _build_dag(self, job):
        self._tasks = job.tasks
        # job.update_tasks_map()

        # Extract features using both methods
        feature_dicts = task_features(job)

        for task in self._tasks:
            task_index, parent_indices = task.task_index, task.task_config.parent_indices
            parent_indices = [parent for parent in parent_indices if
                              parent in [t.task_config.task_index for t in self._tasks]]
            self._nodes.append(task_index)
            for parent_index in parent_indices:
                if parent_index in [t.task_config.task_index for t in self._tasks]:
                    self._edges.append((parent_index, task_index))

            # Combine features from both sources
            extracted_features = features_extract_func_ac(task)
            additional_features = feature_dicts.get(task_index, {})
            # print(f' extracted_features {extracted_features}')
            # print(f' additional_features {additional_features}')

            # Add all features to node features
            #self._node_features[task_index] = extracted_features + list(additional_features.values())

            self._node_features[task_index] = extracted_features

        self.dag_object.add_nodes_from(self._nodes)
        self.dag_object.add_edges_from(self._edges)

        self._job = job



    def get_node_features(self) -> torch.Tensor:
        feature_list = [self._node_features[node] for node in self._nodes]
        return torch.tensor(feature_list, dtype=torch.float32)

    @property
    def tasks(self):
        return self.job.tasks

    @property
    def job(self):
        return self._job

    def __len__(self):
        return len(self.tasks)




    def get_adjacency_matrix(self) -> torch.Tensor:
        num_nodes = len(self._nodes)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        node_to_id = {node: i for i, node in enumerate(self._nodes)}
        for edge in self._edges:
            source_id, target_id = node_to_id[edge[0]], node_to_id[edge[1]]
            adj_matrix[source_id, target_id] = 1
        return adj_matrix

    def get_node_features(self) -> torch.Tensor:
        feature_list = [self._node_features[node] for node in self._nodes]
        return torch.tensor(feature_list, dtype=torch.float32)

    def get_resource_features(self) -> torch.Tensor:
        return torch.tensor(self._resource_features, dtype=torch.float32)

    def _preprocess_features(self, features: Dict) -> List[float]:
        return [
            features['first_layer_task'],
            features['first_layer_instance'],
            features['layers_task'],
            features['child_task_numbers'],
            features['child_instance_numbers']
        ]

    def visualize(self, path, fig_name):
        G = nx.DiGraph()
        G.add_nodes_from(self._nodes)
        # print(f'len of nodes {len(self._nodes)}')
        # print(f'edges are {len(self._edges)}')
        G.add_edges_from(self._edges)

        if len(self._nodes) == 1:
            return None

        # Convert graph nodes to strings
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

        # Ensure labels use string keys
        labels = {str(n): f"{n}" for n in G.nodes()}
      #  print(f'len of labels {len(labels)}')

        pos = nx.spring_layout(G)
        #print('pos is {0}'.format(pos))
        plt.figure(figsize=(15, 6), facecolor='#3b3e46')

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#225b3c')
        nx.draw_networkx_edges(
            G, pos, edge_color='#c1a162', arrows=True,
            arrowstyle='-|>', arrowsize=10, width=2,
            connectionstyle='arc3,rad=0.2'
        )

        title = fig_name
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=15, font_color='#c8b094', font_weight='bold')

        if nx.is_directed_acyclic_graph(G):
            pass
        else:
            cycles = list(nx.simple_cycles(G))
            print(f"Cycles detected in DAG for {title}: {cycles}")

     #   print(f"Graph Edges: {G.edges}")

        plt.title(title, fontsize=24, color='#c8b094')
        plt.axis('off')
        plt.tight_layout()

        # Ensure directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the figure with a unique name
        save_path = os.path.join(path, f"{fig_name}.png")
        plt.savefig(save_path)
        plt.close()

     #   print(f"Figure saved at {save_path}")
