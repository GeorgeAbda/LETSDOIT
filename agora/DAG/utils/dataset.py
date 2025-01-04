import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import networkx as nx

def graph_to_pyg_data(matrix):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph) # Use from_numpy_array for directed graph
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(list(G.nodes), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data