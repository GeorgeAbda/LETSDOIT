import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from agora.DAG.utils.gnn.network import GCN
from agora.DAG.utils.dataset import graph_to_pyg_data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from agora.DAG.utils.gnn.network import GCN
from agora.DAG.utils.dataset import graph_to_pyg_data
import networkx as nx

# Import functions from the dag_processing module
from agora.DAG.utils.preprocess_trace import process_data, jobToDict

# Load and process the data
input_file = '../jobs_files/batch_task.csv'  # Update this path
line_num = 2000  # Update this as needed

ArrivaMatList, jobs_info = process_data(input_file, line_num,True,1)
print(f"Number of jobs: {len(jobs_info)}")
# Convert each adjacency matrix to a graph and then to PyG data
graphs = [graph_to_pyg_data(matrix) for matrix in ArrivaMatList]

# Split into training and testing sets
train_graphs = graphs[:int(len(graphs)*0.8)]
test_graphs = graphs[int(len(graphs)*0.8):]

train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1)



# Instantiate model, optimizer
model = GCN(num_features=ArrivaMatList[0].shape[0], num_classes=10)  # Adjust num_classes
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(model, loader, optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# Testing function
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# Train and evaluate
for epoch in range(1, 201):
    train(model, train_loader, optimizer)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
