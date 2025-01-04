import torch
from typing import Dict, List, Tuple
import networkx as nx

import matplotlib.pyplot as plt
from utils.csv_reader import CSVReader
from utils.DAG import DAG

# Assuming you have a CSV file with job data
csv_file_path = './jobs_files/job.csv'

# Create a CSVReader object
csv_reader = CSVReader(csv_file_path)

# Generate job configurations (let's say we want to get 5 jobs starting from the first one)
job_configs = csv_reader.generate(offset=0, number=5)

# Create and visualize a DAG for each job
for job_config in job_configs:
    # Create a DAG object
    dag = DAG(job_config)

    # Visualize the DAG
    dag.visualize()

    # Print some information about the DAG
    print(f"Job ID: {job_config.job_id}")
    print(f"Number of nodes: {len(dag.nodes)}")
    print(f"Number of edges: {len(dag.edges)}")
    print("Adjacency matrix shape:", dag.get_adjacency_matrix().shape)
    print("Node features shape:", dag.get_node_features().shape)
    print("\n")

# Show all the plotted DAGs
plt.show()