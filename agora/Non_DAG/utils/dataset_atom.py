from operator import attrgetter
import pandas as pd
import numpy as np
import pickle


from cogito.job import JobConfig, TaskConfig

df = pd.read_csv('../jobs_files/jobs.csv')
grouped = df.groupby('job_id')
jobs = [job for _, job in grouped]

dataset_jobs = []
for job in jobs:
    job_len = len(job)
    tasks_job = []
    for i in range(job_len):
        task = df.iloc[i]
        cpu = task.cpu
        memory = task.memory
        disk = task.disk
        task_instances = task.instances_num
        job_id = task.job_id
        tasks_job.append([cpu, memory, disk, task_instances, job_id])
    dataset_jobs.append(tasks_job)

# Save the list to a file using pickle
with open('dataset_atom.pickle', 'wb') as file:
    pickle.dump(dataset_jobs, file)
