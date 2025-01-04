from operator import attrgetter
import pandas as pd
import numpy as np

from cogito.job import JobConfig, TaskConfig
from agora.DAG.utils.feature_synthesize import father_task_indices
from agora.DAG.utils.preprocess_trace import dropOutlier
class CSVReader(object):
    def __init__(self, filename):
        self.filename = filename
        df = pd.read_csv(self.filename)
        df.columns = ['task_name', 'inst_num', 'job_name', 'task_type',
                              'status', 'start_time', 'end_time', 'plan_cpu',
                              'plan_mem']
        df=dropOutlier(df)
        print("length of dataframe: ", len(df))
        df['job_id'] = df['job_name'].apply(lambda x: x.split('_')[-1])
        df.instances_num = df.inst_num.astype(dtype=int)

        job_task_map = {}
        job_submit_time_map = {}
        for i in range(len(df)):
            series = df.iloc[i]
            job_id = series.job_id
            task_id, parent_indices = father_task_indices(series.task_name)

            cpu = series.plan_cpu//100
            memory = series.plan_mem
            #disk = series.disk
            duration = series.end_time - series.start_time

            submit_time = series.start_time
            instances_num = series.inst_num*20
            if duration > 0:
                task_configs = job_task_map.setdefault(job_id, [])
                task_configs.append(TaskConfig(task_id, instances_num, cpu, memory,0, duration, parent_indices))
                job_submit_time_map[job_id] = submit_time


        job_configs = []
        for job_id, task_configs in job_task_map.items():

            job_configs.append(JobConfig(job_id, job_submit_time_map[job_id], task_configs))
        job_configs.sort(key=attrgetter('submit_time'))

        self.job_configs = job_configs

    def generate(self, offset, number):
        number = number if offset + number < len(self.job_configs) else len(self.job_configs) - offset
        ret = self.job_configs[offset: offset + number]
        print("Length of job_configs is  %d" % len(ret))
        the_first_job_config = ret[0]
        submit_time_base = the_first_job_config.submit_time

        tasks_number = 0
        task_instances_numbers = []
        task_instances_durations = []
        task_instances_cpu = []
        task_instances_memory = []
        for job_config in ret:
            job_config.submit_time -= submit_time_base
            tasks_number += len(job_config.task_configs)
            for task_config in job_config.task_configs:
                task_instances_numbers.append(task_config.instances_number)
                task_instances_durations.extend([task_config.duration] * int(task_config.instances_number))
                task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_number))
                task_instances_memory.extend([task_config.memory] * int(task_config.instances_number))
                #print(f'Task instance number {task_config.instances_number} , task_instances_duration {task_config.duration}')
        print('Jobs number: ', len(ret))
        print('Tasks number:', tasks_number)

        print('Task instances number mean: ', np.mean(task_instances_numbers))
        print('Task instances number std', np.std(task_instances_numbers))

        print('Task instances cpu mean: ', np.mean(task_instances_cpu))
        print('Task instances cpu std: ', np.std(task_instances_cpu))

        print('Task instances memory mean: ', np.mean(task_instances_memory))
        print('Task instances memory std: ', np.std(task_instances_memory))
        print('Task instances duration mean: ', np.mean(task_instances_durations))
        print('Task instances duration std: ', np.std(task_instances_durations))

        return ret
