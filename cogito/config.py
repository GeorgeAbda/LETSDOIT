class TaskInstanceConfig(object):
    def __init__(self, task_config):
        self.cpu = task_config.cpu
        self.memory = task_config.memory
        self.disk = task_config.disk
        self.duration = task_config.duration


class TaskConfig(object):
    def __init__(self, task_index, instances_number, cpu, memory, disk, duration, parent_indices=None,task_type=None):
        self.task_index = task_index
        self.instances_number = instances_number
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration
        self.parent_indices = parent_indices
        self.task_type = task_type

class JobConfig(object):
    def __init__(self, idx, submit_time, task_configs):
        self.submit_time = submit_time
        self.task_configs = task_configs
        self.id = idx
        self.tasks_map = {}

        for task_config in self.task_configs:
            task_index = str(task_config.task_index)
            self.tasks_map[task_index] = task_config
