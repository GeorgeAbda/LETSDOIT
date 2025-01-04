from cogito import job as job_module
from agora.DAG.utils.feature_synthesize import task_features

#
# # One way to add feature property
# class Task(job_module.Task):
#     def __init__(self, env, job, task_config):
#         super().__init__(env, job, task_config)
#         self._features = None
#
#     @property
#     def feature(self):
#         self._features = self.job.features[self.task_index]
#         return self._features
#
#
# job_module.Job.task_cls = Task


# One way (Ends)


# Another way to add feature property



# Another way (Ends)


class Job(job_module.Job):
    def __init__(self, env, job_config):
        super().__init__(env, job_config)
        self.features = task_features(self)
class JobBatch:
    def __init__(self, dags, resource_features):
        self.dag = dags
        self.resource_features = resource_features
    @property
    def tasks(self):
        if isinstance(self.dag,list):
            return [task for task_list in self.dag for task in task_list.job.tasks_which_has_waiting_instance]
        return [task for task in self.dag.job.tasks_which_has_waiting_instance]

    def __len__(self):
        return len(self.tasks)

def feature(self):
    self._features = self.job.features[self.task_index]
    return self._features


setattr(job_module.Task, 'feature', property(feature))