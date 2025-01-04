from cogito.job import Job


class Broker(object):
    job_cls = Job

    def __init__(self, env, job_configs,ep=None):
        self.env = env
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.job_configs = job_configs
        self.episode=ep
    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def run(self):
        for i,job_config in enumerate(self.job_configs):
            # if self.episode!=None:
            #
            #
            #     if self.episode<i:
            #         break

            print(f" next job_config submit_time={job_config.submit_time}")
            #print(f"self.env.now={self.env.now}")

            assert job_config.submit_time >= self.env.now
            yield self.env.timeout(job_config.submit_time - self.env.now)
            job = Broker.job_cls(self.env, job_config)
            print('a task arrived at time %f' % self.env.now)
            self.cluster.add_job(job)
        self.destroyed = True
