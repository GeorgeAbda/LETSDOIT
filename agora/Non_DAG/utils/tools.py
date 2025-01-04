import time
import numpy as np
import tensorflow as tf
import math
def transform_to_first_digit(value):
    # Handle zero as a special case
    if value == 0:
        return 0

    # Calculate the magnitude (10 raised to the power of one less than the number of digits)
    magnitude = 10 ** (int(math.log10(abs(value))))

    # Extract the first digit and reconstruct the number
    result = (value // magnitude) * magnitude

    return result
def average_completion(exp):
    completion_time = 0
    number_task = 0
    for job in exp.simulation.cluster.jobs:
        for task in job.tasks:
            number_task += 1
            completion_time += (task.finished_timestamp - task.started_timestamp)
    return completion_time / number_task


def average_energy(exp):
    energy_usage = {"PowerEdgeC5220": [194, 672, 0.9569],
                    "HPProLiantDL2000": [178, 660, 0.7257],
                    "IBMSystemx3630M4": [58.1, 269, 1.5767],
                    "IBMSystemxiDataPlexdx360M3": [92.7, 341, 0.7119],
                    "Systemx3200M3": [45, 119, 1.5324]
                    }
    power = 0
    for machine in exp.simulation.cluster.machines:
        [p_idle, p_busy, r] = energy_usage[machine.name]
        power += p_idle + (p_busy - p_idle) * (machine.state["cpu"]) ** r

    return power


def average_slowdown(exp):
    slowdown = 0
    number_task = 0
    for job in exp.simulation.cluster.jobs:
        for task in job.tasks:
            number_task += 1
            slowdown += (task.finished_timestamp - task.started_timestamp) / task.task_config.duration
    return slowdown / number_task


def multiprocessing_run(episode, trajectories, makespans, average_completions, average_slowdowns):
    np.random.seed(int(time.time()))
    # tf.random.set_random_seed(time.time())
    episode.run()
    trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
    makespans.append(episode.simulation.env.now)
    # print(episode.simulation.env.now)
    average_completions.append(average_completion(episode))
    average_slowdowns.append(average_slowdown(episode))
