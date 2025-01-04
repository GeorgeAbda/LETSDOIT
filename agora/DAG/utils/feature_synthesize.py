
def father_task_indices(task_id):
    task_id = str(task_id)

    father_indices = []
    b = task_id.split('_')
    #print('b: {}'.format(b))
    if not b[-1].isdigit():
        del b[-1]
    #print('b: {}'.format(b))

    if len(b) > 1:
        task_index = b[0][1]
        father_indices = b[1:]
    else:
        task_index = task_id[1]

    return task_index, father_indices


def task_features(job):
    child_indices = {}
    father_indices = {}
    tasks =job.tasks_which_has_waiting_instance

    for task in tasks:
        task_index = task.task_config.task_index
        task_parent_indices = task.task_config.parent_indices
        # print(f'task_parent_indices are {task_parent_indices}')
        father_indices[task_index] = [task_parent_indice for task_parent_indice in task_parent_indices if task_parent_indice in tasks]
        child_indices[task_index] = []
        for parent_indice in task_parent_indices:
            child_indice = child_indices.setdefault(parent_indice, [])
            child_indice.append(task_index)

    descendant_indices = {}
    for task_index in child_indices.keys():
        descendant_indice = []
        queue = [task_index]
        while queue:
            current = queue.pop(0)
            for child in child_indices.get(current, []):
                if child not in descendant_indice:
                    descendant_indice.append(child)
                    queue.append(child)
        descendant_indices[task_index] = descendant_indice

    task_features = {}
    for task_index in child_indices.keys():
        child_index = child_indices[task_index]
        task_feature = task_features.setdefault(task_index, {})
        task_feature['first_layer_task'] = len(child_index)
        task_feature['first_layer_instance'] = sum(
            job.tasks_map[child].task_config.instances_number for child in child_index)
        task_feature['layers_task'] = 0
        task_feature['child_task_numbers'] = len(descendant_indices[task_index])
        task_feature['child_instance_numbers'] = sum(
            job.tasks_map[desc].task_config.instances_number for desc in descendant_indices[task_index])


    # Calculate layers_task
    for task_index in reversed(list(topological_sort(father_indices))):
        child_index = child_indices.get(task_index, [])
        if not child_index:
            task_features[task_index]['layers_task'] = 0
        else:
            task_features[task_index]['layers_task'] = 1 + max(
                task_features[child]['layers_task'] for child in child_index)

    return task_features


def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:

        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    queue = [node for node in graph if in_degree[node] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(graph):
        raise ValueError("Graph has a cycle")

    return result

def weights_calculate(dag):
    weight_tasks = {}
    task_features_dict = task_features(dag)

    for task in dag.nodes:
        feature = task_features_dict[task]
        weight = (feature['first_layer_task'] +
                  feature['first_layer_instance'] +
                  feature['layers_task'] +
                  feature['child_task_numbers'] +
                  feature['child_instance_numbers'])
        weight_tasks.setdefault(weight, []).append(task)

    sorted_weights = sorted(weight_tasks.keys(), reverse=True)
    sorted_tasks = []
    for weight in sorted_weights:
        sorted_tasks.extend(weight_tasks[weight])

    return sorted_tasks