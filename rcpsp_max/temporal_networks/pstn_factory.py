from typing import List

import general.logger
from rcpsp_max.entities.order import Order
from rcpsp_max.entities.resource import Resource
from temporal_networks.pstn import PSTN
import numpy as np
from collections import defaultdict

logger = general.logger.get_logger(__name__)


class RCPSP_PSTN(PSTN):
    def __init__(self, origin_horizon=True):
        super().__init__(origin_horizon)

    @classmethod
    def from_rcpsp_max_instance(cls, orders: List[Order], accepted):
        pstn = cls(origin_horizon=False)
        initial_time = pstn.add_node("INITIAL_EVENT")
        for i in orders:
            if accepted[i.id]:
                for j in i.products:
                    for k in j.jobs:
                        task_start = pstn.add_node(f'{i.id}_{j.id}_{k.id}_{PSTN.EVENT_START}')
                        task_finish = pstn.add_node(f'{i.id}_{j.id}_{k.id}_{PSTN.EVENT_FINISH}')
                        mean, std = lognormal_to_normal_params(k.duration, k.std)
                        if mean < 0:
                            mean = 0
                        lower_bound, upper_bound = compute_lognormal_bounds(mean, std)
                        k.normalized_duration = mean
                        k.normalized_std = std
                        if k.duration == 0:
                            pstn.add_tight_constraint(task_start, task_finish, 0)
                        else:
                            pstn.add_contingent_link(task_start, task_finish, lower_bound, upper_bound, mean, std)
                        pstn.set_ordinary_edge(task_start, initial_time, 0)
                        pstn.set_ordinary_edge(initial_time, task_finish, i.deadline)
                for k in j.jobs:
                    for succ in k.successors:
                        i_idx = pstn.translation_dict_reversed[f'{i.id}_{j.id}_{k.id}_{PSTN.EVENT_START}']
                        j_idx = pstn.translation_dict_reversed[f'{i.id}_{j.id}_{succ.id}_{PSTN.EVENT_START}']
                        pstn.set_ordinary_edge(j_idx, i_idx, -succ.lag)

        return pstn


def compute_lognormal_bounds(mu, sigma, k=3.3):
    lower_bound = int(np.exp(mu - k * sigma))
    upper_bound = int(np.exp(mu + k * sigma))

    return lower_bound, upper_bound


def lognormal_to_normal_params(mu_X, sigma_X):
    if mu_X == 0:
        mu_X += 0.00001
    sigma_sq = np.log(1 + (sigma_X**2 / mu_X**2))
    mu = np.log(mu_X**2 / np.sqrt(mu_X**2 + sigma_X**2))
    if mu < 0:
        mu = 0
    return mu, np.sqrt(sigma_sq)


def remove_all_duplicates(tuples_list):
    unique_tuples = []
    seen = set()

    for current_tuple in tuples_list:
        if current_tuple not in seen:
            unique_tuples.append(current_tuple)
            seen.add(current_tuple)

    return unique_tuples


def get_resource_chains(schedule, resources: List[Resource], orders: List[Order], my_y, complete=False):
    # schedule is a list of dicts of this form:
    # {"task": i, " "start": start, "end": end}
    reserved_until = {}
    for resource in resources:
        reserved_until |= {resource.id: [0] * resource.capacity}

    resource_use = {}

    resource_assignment = []
    for d in sorted(schedule, key=lambda d: d['start']):
        name = d['task']
        i, j, k = map(int, name.split('_'))
        if my_y[i]:
            my_order = next(order for order in orders if order.id == i)
            my_product = next(product for product in my_order.products if product.id == j)
            my_job = next(job for job in my_product.jobs if job.id == k)

            for resource_index, required in enumerate(my_job.resources):
                reservations = reserved_until[resource_index]
                assigned = []
                for idx in range(len(reservations)):
                    if len(assigned) == required:
                        break
                    if reservations[idx] <= d['start']:
                        reservations[idx] = d['end']
                        assigned.append({'task': d['task'],
                                         'resource_group': resource_index,
                                         'id': idx})
                        users = resource_use.setdefault((resource_index, idx), [])
                        users.append(
                            {'Task': d['task'], 'Start': d['start']})

                if len(assigned) < required:
                    ValueError(f'ERROR: only found {len(assigned)} of {required} resources (type {resource_index}) '
                          f'for task {d["task"]}')
                else:
                    assert len(assigned) == required
                    resource_assignment += assigned

    resource_chains = []
    if complete:
        for resource_activities in resource_use.values():
            if len(resource_activities) > 1:  # Check if there are multiple activities assigned to the same resource
                # Sort by start time
                resource_activities = sorted(resource_activities, key=lambda x: x["Start"])
                # To do keep track of edges that should be added to STN
                for i in range(1, len(resource_activities)):
                    for j in range(0, i):
                        predecessor = resource_activities[j]
                        successor = resource_activities[i]
                        resource_chains.append((predecessor["Task"],
                                                successor["Task"]))
    else:
        for resource_activities in resource_use.values():
            if len(resource_activities) > 1:  # Check if there are multiple activities assigned to the same resource
                # Sort by start time
                resource_activities = sorted(resource_activities, key=lambda x: x["Start"])

                # To do keep track of edges that should be added to STN
                for i in range(1, len(resource_activities)):
                    predecessor = resource_activities[i - 1]
                    successor = resource_activities[i]
                    resource_chains.append((predecessor["Task"],
                                            successor["Task"]))
    unique_tuples = remove_all_duplicates(resource_chains)
    return unique_tuples, resource_assignment


def add_resource_chains(stnu, resource_chains):
    for pred_task, succ_task in resource_chains:
        # the finish of the predecessor should precede the start of the successor
        pred_idx_finish = stnu.translation_dict_reversed[
            f"{pred_task}_{PSTN.EVENT_FINISH}"]  # Get translation index from finish of predecessor
        suc_idx_start = stnu.translation_dict_reversed[
            f"{succ_task}_{PSTN.EVENT_START}"]  # Get translation index from start of successor

        # add constraint between predecessor and successor
        stnu.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)

    return stnu


def detect_circular_dependencies(resource_chains):
    """
    Detects circular dependencies in a directed graph represented by resource_chains.

    Args:
        resource_chains (list of tuple): List of directed edges (predecessor, successor).

    Returns:
        list: A list of cycles found. If empty, no circular dependencies exist.
    """
    # Build graph from resource_chains
    graph = defaultdict(list)
    for predecessor, successor in resource_chains:
        graph[predecessor].append(successor)

    def dfs(node, visited, stack, current_path):
        """Recursive Depth-First Search to find cycles."""
        visited.add(node)
        stack.add(node)
        current_path.append(node)

        # Iterate over a static copy of neighbors to avoid modifying the graph while iterating
        for neighbor in list(graph[node]):
            if neighbor not in visited:
                if dfs(neighbor, visited, stack, current_path):
                    return True
            elif neighbor in stack:  # Cycle detected
                # Extract the cycle
                cycle_start = current_path.index(neighbor)
                cycles.append(current_path[cycle_start:])
                return True

        stack.remove(node)
        current_path.pop()
        return False

    # Detect cycles in the graph
    visited = set()
    stack = set()
    cycles = []

    # Iterate over the nodes in the graph
    for node in list(graph.keys()):
        if node not in visited:
            dfs(node, visited, stack, [])

    return cycles
