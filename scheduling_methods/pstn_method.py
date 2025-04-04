import json
import os
import re
import time
import copy
from collections import defaultdict
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

import general
from rcpsp_max.solvers.timestamp_model import TimestampedModel
from rcpsp_max.temporal_networks.pstn_factory import get_resource_chains, RCPSP_PSTN, add_resource_chains, \
    lognormal_to_normal_params
from temporal_networks.cstnu_tool.pstnu_to_xml import pstn_to_xml
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.pstn import PSTN
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_convert_algorithm, run_dc_algorithm
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

logger = general.logger.get_logger(__name__)


def solve_det(timestamp_model: TimestampedModel, increase_for_offline):
    temp_model = copy.deepcopy(timestamp_model)

    # Step 1: Increase job durations
    for order in temp_model.orders:
        for prod in order.products:
            for job in prod.jobs:
                job.duration = int(job.duration + increase_for_offline * sqrt(job.duration))

    # Step 2: Solve full problem
    res, accepted_orders, schedule, t1, t2 = temp_model.solve_cp(60)
    if res:
        return res, accepted_orders, schedule  # Solution found

    # Step 3: Solve with only mandatory orders
    temp_model_mandatory = copy.deepcopy(temp_model)
    temp_model_mandatory.orders = [order for order in temp_model_mandatory.orders if order.required]

    res, accepted_orders, schedule, t1, t2 = temp_model_mandatory.solve_cp(60)
    if not res:
        return res, accepted_orders, schedule

    # Step 4: Try adding optional orders one by one
    feasible_orders = temp_model_mandatory.orders[:]
    optional_orders = sorted(
        [order for order in temp_model.orders if not order.required],
        key=lambda order: order.value,
        reverse=True
    )
    for order in optional_orders:
        temp_model_test = copy.deepcopy(temp_model_mandatory)  # Fresh copy for each attempt
        temp_model_test.orders = feasible_orders + [order]  # Keep previous successful ones

        res, new_accepted_orders, new_schedule, t1, t2 = temp_model_test.solve_cp(60)

        if res:
            feasible_orders.append(order)  # Mark this optional order as permanently added
            accepted_orders = new_accepted_orders
            schedule = new_schedule

        del temp_model_test  # Explicitly remove model to free memory

    return res, accepted_orders, schedule

def run_pstn_offline(schedule, timestamp_model: TimestampedModel, sample, increase_for_offline=0, std_k=0, num_orders=0,
                     num_pr=0, accepted={}):
    data_dict = {"size": len(timestamp_model.orders) * len(timestamp_model.orders[0].products),
                 "can_build_pstn": True, "can_convert_to_dc_stnu": None, "offline_time": None, "std": std_k, "cons_l": increase_for_offline,
                 "deterministic_obj": None, "probability_mass": None,
                 "real_objective": None, "number_produced": None, "rte_time": None}
    # Build the PSTN using the instance information and the resource chains
    schedule = schedule[schedule['start'].notna()]
    schedule = schedule.to_dict('records')
    resource_chains, resource_assignments = get_resource_chains(schedule, timestamp_model.resources,
                                                                timestamp_model.orders, accepted,
                                                                complete=True)
    pstn = RCPSP_PSTN.from_rcpsp_max_instance(timestamp_model.orders, accepted)
    pstn = add_resource_chains(pstn, resource_chains)
    output_directory = f"{num_orders}_{num_pr}_{increase_for_offline}_conservative"
    output_path = f"temporal_networks/cstnu_tool/xml_files/{output_directory}"

    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    output_location = f"{output_directory}/{num_orders}_{num_pr}_{sample}_{std_k}_{increase_for_offline}_pstn"


    pstn_to_xml(pstn, output_location, "temporal_networks/cstnu_tool/xml_files")
    return output_location, data_dict


def convert_to_stnu(filename, data_dict):
    res, output_location = run_convert_algorithm("temporal_networks/cstnu_tool/xml_files", filename)
    if res and res['is_convertible'] is not None and res['is_convertible'] is True:
        estnu = STNU.from_graphml(output_location)
        data_dict["can_convert_to_dc_stnu"] = True
        data_dict["probability_mass"] = res["probability_mass"]
    else:
        estnu = None
        data_dict["can_convert_to_dc_stnu"] = False

    return estnu, data_dict, output_location


def run_pstn_online(my_estnu, time_model, dc_stnu_file, number_of_samples, data_dict):
    real_profit = 0
    avg_time = 0
    data = []
    for ik in range(number_of_samples):
        start_time = time.time()
        estnu = copy.deepcopy(my_estnu)
        sample = generate_samples(estnu, time_model)
        res, rte_data, not_succeeded = rte_star(estnu, oracle="sample", sample=sample)
        completed_global = set()
        dc_stnu = STNU.from_graphml(dc_stnu_file)
        while not res:
            current_rte = rte_data
            current_schedule = {
                estnu.translation_dict.get(key, key): value
                for key, value in current_rte.f.items()
            }  # Convert translation indices to original names

            completed, cancelled = adjust_network(dc_stnu, current_rte, current_schedule, time_model, sample)
            completed_global.update(completed)
            remove_cancelled_jobs(set(cancelled), dc_stnu)
            remove_cancelled_jobs(set(completed), dc_stnu)
            alter_deadlines(dc_stnu, rte_data.now)
            stnu_to_xml(dc_stnu, "check_graph", "temporal_networks/cstnu_tool/xml_files")
            dc, output_location_estnu = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files",
                                                          "check_graph")
            altered_estnu = STNU.from_graphml(output_location_estnu)
            res, rte_data, not_succeeded = rte_star(altered_estnu, oracle="sample", sample=sample)
        completed, cancelled = adjust_network(dc_stnu, rte_data, [], time_model, sample)
        completed_global.update(completed)
        real_profit += calculate_real_profit(completed_global, time_model)
        end_time = time.time()
        avg_time += end_time - start_time
    data_dict["real_objective"] = real_profit / number_of_samples
    data_dict["rte_time"] = avg_time / number_of_samples
    return data_dict


def find_jobs_by_status(estnu, current_schedule, status):
    scheduled_jobs = []
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_' + status)

    for key in estnu.nodes & current_schedule.keys():
        activity = estnu.translation_dict.get(key, "")
        if activity not in {'Z', 'INITIAL_EVENT'}:
            match = pattern.match(activity)
            if match:
                scheduled_jobs.append(tuple(map(int, match.groups())))

    return scheduled_jobs


def adjust_network(estnu, rte_data, schedule_data, time_model, sample):
    current_rte = rte_data
    current_schedule = current_rte.f
    timestamp_now = current_rte.now

    start_jobs = set(find_jobs_by_status(estnu, current_schedule, 'start'))
    finish_jobs = set(find_jobs_by_status(estnu, current_schedule, 'finish'))

    completed_jobs = start_jobs & finish_jobs
    only_started = start_jobs - finish_jobs

    to_remove = set()
    to_add = set()

    for (i, j, k) in only_started:
        my_order, my_product, my_job = find_relevant_job_in_time_model(i, j, k, time_model)
        if my_job.duration == 0:
            to_remove.add((i, j, k))
            to_add.add((i, j, k))

    # Properly update only_started and completed_jobs
    only_started = only_started - to_remove  # Avoid modifying while iterating
    completed_jobs |= to_add  # Ensure updates are correct

    # Find cancelled jobs
    cancelled = set()
    for (i, j, k) in only_started:
        all_cancelled_jobs = find_all_relevant_jobs(i, j, time_model)
        cancelled.update(set(all_cancelled_jobs))  # Ensure consistent set usage

    # Ensure correct subtraction
    completed_jobs -= cancelled

    partially_finish = set(find_products_that_have_not_fully_completed(completed_jobs, time_model))
    completed_jobs -= partially_finish
    cancelled.update(partially_finish)

    # Convert cancelled to list if needed
    cancelled = list(cancelled)

    return completed_jobs, cancelled


def alter_deadlines(estnu, now):
    edges = estnu.edges.get(0, {})
    for node_to, edge in edges.items():
        node_idx = estnu.translation_dict.get(node_to)

        if node_idx and "finish" in node_idx:
            edge.weight = now - edge.weight if edge.weight is not None else now


def find_edges_to_zero(estnu):
    edges_to_zero = {}

    for node_from, edges in estnu.edges.items():
        if 0 in edges:  # Check if there is an edge to node 0
            edges_to_zero[node_from] = edges[0]  # Store edge information

    return edges_to_zero


def remove_cancelled_jobs(cancelled, estnu):
    for (i, j, k) in cancelled:
        i_key = f'{i}_{j}_{k}_start'
        j_key = f'{i}_{j}_{k}_finish'

        i_idx = estnu.translation_dict_reversed.get(i_key)
        j_idx = estnu.translation_dict_reversed.get(j_key)
        if i_idx is not None:
            remove_node_and_edges(estnu, i_idx, i_key)

        if j_idx is not None:
            remove_node_and_edges(estnu, j_idx, j_key)


def remove_node_and_edges(estnu, node_idx, node_key):
    # Remove the node from the list of nodes
    if node_idx in estnu.nodes:
        estnu.nodes.remove(node_idx)

    # Remove all edges connected to this node
    if node_idx in estnu.edges:
        del estnu.edges[node_idx]  # Remove outgoing edges

    # Remove edges where this node appears as a destination
    for from_node in list(estnu.edges.keys()):  # Copy keys to avoid modifying during iteration
        if node_idx in estnu.edges[from_node]:
            del estnu.edges[from_node][node_idx]

    # Clean up translation dictionaries
    estnu.translation_dict.pop(node_idx, None)
    estnu.translation_dict_reversed.pop(node_key, None)

def adjust_deadlines(estnu: STNU, now):
    edges = estnu.edges[0]
    for edge in edges:
        edge.weight = edge.weight - now


def find_relevant_job_in_time_model(i, j, k, time_model):
    for order in time_model.orders:
        if order.id == int(i):
            my_order = order
            break
    for product in my_order.products:
        if product.id == int(j):
            my_product = product
            break
    for job in my_product.jobs:
        if job.id == int(k):
            my_job = job
            break
    return my_order, my_product, my_job

def find_products_that_have_not_fully_completed(completed_jobs, time_model):
    products = []
    for (i, j, k) in completed_jobs:
        order, product, job = find_relevant_job_in_time_model(i, j, k, time_model)
        for rel_job in product.jobs:
            if (order.id, product.id, rel_job.id) not in completed_jobs:
                products.append((i, j, k))
                break
    jobs = set()
    for (i, j, k) in products:
        jobs.update(find_all_relevant_jobs(i, j, time_model))
    return jobs


def calculate_real_profit(completed, time_model):
    profit = 0
    visited = set()
    for (i, j, k) in completed:
        if (i, j) not in visited:
            order, product, job = find_relevant_job_in_time_model(i, j, k, time_model)
            visited.add((i, j))
            profit += product.value
    return profit



def find_all_relevant_jobs(i, j, time_model):
    for order in time_model.orders:
        if order.id == int(i):
            my_order = order
            break
    for product in my_order.products:
        if product.id == int(j):
            my_product = product
            break
    jobs = []
    for job in my_product.jobs:
        jobs.append((my_order.id, my_product.id, job.id))
    return jobs




def generate_samples(estnu, time_model):
    """
    Generates log-normal samples for contingent nodes based on job durations.
    """
    sample = {}
    for order in time_model.orders:
        for product in order.products:
            for job in product.jobs:
                task = f'{order.id}_{product.id}_{job.id}'
                activ_node = estnu.translation_dict_reversed[f'{task}_{PSTN.EVENT_START}']
                if job.duration != 0:
                    contingent_node = estnu.translation_dict_reversed[f'{task}_{PSTN.EVENT_FINISH}']
                    mean = job.normalized_duration
                    std = job.normalized_std
                    s = int(np.random.lognormal(mean=mean, sigma=std))
                    sample[contingent_node] = s
                    sample[activ_node] = s
                else:
                    sample[activ_node] = 0
    with open("samples.txt", "w") as f:
        for key, value in sample.items():
            f.write(f"{key}={value}\n")

    return sample


def alter_time_lags(estnu: STNU, completed_jobs, rte_data, now, time_model):
    edges_to_zero = find_edges_to_zero(estnu)
    for e in edges_to_zero.values():
        if e.weight < 0:
            remaining_lag = -e.weight - now
            if remaining_lag >= 0:
                estnu.edges[e.node_from][0].weight = min(0, -remaining_lag)
    for (i, j, k) in completed_jobs:
        my_order, my_product, my_job = find_relevant_job_in_time_model(i, j, k, time_model)
        for succ in my_job.successors:
            succ_start = estnu.translation_dict_reversed.get(f"{my_order.id}_{my_product.id}_{succ.id}_start")
            if succ_start is not None:
                if succ_start in estnu.nodes:
                    initial_lag = succ.lag * 1000
                    time_past = now - rte_data[f'{i}_{j}_{k}_start']
                    # if initial_lag < 0:
                    #     remaining_lag = initial_lag + time_past
                    #     if remaining_lag <= 0:
                    #         estnu.set_ordinary_edge(succ_start, 0, -remaining_lag * 1000)
                    if initial_lag > 0:
                        remaining_lag = initial_lag - time_past
                        if remaining_lag >= 0:
                            estnu.set_ordinary_edge(succ_start, 0, -remaining_lag)