import copy
import math
import random
import time

import numpy as np
import pandas as pd

from rcpsp_max.solvers.timestamp_model import TimestampedModel
from rcpsp_max.temporal_networks.pstn_factory import compute_lognormal_bounds, lognormal_to_normal_params

random.seed(42)


def run_proactive_time_offline(time_model: TimestampedModel, mode=1):
    # Initialize data
    time_limit = 6
    data_dict = {
        "method": f"proactive_{mode}",
        "time_limit": time_limit,
        "feasibility": False,
        "obj": np.inf,
        "time_offline": np.inf,
        "time_online": np.inf,
        "start_times": None,
        "real_durations": None,
        "mode": mode
    }
    temp_model = copy.deepcopy(time_model)

    def get_quantile(lb, ub, p):
        if lb == ub:
            quantile = lb
        else:
            quantile = int(lb + p * (ub - lb + 1) - 1)

        return quantile

    start_offline = time.time()
    # Solve very conservative schedule
    lb = {}
    ub = {}
    for i in time_model.orders:
        for j in i.products:
            for k in j.jobs:
                mean, std = lognormal_to_normal_params(k.duration, k.std)
                k.normalized_duration = mean
                k.normalized_std = std
                if k.duration == 0:
                    lb[f'{i.id}_{j.id}_{k.id}'] = 0
                    ub[f'{i.id}_{j.id}_{k.id}'] = 0
                else:
                    lb[f'{i.id}_{j.id}_{k.id}'] = int(np.exp(mean - 3.3 * k.std))
                    ub[f'{i.id}_{j.id}_{k.id}'] = int(np.exp(mean + 3.3 * k.std))

    if mode == 1:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = ub[f'{i.id}_{j.id}_{k.id}']
        res, accepted_orders, data, init_time, solve_time = solve_cp_proactive(temp_model)
        if res:
            start_times = data['start'].tolist()

    elif mode == 0.25:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = get_quantile(lb[f'{i.id}_{j.id}_{k.id}'], ub[f'{i.id}_{j.id}_{k.id}'], 0.25)
        res, accepted_orders, data, init_time, solve_time = solve_cp_proactive(temp_model)
        if res:
            start_times = data['start'].tolist()

    elif mode == 0.4:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = get_quantile(lb[f'{i.id}_{j.id}_{k.id}'], ub[f'{i.id}_{j.id}_{k.id}'], 0.4)
        res, accepted_orders, data, init_time, solve_time = solve_cp_proactive(temp_model)
        if res:
            start_times = data['start'].tolist()

    elif mode == 0.75:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = get_quantile(lb[f'{i.id}_{j.id}_{k.id}'], ub[f'{i.id}_{j.id}_{k.id}'], 0.75)
        res, accepted_orders, data, init_time, solve_time = solve_cp_proactive(temp_model)
        if res:
            start_times = data['start'].tolist()

    elif mode == 0.5:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = get_quantile(lb[f'{i.id}_{j.id}_{k.id}'], ub[f'{i.id}_{j.id}_{k.id}'], 0.5)
        res, accepted_orders, data, init_time, solve_time = temp_model.solve_cp(time_limit)
        if res:
            start_times = data['start'].tolist()

    elif mode == 0.9:
        for i in temp_model.orders:
            for j in i.products:
                for k in j.jobs:
                    k.duration = get_quantile(lb[f'{i.id}_{j.id}_{k.id}'], ub[f'{i.id}_{j.id}_{k.id}'], 0.9)
        res, accepted_orders, data, init_time, solve_time = solve_cp_proactive(temp_model)
        if res:
            start_times = data['start'].tolist()

    else:
        raise NotImplementedError

    return res, accepted_orders, data

def solve_cp_proactive(temp_model: TimestampedModel):
    temp_model_mandatory = copy.deepcopy(temp_model)
    temp_model_mandatory.orders = [order for order in temp_model_mandatory.orders if order.required]

    res, accepted_orders, schedule, t1, t2 = temp_model_mandatory.solve_cp(60)
    if not res:
        return res, accepted_orders, schedule, t1, t2

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

    return res, accepted_orders, schedule, t1, t2


def run_proactive_online(time_model, my_std, data):
    """
    Evaluate the robust approach
    """
    sample = {}
    for i in time_model.orders:
        for j in i.products:
            for k in j.jobs:
                mean, std = lognormal_to_normal_params(k.duration, my_std * math.sqrt(k.duration))
                sample[f'{i.id}_{j.id}_{k.id}'] = int(np.random.lognormal(mean, std))

    schedule = pd.DataFrame(columns=['task', 'start', 'end'])
    accepted = {}
    profit = 0

    for i in time_model.orders:
        for j in i.products:
            for k in j.jobs:
                task_id = f'{i.id}_{j.id}_{k.id}'
                task_entry = data[data["task"] == task_id]

                if not task_entry.empty:
                    start_time = task_entry["start"].iloc[0]
                    end_time = start_time + sample[task_id]

                    schedule = pd.concat(
                        [schedule, pd.DataFrame({'task': [task_id], 'start': [start_time], 'end': [end_time]})],
                        ignore_index=True)
                    accepted[i.id] = True
                else:
                    accepted[i.id] = False

    for i in time_model.orders:
        if accepted.get(i.id, False):
            for j in i.products:
                profit += j.value

    sat1 = time_model.check_time_lag_constraint(schedule)
    sat2 = time_model.check_deadline_constraint(schedule)
    sat3 = time_model.check_obligatory_constraint(schedule, accepted)
    sat4 = time_model.check_resource_constraint(schedule)

    if sat1 and sat2 and sat3 and sat4:
        return schedule, profit

    return schedule, 0
