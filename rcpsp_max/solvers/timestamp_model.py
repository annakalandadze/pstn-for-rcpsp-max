from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
from typing import List
from docplex.cp.model import *

import rcpsp_max.entities.resource
from rcpsp_max.entities.order import Order
from rcpsp_max.entities.product import Product
from rcpsp_max.entities.resource import Resource

context.solver.local.execfile = '/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer'


class TimestampedModel:
    def __init__(self, orders: List[Order], resources: List[rcpsp_max.entities.resource.Resource],
                 products: List[Product], timespan):
        self.orders = orders
        self.resources = resources
        self.products = products
        self.times = [i for i in range(timespan)]

    # JSON helpers
    def to_dict(self):
        return {
            "orders": [order.to_dict() for order in self.orders],
            "obj_res": [resource.to_dict() for resource in self.resources],
            "products": [product.to_dict() for product in self.products],
            "timespan": len(self.times)
        }

    @classmethod
    def from_dict(cls, data):
        orders = [Order.from_dict(order) for order in data["orders"]]
        products = [Product.from_dict(prod) for prod in data["products"]]
        resources = [Resource.from_dict(res) for res in data["obj_res"]]
        return cls(orders, resources, products, data["timespan"])

    def solve_cp(self, time_limit=None):
        start_time = time.time()
        mdl = CpoModel()
        tasks = defaultdict(lambda: defaultdict(dict))

        # Following model in latex

        # create interval vars for schedule: if order is required - interval, otherwise optional interval
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    task_name = f'T: order {i.id}, product {j.id}, job {k.id}'
                    tasks[i.id][j.id][k.id] = interval_var(
                        name=task_name,
                        size=k.duration,
                        optional=(i.required != 1)
                    )

        # create inventory
        inventory = integer_var_dict(
            [(j.id, t) for j in self.products for t in self.times],
            name="inventory"
        )

        # quantity produced at timestamp t of each product
        quantity_produced = integer_var_dict(
            [(j.id, t) for j in self.products for t in self.times],
            name="quantity_produced"
        )

        # calculate whether all tasks were present in each order
        all_tasks_present = {}
        for i in self.orders:
            all_tasks_present[i.id] = (
                [presence_of(tasks[i.id][j.id][k.id]) if tasks[i.id][j.id][k.id].is_optional() else True
                 for j in i.products
                 for k in j.jobs])
        y = binary_var_dict(
            [i.id for i in self.orders],
            name="y"
        )

        for i in self.orders:
            task_presence = all_tasks_present[i.id]
            mdl.add(y[i.id] == logical_and(task_presence))

        # calculate quantity produced
        for j in self.products:
            for t in self.times:
                last_jobs = []
                for i in self.orders:
                    if j.id in tasks[i.id]:
                        end_times = [y[i.id] * end_of(task) for task in tasks[i.id][j.id].values()]
                        if end_times:
                            end_of_job = max(end_times)
                            last_jobs.append(end_of_job == t)

                mdl.add(quantity_produced[j.id, t] == sum(last_jobs))

        # inventory from the beginning is from the input
        for j in self.products:
            mdl.add(inventory[j.id, 0] == j.inventory)

        # update equality for inventory
        for j in self.products:
            for t in range(1, len(self.times)):
                mdl.add(
                    inventory[j.id, t] == inventory[j.id, t - 1] +
                    quantity_produced[j.id, t] - j.demand[t]
                )

        # inventory is never negative
        for i in self.orders:
            for j in i.products:
                for t in self.times:
                    mdl.add(inventory[j.id, t] >= 0)

        # all orders must be completed before the deadline if accepted
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.add(if_then(y[i.id] == 1, end_of(tasks[i.id][j.id][k.id]) <= i.deadline))

        # lags in time are satisfied
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    for succ in k.successors:
                        mdl.add(if_then(y[i.id] == 1, start_of(tasks[i.id][j.id][k.id]) + succ.lag <= start_of(
                            tasks[i.id][j.id][succ.id])))

        # if order is not accepted, make tasks absent
        capacities = [resource.capacity for resource in self.resources]
        for i in self.orders:
            if i.required == 0:
                for j in i.products:
                    for k in j.jobs:
                        mdl.add(if_then(y[i.id] == 0, presence_of(tasks[i.id][j.id][k.id]) == 0))

        nb_resources = len(self.resources)
        # for each resource, ensure the usage does not exceed its capacity
        for r in range(nb_resources):
            mdl.add(
                sum(
                    pulse(tasks[i.id][j.id][k.id], k.resources[r])
                    for i in self.orders
                    for j in i.products
                    for k in j.jobs
                    if k.resources[r] > 0
                ) <= capacities[r]
            )

        # objective
        mdl.add(maximize(sum(y[i.id] * i.value for i in self.orders)))
        finish_time = time.time()
        init_time = finish_time - start_time

        start_time = time.time()
        res = mdl.solve(TimeLimit=time_limit)
        end_time = time.time()

        solve_time = end_time - start_time
        data = []
        accepted_orders = {}
        if res:
            for i in self.orders:
                if res.get_var_solution(y[i.id]).value == 1:
                    accepted_orders[i.id] = True
                    for j in i.products:
                        for k in j.jobs:
                            start = res.get_var_solution(tasks[i.id][j.id][k.id]).start
                            end = res.get_var_solution(tasks[i.id][j.id][k.id]).end
                            data.append(
                                {"task": f"{i.id}_{j.id}_{k.id}", "start": start, "end": end})
                else:
                    accepted_orders[i.id] = False
            data_df = pd.DataFrame(data)
        else:
            print("No Solution Found!")
            data_df = None

        return res, accepted_orders, data_df, init_time, solve_time

    def solve_cp_with_lateness(self, time_limit=None):
        start_time = time.time()
        mdl = CpoModel()
        tasks = defaultdict(lambda: defaultdict(dict))

        # Following model in latex

        # create interval vars for schedule: if order is required - interval, otherwise optional interval
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    task_name = f'T: order {i.id}, product {j.id}, job {k.id}'
                    tasks[i.id][j.id][k.id] = interval_var(
                        name=task_name,
                        size=k.duration,
                        optional=(i.required != 1)
                    )
        lateness = integer_var_dict([i.id for i in self.orders], name="lateness")

        # create inventory
        inventory = integer_var_dict(
            [(j.id, t) for j in self.products for t in self.times],
            name="inventory"
        )

        # quantity produced at timestamp t of each product
        quantity_produced = integer_var_dict(
            [(j.id, t) for j in self.products for t in self.times],
            name="quantity_produced"
        )

        # calculate whether all tasks were present in each order
        all_tasks_present = {}
        for i in self.orders:
            all_tasks_present[i.id] = (
                [presence_of(tasks[i.id][j.id][k.id]) if tasks[i.id][j.id][k.id].is_optional() else True
                 for j in i.products
                 for k in j.jobs])
        y = binary_var_dict(
            [i.id for i in self.orders],
            name="y"
        )
        for i in self.orders:
            max_end_time = max(end_of(tasks[i.id][j.id][k.id]) for j in i.products for k in j.jobs)
            mdl.add(lateness[i.id] >= max_end_time - i.deadline)
            mdl.add(lateness[i.id] >= 0)

        for i in self.orders:
            task_presence = all_tasks_present[i.id]
            mdl.add(y[i.id] == logical_and(task_presence))

        # calculate quantity produced
        for j in self.products:
            for t in self.times:
                last_jobs = []
                for i in self.orders:
                    if j.id in tasks[i.id]:
                        end_times = [y[i.id] * end_of(task) for task in tasks[i.id][j.id].values()]
                        if end_times:
                            end_of_job = max(end_times)
                            last_jobs.append(end_of_job == t)

                mdl.add(quantity_produced[j.id, t] == sum(last_jobs))

        # inventory from the beginning is from the input
        for j in self.products:
            mdl.add(inventory[j.id, 0] == j.inventory)

        # update equality for inventory
        for j in self.products:
            for t in range(1, len(self.times)):
                mdl.add(
                    inventory[j.id, t] == inventory[j.id, t - 1] +
                    quantity_produced[j.id, t] - j.demand[t]
                )

        # inventory is never negative
        for i in self.orders:
            for j in i.products:
                for t in self.times:
                    mdl.add(inventory[j.id, t] >= 0)

        # all orders must be completed before the deadline if accepted


        # lags in time are satisfied
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    for succ in k.successors:
                        mdl.add(if_then(y[i.id] == 1, start_of(tasks[i.id][j.id][k.id]) + succ.lag <= start_of(
                            tasks[i.id][j.id][succ.id])))

        # if order is not accepted, make tasks absent
        capacities = [resource.capacity for resource in self.resources]
        for i in self.orders:
            if i.required == 0:
                for j in i.products:
                    for k in j.jobs:
                        mdl.add(if_then(y[i.id] == 0, presence_of(tasks[i.id][j.id][k.id]) == 0))

        nb_resources = len(self.resources)
        # for each resource, ensure the usage does not exceed its capacity
        for r in range(nb_resources):
            mdl.add(
                sum(
                    pulse(tasks[i.id][j.id][k.id], k.resources[r])
                    for i in self.orders
                    for j in i.products
                    for k in j.jobs
                    if k.resources[r] > 0
                ) <= capacities[r]
            )

        total_value = sum(y[i.id] * i.value for i in self.orders)
        lateness_penalty = sum(0.3 * lateness[i.id] for i in self.orders)

        mdl.add(maximize(total_value - lateness_penalty))
        # objective
        mdl.add(maximize(sum(y[i.id] * i.value for i in self.orders)))
        finish_time = time.time()
        init_time = finish_time - start_time

        start_time = time.time()
        res = mdl.solve(TimeLimit=time_limit)
        end_time = time.time()

        solve_time = end_time - start_time
        data = []
        accepted_orders = {}
        if res:
            for i in self.orders:
                if res.get_var_solution(y[i.id]).value == 1:
                    accepted_orders[i.id] = True
                    for j in i.products:
                        for k in j.jobs:
                            start = res.get_var_solution(tasks[i.id][j.id][k.id]).start
                            end = res.get_var_solution(tasks[i.id][j.id][k.id]).end
                            data.append(
                                {"task": f"{i.id}_{j.id}_{k.id}", "start": start, "end": end})
                else:
                    accepted_orders[i.id] = False
            data_df = pd.DataFrame(data)
        else:
            print("No Solution Found!")
            data_df = None

        return res, accepted_orders, data_df, init_time, solve_time

    def solve_milp_gurobi(self, time_limit=None):
        time_limit = 120
        start_init_time = time.time()

        M = 1000
        mdl = gp.Model()

        if time_limit:
            mdl.setParam(GRB.Param.TimeLimit, time_limit)

        # Decision variables
        inv = mdl.addVars(((j.id, t) for j in self.products for t in self.times), vtype=GRB.INTEGER, name="inv")
        s = mdl.addVars(((i.id, j.id, k.id, t) for i in self.orders for j in i.products
                         for k in j.jobs for t in self.times), vtype=GRB.BINARY, name="s")
        y = mdl.addVars([i.id for i in self.orders], vtype=GRB.BINARY, name="y")
        e = mdl.addVars(((i.id, j.id, t) for i in self.orders for j in i.products for t in range(len(self.times) + 1)),
                        vtype=GRB.BINARY, name="e")
        t_start = mdl.addVars(((i.id, j.id, k.id) for i in self.orders for j in i.products for k in j.jobs),
                              vtype=GRB.INTEGER, name="t_start")

        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.addConstr(
                        t_start[i.id, j.id, k.id] ==
                        gp.quicksum(t * s[i.id, j.id, k.id, t] for t in self.times)
                    )

        for i in self.orders:
            mdl.addConstr(y[i.id] >= i.required)

        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.addConstr(gp.quicksum(s[i.id, j.id, k.id, t] for t in self.times) >= y[i.id])
                    mdl.addConstr(gp.quicksum(s[i.id, j.id, k.id, t] for t in self.times) <= 1)

        for i in self.orders:
            for j in i.products:
                mdl.addConstr(gp.quicksum(e[i.id, j.id, t] for t in self.times) == y[i.id])

        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.addConstr(
                        gp.quicksum(t * e[i.id, j.id, t] for t in range(len(self.times)))
                        >= t_start[i.id, j.id, k.id] + k.duration - M * (1 - y[i.id])
                    )

        for j in self.products:
            for t in range(1, len(self.times)):
                if j.demand[t] != 0:
                    mdl.addConstr(
                        inv[j.id, t] == inv[j.id, t - 1] +
                        gp.quicksum(e[i.id, j.id, t] for i in self.orders if j in i.products) -
                        j.demand[t]
                    )

        for j in self.products:
            mdl.addConstr(inv[j.id, 0] == j.inventory)

        for j in self.products:
            for t in self.times:
                mdl.addConstr(inv[j.id, t] >= 0)

        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.addConstr(t_start[i.id, j.id, k.id] + k.duration <= i.deadline + M * (1 - y[i.id]))

        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    for succ in k.successors:
                        mdl.addConstr(t_start[i.id, j.id, succ.id] >=
                                      t_start[i.id, j.id, k.id] + succ.lag - M * (1 - y[i.id]))

        valid_combinations = {}
        for l in self.resources:
            valid_combinations[l.id] = {}
            for t in self.times:
                valid_combinations[l.id][t] = [
                    (i.id, j.id, k, tau)
                    for i in self.orders for j in i.products for k in j.jobs
                    for tau in range(max(t - k.duration + 1, 0), t + 1)
                    if k.resources[l.id] > 0
                ]

        for l in self.resources:
            for t in self.times:
                coefficients = [k.resources[l.id] for i, j, k, tau in valid_combinations[l.id][t]]
                variables = [s[i, j, k.id, tau] for i, j, k, tau in valid_combinations[l.id][t]]
                mdl.addConstr(gp.quicksum(coeff * var for coeff, var in zip(coefficients, variables)) <= l.capacity)

        mdl.setObjective(gp.quicksum(y[i.id] * i.value for i in self.orders), GRB.MAXIMIZE)

        end_init_time = time.time()
        init_time = end_init_time - start_init_time

        start_time = time.time()
        mdl.optimize()
        end_time = time.time()

        solve_time = end_time - start_time

        # output the solution
        data = []
        accepted_orders = {}
        if mdl.status == GRB.OPTIMAL:
            for i in self.orders:
                if y[i.id].X == 1:
                    accepted_orders[i.id] = True
                    for j in i.products:
                        for k in j.jobs:
                            start = t_start[i.id, j.id, k.id].X
                            end = start + k.duration
                            data.append(
                                {"task": f"{i.id}_{j.id}_{k.id}", "start": start, "end": end})
                else:
                    accepted_orders[i.id] = False
            data_df = pd.DataFrame(data)
        else:
            print("No Solution Found!")
            data_df = None

        return mdl, accepted_orders, data_df, init_time, solve_time,

    def check_obligatory_constraint(self, schedule, accepted_orders):
        scheduled_tasks = set(schedule["task"])
        for i in self.orders:
            if i.required:
                if accepted_orders.get(i.id) != 1:
                    return False
                for j in i.products:
                    for k in j.jobs:
                        task_id = f"{i.id}_{j.id}_{k.id}"
                        if task_id not in scheduled_tasks:
                            return False
        return True

    def check_deadline_constraint(self, schedule):
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    task_id = f"{i.id}_{j.id}_{k.id}"

                    task_entry = schedule[schedule["task"] == task_id]

                    if not task_entry.empty:
                        task_end = task_entry["end"].values[0]

                        if task_end > i.deadline:
                            return False

        return True

    def check_time_lag_constraint(self, schedule):
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    for succ in k.successors:
                        task_id = f"{i.id}_{j.id}_{k.id}"
                        succ_id = f"{i.id}_{j.id}_{succ.id}"

                        task_entry = schedule[schedule["task"] == task_id]
                        succ_entry = schedule[schedule["task"] == succ_id]

                        if not task_entry.empty and not succ_entry.empty:
                            task_start = task_entry["start"].values[0]
                            succ_start = succ_entry["start"].values[0]

                            if task_start + succ.lag > succ_start:
                                return False

        return True

    def check_resource_constraint(self, schedule):
        resource_capacities = {r: self.resources[r].capacity for r in range(len(self.resources))}
        resource_usage = defaultdict(lambda: defaultdict(int))
        for _, row in schedule.iterrows():
            task_id = row["task"]
            start, end = row["start"], row["end"]
            i_id, j_id, k_id = map(int, task_id.split("_"))

            job = next(k for i in self.orders if i.id == i_id
                       for j in i.products if j.id == j_id
                       for k in j.jobs if k.id == k_id)
            for t in range(int(start), int(end)):
                for r, usage in enumerate(job.resources):
                    resource_usage[r][t] += usage

        for r in resource_capacities:
            for t in resource_usage[r]:
                if resource_usage[r][t] > resource_capacities[r]:
                    return False

        return True
    def calculate_profit(self, schedule):
        total_profit = 0  # Initialize total profit

        for order in self.orders:
            for product in order.products:
                # Get all task IDs for this product
                task_ids = {f"{order.id}_{product.id}_{job.id}" for job in product.jobs}

                # Check if all jobs for this product are in schedule
                scheduled_tasks = set(schedule["task"])
                if task_ids.issubset(scheduled_tasks):  # If all jobs are scheduled
                    total_profit += product.value  # Add product value to total profit

        return total_profit  # Return the computed profit

    def minimize_res_use(self):
        mdl = CpoModel()
        tasks = defaultdict(lambda: defaultdict(dict))

        # Create interval variables for jobs
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    task_name = f'T: order {i.id}, product {j.id}, job {k.id}'
                    tasks[i.id][j.id][k.id] = interval_var(
                        name=task_name,
                        size=k.duration,
                        optional=False
                    )

        # Resource usage tracking
        resource_usage = integer_var_dict(
            [r.id for r in self.resources],
            name="resource_usage"
        )

        # Compute resource usage dynamically with `pulse()`
        for r in self.resources:
                mdl.add(sum(
                        pulse(tasks[i.id][j.id][k.id], k.resources[r.id])
                        for i in self.orders
                        for j in i.products
                        for k in j.jobs
                        if k.resources[r.id] > 0  # Avoid unnecessary terms
                    ) <= resource_usage[r.id]
                )

        # All orders must complete before their deadlines
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    mdl.add(end_of(tasks[i.id][j.id][k.id]) <= i.deadline)

        # Ensure job precedence constraints
        for i in self.orders:
            for j in i.products:
                for k in j.jobs:
                    for succ in k.successors:
                        mdl.add(start_of(tasks[i.id][j.id][k.id]) + succ.lag <= start_of(tasks[i.id][j.id][succ.id]))

        # Objective: Minimize max resource usage over time
        mdl.add(minimize(max(resource_usage[r.id] for r in self.resources)))

        res = mdl.solve(TimeLimit=60)

        data = []
        accepted_orders = {}
        if res:
            # for i in self.orders:
            #     if res.get_var_solution(y[i.id]).value == 1:
            #         accepted_orders[i.id] = True
            #         for j in i.products:
            #             for k in j.jobs:
            #                 start = res.get_var_solution(tasks[i.id][j.id][k.id]).start
            #                 end = res.get_var_solution(tasks[i.id][j.id][k.id]).end
            #                 data.append(
            #                     {"task": f"{i.id}_{j.id}_{k.id}", "start": start, "end": end})
            #     else:
            #         accepted_orders[i.id] = False
            data_df = pd.DataFrame(data)
        else:
            print("No Solution Found!")
            data_df = None

        return res, accepted_orders, data_df



