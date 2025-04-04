import math
import random
from typing import List

from rcpsp_max.entities.job import Job
from rcpsp_max.entities.order import Order
from rcpsp_max.entities.product import Product
from rcpsp_max.entities.resource import Resource
from rcpsp_max.entities.successors import Successor

random.seed(42)


## SECTION: parsing from file such as mock_small
def calculate_demand(products: List[Product], times: List[int], orders: List[Order]):
    demand = [[0 for _ in range(len(times))] for _ in range(len(products))]

    for order in orders:
        for product in order.products:
            if order.deadline < len(times):
                demand[product.id][order.deadline] += 1

    return demand


def parse_from_file(filename):
    self_orders = []
    self_products = []
    self_jobs = []
    self_resources = []
    self_timespan = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    section = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith('#'):
            i += 1
            continue

        if line.isalpha():
            section = line
            i += 1
            continue

        if section == 'ORDERS':
            order_id = int(line)
            product_list = list(map(int, lines[i + 1].strip().split()))
            deadline = int(lines[i + 2].strip())
            required = int(lines[i + 3].strip())
            value = int(lines[i + 4].strip())
            self_orders.append({
                'order_id': order_id,
                'products': product_list,
                'deadline': deadline,
                'required': required,
                'value': value
            })
            i += 5

        elif section == 'PRODUCTS':
            product_id = int(line)
            jobs = list(map(int, lines[i + 1].strip().split()))
            inventory = int(lines[i + 2].strip())
            self_products.append({
                'product_id': product_id,
                'jobs': jobs,
                'inventory': inventory
            })
            i += 3

        elif section == 'JOBS':
            job_id = int(line)
            duration = int(lines[i + 1].strip())
            successors_line = lines[i + 2].strip()
            successors = list(map(int, successors_line.split())) if successors_line != 'EMPTY' else []
            demand = list(map(int, lines[i + 3].strip().split()))
            self_jobs.append({
                'job_id': job_id,
                'duration': duration,
                'successors': successors,
                'demand': demand
            })
            i += 4

        elif section == 'RESOURCES':
            self_resources = list(map(int, line.split()))
            i += 1

        elif section == 'TIMESPAN':
            self_timespan = int(line)
            i += 1

    # convert
    resources = []
    for i, resource in enumerate(self_resources):
        resources.append(Resource(i, resource))
    jobs = []
    for j in self_jobs:
        successors = []
        for a, b in zip(j["successors"][::2], j["successors"][1::2]):
            successors.append(Successor(a, b))
        jobs.append(Job(j["job_id"], j["duration"], successors, j["demand"]))

    products = []
    for i in self_products:
        jobs_for_product = []
        for j in i["jobs"]:
            for job in jobs:
                if job.id == j:
                    jobs_for_product.append(job)
                    break
        products.append(Product(i["product_id"], jobs_for_product, 0, i["inventory"]))

    orders = []
    for i in self_orders:
        products_for_order = []
        for j in i["products"]:
            for product in products:
                if product.id == j:
                    products_for_order.append(product)
                    break
        orders.append(Order(i["order_id"], products_for_order, i["deadline"], i["required"], i["value"]))

    demand = calculate_demand(products, range(self_timespan), orders)

    for i in products:
        i.demand = demand[i.id]

    return orders, resources, products, jobs, self_timespan


def read_makespan_file(filepath):
    makespan_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = f"{parts[0]},{parts[1]}"  # Combine first two columns as the key
            makespan = int(parts[-1])  # Use the last column as the value
            if makespan != 0:
                makespan_dict[key] = makespan
    return makespan_dict


## SECTION: paring PSP file
def parse_psp_file(filepath, num, some_i):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    product_id = int(num)

    # Parse general problem information (e.g., number of jobs, resources)
    num_jobs, num_resources, *_ = map(int, lines[0].split())
    rs = []
    res_line = lines[-1].split()
    for i in range(num_resources):
        rs.append(int(res_line[i]))

    # Parse job details
    jobs = []
    for i in range(1, num_jobs + 3):
        parts = lines[i].split()
        job_id = int(parts[0])
        duration = int(lines[num_jobs + job_id + 3].split()[2])
        std = 0.3 * duration
        successors_count = int(parts[2])
        resources = [0 for _ in range(num_resources)]
        for k in range(num_resources):
            resources[k] = (int(lines[num_jobs + job_id + 3].split()[k + 3]))
        successors = []
        for j in range(successors_count):
            successor_id = int(parts[3 + j]) + some_i
            lag = int(parts[3 + successors_count + j].strip('[]'))
            successors.append(Successor(successor_id, lag))
        jobs.append(Job(job_id + some_i, duration, successors, resources, std))

    product = Product(product_id, jobs, [], 0)

    some_i += len(jobs)
    return product, rs, some_i


def construct_initial_instance(product_list, num_orders, max_products_in_order, rs):
    orders = []
    prs = []
    resources = [0, 0, 0, 0, 0]
    min_of_each = [0, 0, 0, 0, 0]
    for i in range(num_orders):
        num_products_in_order = max_products_in_order
        products_for_order = []
        for j in range(num_products_in_order):
            pr_id = random.randint(0, len(product_list) - 1)
            if not product_list[pr_id] in products_for_order:
                products_for_order.append(product_list[pr_id])
                resources = [r + rs_val for r, rs_val in zip(resources, rs[pr_id])]
                for res_i in range(5):
                    if rs[pr_id][res_i] > min_of_each[res_i]:
                        min_of_each[res_i] = rs[pr_id][res_i]
                prs.append(product_list[pr_id])
        if i == 0 or 5 < i < 9:
            order = Order(i, products_for_order, 0, 0, 0)
        else:
            order = Order(i, products_for_order, 0, 1, 0)
        orders.append(order)
    obj_res = []
    for i, resource in enumerate(resources):
        resources[i] = random.randint(int(20 * min_of_each[i]), int(20 * min_of_each[i]))
    for i, j in enumerate(resources):
        obj_res.append(Resource(i, j))
    products = set()
    for order in orders:
        products.update(order.products)

    products = list(products)
    return orders, obj_res, products


def diff_parse(filename, num):

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract the header information
    header = lines[0].strip().split()
    n_tasks = int(header[0])
    n_res = int(header[1])

    # Initialize structures
    durations = [0] * (n_tasks + 2)  # Assuming tasks are numbered from 0 to n_tasks + 1
    needs = []
    temporal_relations = [[] for _ in range(n_tasks + 2)]

    # Parse each task line
    for line in lines[1:n_tasks + 2]:
        parts = line.strip().split()
        task_id = int(parts[0])
        num_successors = int(parts[2])
        successors = parts[3: 3 + num_successors]
        lags = parts[3 + num_successors:]
        for i, suc in enumerate(successors):
            eval_lags = lags[i]
            eval_lags = eval_lags.strip('[]').split(',')
            eval_lags = [int(i) for i in eval_lags]
            for lag in eval_lags:
                temporal_relations[task_id].append((int(lag), int(suc)))

    for line in lines[n_tasks + 3:-1]:
        parts = line.strip().split()
        task_id = int(parts[0])
        duration = int(parts[2])
        durations[task_id] = duration
        resource_needs = parts[3:]
        resource_needs = [int(i) for i in resource_needs]
        needs.append(resource_needs)

    # Resource capacities and the last resource line
    capacity = list(map(int, lines[-1].strip().split()))

    jobs = []
    for i, duration in enumerate(durations):
        job_id = i
        job_duration = duration
        succ = []
        for lag, successor in temporal_relations[i]:
            suc = Successor(successor, lag)
            succ.append(suc)
        job_successors = succ
        job_resources = needs[i]
        jobs.append(Job(job_id, job_duration, job_successors, job_resources, 0))

    product = Product(num, jobs, [], 0, 0)

    return product, capacity