import math
import random

from rcpsp_max.helpers.data_generation import read_makespan_file, construct_initial_instance, diff_parse
from rcpsp_max.solvers.timestamp_model import TimestampedModel

MAKESPAN_FILE_PATH = "/Users/akalandadze/Desktop/Thesis/rcpsp-max-pstn/aaai25/experiments/data_files" \
                     "/deterministic_makespan.csv"
INSTANCES = ['j10']
random.seed(42)
range_products = range(1, 271)


def get_products():
    makespan = read_makespan_file(MAKESPAN_FILE_PATH)
    num = 0
    products = []
    resources = []
    for instance in INSTANCES:
        for i in range_products:
            key = f"{instance},{i}"
            if key in makespan:
                product, rs = diff_parse(
                    f'/Users/akalandadze/Desktop/Thesis/rcpsp-max-pstn/rcpsp_max/data/{instance}/PSP{i}.SCH',
                    num)
                num += 1
                product.min_makespan = makespan.get(key)
                product.value = random.randint(0, 100)
                products.append(product)
                resources.append(rs)
    return products, resources


def generate_instances(num_of_orders, num_of_products_per_order):
    products, resources = get_products()
    orders, obj_res, product_list = construct_initial_instance(products, num_of_orders, num_of_products_per_order,
                                                               resources)

    for order in orders:
        min_deadline = max(product.min_makespan for product in order.products)
        min_deadline = random.randint(min_deadline + int(10 * min_deadline), min_deadline + int(20 * min_deadline))
        order.deadline = int(min_deadline)

        for o_product in order.products:
            order.value += o_product.value

    timespan = max(order.deadline for order in orders) + 1

    for product in product_list:
        product.demand = [0 for _ in range(timespan)]

    for order in orders:
        if order.required == 1:
            for product in order.products:
                product.demand[order.deadline] += 1

    time_model = TimestampedModel(orders, obj_res, product_list, timespan)

    # time_model.minimize_res_use()

    return time_model


def introduce_std(time_mode: TimestampedModel, k):
    for order in time_mode.orders:
        for product in order.products:
            for job in product.jobs:
                job.std = k * math.sqrt(job.duration)
