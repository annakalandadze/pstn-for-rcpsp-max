import os
import random

import numpy as np
import pandas as pd

from rcpsp_max.helpers.instances_generation import generate_instances, introduce_std
from scheduling_methods.proactive_time_model_method import run_proactive_time_offline
from scheduling_methods.pstn_method import run_pstn_offline

random.seed(42)
num_of_orders = [10]
num_of_products_per_order = [5]
num_of_samples = 1
quantile_value = [0.25]
stdd_k = [0.05]

profit_results = []

for num_orders in num_of_orders:
    for num_pr in num_of_products_per_order:
        for sample in range(num_of_samples):
            time_model = generate_instances(num_orders, num_pr)

            for quantile in quantile_value:
                for std_k in stdd_k:
                    introduce_std(time_model, std_k)
                    res, accepted_orders, data = run_proactive_time_offline(time_model, quantile)
                    if res:

                        output_location_pstn, data_dict_pstn = run_pstn_offline(data, time_model, sample, quantile,
                                                                        std_k, num_orders, num_pr, accepted_orders)
