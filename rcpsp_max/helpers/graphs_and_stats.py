from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt


def resource_usage_stats(timestamp_model, schedule):
    resource_usage = defaultdict(lambda: [0] * len(timestamp_model.times))
    for t_index, t in enumerate(timestamp_model.times):
        for m in range(len(schedule)):
            if (schedule['start'][m] != None):
                if int(schedule['start'][m]) <= t < int(schedule['end'][m]):
                    i, j, k = map(int, schedule['task'][m].split('_'))
                    order = timestamp_model.orders[i]
                    for pr in order.products:
                        if pr.id == j:
                            my_pr = pr
                    for j in my_pr.jobs:
                        if j.id == k:
                            job = j
                    for succ in job.successors:
                        h = 0
                        for b in schedule['task']:
                            if b == f'{i}_{j}_{succ.id}':
                                if not int(schedule['start'][m]) + succ.lag <= int(schedule['start'][h]):
                                    print("not sat")
                                break
                            h += 1
                    res = job.resources
                    for resource_id, usage in enumerate(job.resources):
                        resource_usage[resource_id][t_index] += usage
    print(resource_usage)

def gantt_resources_graphs(timestamp_model, schedule):
    resource_capacities = []
    for res in timestamp_model.resources:
        resource_capacities.append(res.capacity)
    new_schedule_data = []
    for task in schedule:
        name = task['task']
        i, j, k = map(int, name.split('_'))

        # Find the corresponding order, product, and job
        for order in timestamp_model.orders:
            if order.id == i:
                my_order = order
                break
        for product in my_order.products:
            if product.id == j:
                my_product = product
        for job in my_product.jobs:
            if job.id == k:
                my_job = job

        # Add a row for each resource with a non-zero demand
        for ij, demand in enumerate(my_job.resources):
            if demand > 0:
                new_schedule_data.append({
                    "task": task["task"],
                    "start": task["start"],
                    "end": task["end"],
                    "resource": ij
                })
    import matplotlib.pyplot as plt
    # transform_and_plot_separate(schedule, timestamp_model.orders, timestamp_model.resources)
    time_range = range(min(task["start"] for task in schedule), max(task["end"] for task in schedule) + 1)
    resource_usage = {i: np.zeros(len(time_range)) for i in range(len(resource_capacities))}

    for task in schedule:
        name = task['task']
        i, j, k = map(int, name.split('_'))
        for order in timestamp_model.orders:
            if order.id == i:
                my_order = order
                break
        for product in my_order.products:
            if product.id == j:
                my_product = product
        for job in my_product.jobs:
            if job.id == k:
                my_job = job
        for ij, demand in enumerate(my_job.resources):
            resource_usage[ij][task["start"] - time_range.start: task["end"] - time_range.start] += demand

    fig, (ax_gantt, ax_resources) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Convert schedule to DataFrame for easier processing
    tasks = [item["task"] for item in schedule]
    start_times = [item["start"] for item in schedule]
    durations = [item["end"] - item["start"] for item in schedule]

    # Plot Gantt chart
    for i, task in enumerate(tasks):
        ax_gantt.barh(task, durations[i], left=start_times[i], color="skyblue", edgecolor="black")

    ax_gantt.set_xlabel("Time")
    ax_gantt.set_ylabel("Tasks")
    ax_gantt.set_title("Gantt Chart")
    ax_gantt.grid(True, linestyle="--", alpha=0.6)

    # Resource usage chart
    for i, usage in resource_usage.items():
        ax_resources.plot(time_range, usage, label=f"Resource {i + 1}", marker="o")

    # Plot resource capacities
    for i, capacity in enumerate(resource_capacities):
        ax_resources.axhline(y=capacity, color=f"C{i}", linestyle="--", alpha=0.7, label=f"Capacity {i + 1}")

    ax_resources.set_xlabel("Time")
    ax_resources.set_ylabel("Resource Usage")
    ax_resources.set_title("Resource Usage Chart")
    ax_resources.legend()
    ax_resources.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def general_resource_usage_graph(timestamp_model, schedule):
    schedule = schedule.to_dict('records')
    resource_capacities = []
    for res in timestamp_model.resources:
        resource_capacities.append(res.capacity)
    time_range = range(min(int(task["start"]) for task in schedule), max(int(task["end"]) for task in schedule) + 1)
    resource_usage = {i: np.zeros(len(time_range)) for i in range(len(resource_capacities))}

    for task in schedule:
        name = task['task']
        i, j, k = map(int, name.split('_'))
        for order in timestamp_model.orders:
            if order.id == i:
                my_order = order
                break
        for product in my_order.products:
            if product.id == j:
                my_product = product
        for job in my_product.jobs:
            if job.id == k:
                my_job = job
        for ij, demand in enumerate(my_job.resources):
            resource_usage[ij][int(task["start"]) - int(time_range.start): int(task["end"]) - int(time_range.start)] += demand

    fig, (ax_gantt, ax_resources) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Convert schedule to DataFrame for easier processing
    tasks = [item["task"] for item in schedule]
    start_times = [item["start"] for item in schedule]
    durations = [item["end"] - item["start"] for item in schedule]

    # Plot Gantt chart
    for i, task in enumerate(tasks):
        ax_gantt.barh(task, durations[i], left=start_times[i], color="skyblue", edgecolor="black")

    ax_gantt.set_xlabel("Time")
    ax_gantt.set_ylabel("Tasks")
    ax_gantt.set_title("Gantt Chart")
    ax_gantt.grid(True, linestyle="--", alpha=0.6)

    # Resource usage chart
    for i, usage in resource_usage.items():
        ax_resources.plot(time_range, usage, label=f"Resource {i + 1}", marker="o")

    # Plot resource capacities
    for i, capacity in enumerate(resource_capacities):
        ax_resources.axhline(y=capacity, color=f"C{i}", linestyle="--", alpha=0.7, label=f"Capacity {i + 1}")

    ax_resources.set_xlabel("Time")
    ax_resources.set_ylabel("Resource Usage")
    ax_resources.set_title("Resource Usage Chart")
    ax_resources.legend()
    ax_resources.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

def transform_and_plot_separate(schedule, orders, resources):
    fig, axes = plt.subplots(len(resources), 1, figsize=(12, 6 * len(resources)), sharex=True)
    if len(resources) == 1:
        axes = [axes]  # Ensure axes is a list for single subplot

    for num, (resource, ax) in enumerate(zip(resources, axes)):
        new_sc = []  # Transformed data for this resource
        machine_timeline = {h: 0 for h in range(resource.capacity)}  # Tracks end time for each machine

        for task in sorted(schedule, key=lambda task: task['start']):
            name = task['task']
            i, j, k = map(int, name.split('_'))

            # Find the corresponding order, product, and job
            for order in orders:
                if order.id == i:
                    my_order = order
                    break
            for product in my_order.products:
                if product.id == j:
                    my_product = product
            for job in my_product.jobs:
                if job.id == k:
                    my_job = job

            # Assign the task to the first available machine for this resource
            for _ in range(my_job.resources[num]):  # Check the resource requirement
                for h in range(resource.capacity):
                    if machine_timeline[h] <= task["start"]:  # Machine is free
                        # Assign the task to this machine
                        new_sc.append((task['task'], task['start'], task['end'], h))
                        # Update the machine's timeline
                        machine_timeline[h] = task['end']
                        break

        # Extract data for plotting
        tasks = [item[0] for item in new_sc]
        start_times = [item[1] for item in new_sc]
        end_times = [item[2] for item in new_sc]
        machines = [item[3] for item in new_sc]
        durations = [end - start for start, end in zip(start_times, end_times)]

        # Plot Gantt chart for this resource
        for i, task in enumerate(tasks):
            machine = machines[i]
            start = start_times[i]
            duration = durations[i]

            # Plot the task bar
            ax.barh(machine, duration, left=start, color="skyblue", edgecolor="black")

            # Add task name inside the bar
            ax.text(start + duration / 2, machine, task, ha="center", va="center", color="black", fontsize=10)

        # Customize the subplot
        ax.set_title(f"Resource {num + 1} (Capacity: {resource.capacity})")
        ax.set_ylabel("Machines")
        ax.set_yticks(range(resource.capacity))
        ax.set_yticklabels([f"Machine {h}" for h in range(resource.capacity)])
        ax.grid(True, linestyle="--", alpha=0.6)

    # Common X-axis label
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def transform_and_plot_separate(schedule, orders, resources):
    label_mapping = {
        "0_0_0": "o1p1j1",
        "1_1_1": "o2p2j2",
        "0_0_2": "o1p1j3",
        "0_1_1": "o1p2j2"
    }
    fig, axes = plt.subplots(len(resources), 1, figsize=(12, 6 * len(resources)), sharex=True)
    if len(resources) == 1:
        axes = [axes]  # Ensure axes is a list for single subplot

    for num, (resource, ax) in enumerate(zip(resources, axes)):
        new_sc = []  # Transformed data for this resource
        machine_timeline = {h: 0 for h in range(resource.capacity)}  # Tracks end time for each machine

        for task in sorted(schedule, key=lambda task: task['start']):
            name = task['task']
            i, j, k = map(int, name.split('_'))

            # Find the corresponding order, product, and job
            for order in orders:
                if order.id == i:
                    my_order = order
                    break
            for product in my_order.products:
                if product.id == j:
                    my_product = product
            for job in my_product.jobs:
                if job.id == k:
                    my_job = job

            # Assign the task to the first available machine for this resource
            for _ in range(my_job.resources[num]):  # Check the resource requirement
                for h in range(resource.capacity):
                    if machine_timeline[h] <= task["start"]:  # Machine is free
                        # Assign the task to this machine
                        new_sc.append((task['task'], task['start'], task['end'], h))
                        # Update the machine's timeline
                        machine_timeline[h] = task['end']
                        break

        # Extract data for plotting
        tasks = [item[0] for item in new_sc]
        start_times = [item[1] for item in new_sc]
        end_times = [item[2] for item in new_sc]
        machines = [item[3] for item in new_sc]
        durations = [end - start for start, end in zip(start_times, end_times)]

        # Plot Gantt chart for this resource
        for i, task in enumerate(tasks):
            machine = machines[i]
            start = start_times[i]
            duration = durations[i]

            # Plot the task bar
            ax.barh(machine, duration, left=start, color="skyblue", edgecolor="black")

            # Add task name inside the bar
            ax.text(start + duration / 2, machine, label_mapping[task], ha="center", va="center", color="black", fontsize=10)

        # Customize the subplot
        ax.set_title(f"Resource {num + 1} (Capacity: {resource.capacity})")
        ax.set_ylabel("Machines")
        ax.set_yticks(range(resource.capacity))
        ax.set_yticklabels([f"Machine {h}" for h in range(resource.capacity)])
        ax.grid(True, linestyle="--", alpha=0.6)

    # Common X-axis label
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()