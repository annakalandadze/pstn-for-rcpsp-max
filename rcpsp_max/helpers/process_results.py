import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Directory containing the results JSON files
results_dir = "/Users/akalandadze/Desktop/Thesis/rcpsp-max-pstn/rcpsp_max/helpers"

# Initialize storage for plotting
problem_sizes = []
cp_solve_times = []
cp_init_times = []
milp_solve_times = []
milp_init_times = []

# Read all JSON files in the results directory
for filename in os.listdir(results_dir):
    if filename.endswith(".json") and "results" and "seed42" and "cp" in filename:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as file:
            results = json.load(file)
            for entry in results:
                # Extract parameters
                num_orders = entry["parameters"]["num_orders"]
                num_products_per_order = entry["parameters"]["num_products_per_order"]
                problem_size = num_orders * num_products_per_order

                # Append data for CP
                if entry["cp_result"]["feasible"]:
                    problem_sizes.append(problem_size)
                if entry["cp_result"]["feasible"]:
                    cp_solve_times.append(entry["cp_result"]["solve_time"])
                    cp_init_times.append(entry["cp_result"]["init_time"])

                # if entry["milp_result"]["feasible"]:
                #     # Append data for MILP
                #     milp_solve_times.append(entry["milp_result"]["solve_time"])
                #     milp_init_times.append(entry["milp_result"]["init_time"])

# Sort data by problem size for better plotting
sorted_indices = sorted(range(len(problem_sizes)), key=lambda i: problem_sizes[i])
problem_sizes = [problem_sizes[i] for i in sorted_indices]
cp_solve_times = [cp_solve_times[i] for i in sorted_indices]
cp_init_times = [cp_init_times[i] for i in sorted_indices]
# milp_solve_times = [milp_solve_times[i] for i in sorted_indices]
# milp_init_times = [milp_init_times[i] for i in sorted_indices]

# Plot graphs
plt.figure(figsize=(14, 7))

# Solve Time Graph
# plt.subplot(1, 2, 1)
plt.plot(problem_sizes, cp_solve_times, label="CP Solve Time")
# plt.plot(problem_sizes, milp_solve_times, label="MILP Solve Time")
plt.title("Problem Size vs Solve Time")
plt.xlabel("Problem Size (Orders x Products per Order)")
plt.ylabel("Solve Time (s)")
plt.legend()
plt.grid()

# Initialization Time Graph
# plt.subplot(1, 2, 2)
# plt.plot(problem_sizes, cp_init_times, label="CP Initialization Time")
# # plt.plot(problem_sizes, milp_init_times, label="MILP Initialization Time")
# plt.title("Problem Size vs Initialization Time")
# plt.xlabel("Problem Size (Orders x Products per Order)")
# plt.ylabel("Initialization Time (s)")
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()


resource_restrictions = []
cp_solve_times = []
cp_init_times = []
milp_solve_times = []
milp_init_times = []

# Read all JSON files in the results directory
for filename in os.listdir(results_dir):
    if filename.endswith(".json") and "results" and "seed42" and "gur" in filename:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as file:
            results = json.load(file)
            for entry in results:
                # Extract resource restriction parameter
                res_restr = entry["parameters"]["resource_restriction"]

                # Append data for CP
                resource_restrictions.append(res_restr)
                cp_solve_times.append(entry["cp_result"]["solve_time"])
                cp_init_times.append(entry["cp_result"]["init_time"])

                # Append data for MILP
                milp_solve_times.append(entry["milp_result"]["solve_time"])
                milp_init_times.append(entry["milp_result"]["init_time"])

# Sort data by resource restriction for better plotting
sorted_indices = sorted(range(len(resource_restrictions)), key=lambda i: resource_restrictions[i])
resource_restrictions = [resource_restrictions[i] for i in sorted_indices]
cp_solve_times = [cp_solve_times[i] for i in sorted_indices]
cp_init_times = [cp_init_times[i] for i in sorted_indices]
milp_solve_times = [milp_solve_times[i] for i in sorted_indices]
milp_init_times = [milp_init_times[i] for i in sorted_indices]

mean_cp_solve_times = defaultdict(list)
mean_milp_solve_times = defaultdict(list)

for res_restr, cp_time, milp_time in zip(resource_restrictions, cp_solve_times, milp_solve_times):
    mean_cp_solve_times[res_restr].append(cp_time)
    mean_milp_solve_times[res_restr].append(milp_time)

mean_cp_solve_times = {k: np.mean(v) for k, v in mean_cp_solve_times.items()}
mean_milp_solve_times = {k: np.mean(v) for k, v in mean_milp_solve_times.items()}

# Create four scatter plots
plt.figure(figsize=(14, 7))

# CP Solve Time
plt.subplot(1, 2, 1)
plt.scatter(resource_restrictions, cp_solve_times, color='blue', label="CP Solve Time")
plt.plot(list(mean_cp_solve_times.keys()), list(mean_cp_solve_times.values()), color='cyan', label="Mean CP Solve Time", linewidth=2)
plt.title("CP Solve Time vs Resource Restriction")
plt.xlabel("Resource Restriction Factor")
plt.ylabel("Solve Time (s)")
plt.grid()
plt.legend()

# MILP Solve Time
plt.subplot(1, 2, 2)
plt.scatter(resource_restrictions, milp_solve_times, color='red', label="MILP Solve Time")
plt.plot(list(mean_milp_solve_times.keys()), list(mean_milp_solve_times.values()), color='orange', label="Mean MILP Solve Time", linewidth=2)
plt.title("MILP Solve Time vs Resource Restriction")
plt.xlabel("Resource Restriction Factor")
plt.ylabel("Solve Time (s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

table_data = []
for res_restr in sorted(mean_cp_solve_times.keys()):
    table_data.append([res_restr, mean_cp_solve_times[res_restr], mean_milp_solve_times[res_restr]])

# Create the figure
fig, ax = plt.subplots(figsize=(10, 5))

# Hide axes
ax.axis('off')

# Create the table
table = ax.table(cellText=table_data,
                colLabels=["Resource Restriction", "Mean CP Solve Time (s)", "Mean MILP Solve Time (s)"],
                loc="center",
                cellLoc="center",
                colColours=["lightgrey", "lightblue", "lightblue"])

# Display the table
plt.title("Mean CP and MILP Solve Times by Resource Restriction", fontsize=14)
plt.show()
