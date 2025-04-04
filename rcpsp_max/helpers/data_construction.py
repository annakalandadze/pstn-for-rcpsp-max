import json
import random

from gurobipy import GRB

from rcpsp_max.helpers.instances_generation import generate_instances

# Parse data and populate products and resources
num_of_orders = [3, 10]
num_of_products_per_order = [3, 5]
num_samples = 10  # Adjust this to your desired number of scenarios

results = []

for num_orders in num_of_orders:
    for num_pr in num_of_products_per_order:
        sample_results = []

        for sample in range(num_samples):
            # Generate instance for this sample
            time_model = generate_instances(num_orders, num_pr)


            # Solve using CP
            res1, acc1, data_df, solve_time_cp, init_time_cp = time_model.solve_cp()
            cp_result = {
                "model": "CP",
                "objective": res1.get_objective_value() if res1 else None,
                "number_of_accepted": acc1,
                "feasible": bool(res1),
                "solve_time": solve_time_cp,
                "init_time": init_time_cp
            }

            # Solve using MILP
            res2, acc2, data_df2, solve_time_milp, init_time_milp = time_model.solve_milp_gurobi()
            milp_result = {
                "model": "MILP",
                "objective": res2.ObjVal if res2.status == GRB.OPTIMAL else None,
                "number_of_accepted": acc2,
                "feasible": res2.status == GRB.OPTIMAL,
                "solve_time": solve_time_milp,
                "init_time": init_time_milp
            }

            # Store the result for this sample
            sample_results.append({
                "sample": sample,
                "cp_result": cp_result,
                "milp_result": milp_result
            })

        # Calculate averages for CP and MILP results
        avg_cp_obj = sum([s["cp_result"]["objective"] for s in sample_results if s["cp_result"]["objective"] is not None]) / num_samples
        avg_milp_obj = sum([s["milp_result"]["objective"] for s in sample_results if s["milp_result"]["objective"] is not None]) / num_samples
        avg_cp_time = sum([s["cp_result"]["solve_time"] for s in sample_results]) / num_samples
        avg_milp_time = sum([s["milp_result"]["solve_time"] for s in sample_results]) / num_samples

        # Store aggregated results
        results.append({
            "parameters": {
                "num_orders": num_orders,
                "num_products_per_order": num_pr
            },
            "average_results": {
                "cp": {
                    "average_objective": avg_cp_obj,
                    "average_solve_time": avg_cp_time
                },
                "milp": {
                    "average_objective": avg_milp_obj,
                    "average_solve_time": avg_milp_time
                }
            },
            "samples": sample_results
        })

        # Save results to a JSON file
        results_filename = f"results_orders_{num_orders}_products_{num_pr}_saa.json"
        with open(results_filename, "w") as results_file:
            json.dump(results, results_file, indent=4)


