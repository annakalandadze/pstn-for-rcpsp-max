import pandas as pd
from docplex.cp.config import context

from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
context.solver.local.execfile = '/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer'

DIRECTORY_INSTANCES = 'rcpsp_max/data'
INSTANCE_FOLDERS = ["j10", "j20", "j30"]
INSTANCE_IDS = range(1, 271)

# include instances that are solved fastly
time_limit = 6

def construct_makespan_file(duration_factor):
    data = []
    output_file = f'deterministic_makespan.csv'
    for instance_folder in INSTANCE_FOLDERS:
        for instance_id in INSTANCE_IDS:
            rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, 1,
                                                        duration_factor)
            res, _ = rcpsp_max.solve(time_limit=time_limit, mode="Quiet")
            if res:
                data.append({"instance_folder": instance_folder, "instance_id": instance_id,
                                "time_limit": time_limit,
                                "solve_time": res.get_solve_time(),
                                "solver_status": res.get_solve_status(),
                                "makespan": res.get_objective_value()})
                print("done " + str(instance_id) + str(instance_folder))
            else:
                data.append({"instance_folder": instance_folder, "instance_id": instance_id,
                                "time_limit": time_limit,
                                "solve_time": 0,
                                "solver_status": 0,
                                "makespan": 0})
                print("done " + str(instance_id) + str(instance_folder))

    data_df = pd.DataFrame(data)

    # Save to a CSV file
    data_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


construct_makespan_file(3)