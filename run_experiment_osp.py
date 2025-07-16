import argparse
import json
import os
import time

import pandas as pd
from pymoo.config import Config

from problems.osp import OSP2, OSP4
from experiment import get_algorithm_osp, run_experiment

Config.warnings["not_compiled"] = False

parser = argparse.ArgumentParser(
    prog="MOEAW",
    description="Run experiments"
)
parser.add_argument("-a", "--algorithm", type=str, default="NSGA2W")
parser.add_argument("-p", "--problem", type=str, default="Hanoi")
parser.add_argument("-t", "--trial", type=int, default=10)

args = parser.parse_args()
algorithm_name = args.algorithm
problem_name = args.problem
n_trial = args.trial

settings = json.load(open("config/settings_osp.json", "r"))[problem_name]
impact_matrix1 = pd.read_csv(f".data/osp/{problem_name}_det_times.csv",
                             dtype={"sensor": "str", "scenario": "str", "time": "float"})
impact_matrix2 = pd.read_csv(f".data/osp/{problem_name}_vol_contam.csv",
                             dtype={"sensor": "str", "scenario": "str", "time": "float"})

for n_var, n_obj, n_gen, n_partitions, budget in zip(
        settings["n_var"],
        settings["n_obj"],
        settings["n_gen"],
        settings["n_partitions"],
        settings["budget"]
):
    if n_obj == 2:
        problem = OSP2(impact_matrix=impact_matrix1, budget=budget)
    else:
        problem = OSP4(impact_matrix1=impact_matrix1, impact_matrix2=impact_matrix2, budget=budget)
    res_base_path = f".results/{problem_name}/d{n_var}_m{n_obj}_b{budget}"
    os.makedirs(res_base_path, exist_ok=True)
    print("+----------------------------------+")
    print(f"Algorithm: {algorithm_name}")
    print(f"Problem: {problem_name}")
    print(f"N. Var.: {n_var}")
    print(f"N. Obj.: {n_obj}")
    print(f"Budget: {budget}")
    print("+----------------------------------+")
    for trial in range(1, n_trial + 1):
        respath = (f"{res_base_path}/"
                   f"{problem_name}_{n_var}_{n_obj}_{budget}_{algorithm_name}_{trial}.parquet")
        if os.path.exists(respath):
            continue
        print(f"Trial {trial}/{n_trial}")
        start = time.perf_counter()
        algorithm = get_algorithm_osp(algorithm_name, n_obj=n_obj, n_partitions=n_partitions, seed=trial)
        res = run_experiment(problem, algorithm, termination=("n_gen", n_gen), seed=trial)
        res.to_parquet(respath, compression="brotli")
        print(f" [{time.perf_counter() - start:.2f} sec]")
