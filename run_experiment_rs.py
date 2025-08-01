import argparse
import json
import os
import time

import pandas as pd
from pymoo.config import Config

from experiment import run_experiment, get_algorithm_rs
from problems.top_k_rs import TopKRS

Config.warnings["not_compiled"] = False

parser = argparse.ArgumentParser(
    prog="MOEAW",
    description="Run experiments"
)
parser.add_argument("-a", "--algorithm", type=str, default="MOEAD")
parser.add_argument("-p", "--problem", type=str, default="MovieLens1k")
parser.add_argument("-t", "--trial", type=int, default=10)

args = parser.parse_args()
algorithm_name = args.algorithm
problem_name = args.problem
n_trial = args.trial

settings = json.load(open("config/settings_rs.json", "r"))[problem_name]
rating_matrix = pd.read_csv(f".data/rs/{problem_name}_complete.csv")
ratings_per_item = json.load(open(f".data/rs/{problem_name}_ratings_per_item.json", "r"))

for n_obj, n_gen, n_partitions, k in zip(
        settings["n_obj"],
        settings["n_gen"],
        settings["n_partitions"],
        settings["k"],
):
    problem = TopKRS(rating_matrix, ratings_per_item, k=k)
    res_base_path = f".results/{problem_name}/d{problem.n_var}_m{problem.n_obj}_k{k}"
    os.makedirs(res_base_path, exist_ok=True)
    print("+----------------------------------+")
    print(f"Algorithm: {algorithm_name}")
    print(f"Problem: {problem_name}")
    print(f"N. Obj.: {n_obj}")
    print(f"K: {k}")
    print("+----------------------------------+")
    for trial in range(1, n_trial + 1):
        respath = (f"{res_base_path}/"
                   f"{problem_name}_{problem.n_var}_{problem.n_obj}_{k}_{algorithm_name}_{trial}.parquet")
        if os.path.exists(respath):
            continue
        print(f"Trial {trial}/{n_trial}")
        start = time.perf_counter()
        algorithm = get_algorithm_rs(algorithm_name, n_obj=n_obj, n_partitions=n_partitions, seed=trial)
        res = run_experiment(problem, algorithm, termination=("n_gen", n_gen), seed=trial)
        res.to_parquet(respath, compression="brotli")
        print(f" [{time.perf_counter() - start:.2f} sec]")
