import argparse
import json
import os
import time

from pymoo.config import Config
from pymoo.problems.multi import DASCMOP

from experiment import get_problem, get_algorithm, run_experiment

Config.warnings["not_compiled"] = False

parser = argparse.ArgumentParser(
    prog="MOEAW",
    description="Run experiments"
)
parser.add_argument("-a", "--algorithm", type=str, default="NSGA2")
parser.add_argument("-p", "--problem", type=str, default="DLTZ2")
parser.add_argument("-t", "--trial", type=int, default=10)

args = parser.parse_args()
algorithm_name = args.algorithm
problem_name = args.problem
n_trial = args.trial

settings = json.load(open("config/settings.json", "r"))[problem_name]

for n_var, n_obj, difficulty, n_gen, n_partitions in zip(
        settings["n_var"],
        settings["n_obj"],
        settings["difficulty"],
        settings["n_gen"],
        settings["n_partitions"]
):
    problem = get_problem(problem_name, n_var, n_obj, difficulty)
    res_base_path = f".results/{problem_name}/d{n_var}_m{n_obj}"
    if isinstance(problem, DASCMOP):
        res_base_path += f"_{difficulty}"
    os.makedirs(res_base_path, exist_ok=True)
    print("+----------------------------------+")
    print(f"Algorithm: {algorithm_name}")
    print(f"Problem: {problem_name}")
    print(f"N. Var.: {n_var}")
    print(f"N. Obj.: {n_obj}")
    if isinstance(problem, DASCMOP):
        print(f"Difficulty: {difficulty}")
    print("+----------------------------------+")
    for trial in range(1, n_trial + 1):
        respath = (f"{res_base_path}/"
                   f"{problem_name}_{n_var}_{n_obj}_{algorithm_name}_{trial}.parquet")
        if isinstance(problem, DASCMOP):
            respath = (f"{res_base_path}/"
                       f"{problem_name}_{n_var}_{n_obj}_{difficulty}_{algorithm_name}_{trial}.parquet")
        if os.path.exists(respath):
            continue
        print(f"Trial {trial}/{n_trial}", end="")
        start = time.perf_counter()
        algorithm = get_algorithm(algorithm_name, n_obj=n_obj, n_partitions=n_partitions, seed=trial)
        res = run_experiment(problem, algorithm, termination=("n_gen", n_gen), seed=trial)
        res.to_parquet(respath, compression="brotli")
        print(f" [{time.perf_counter() - start:.2f} sec]")
