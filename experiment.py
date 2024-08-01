import time
from typing import Tuple, Optional

import pandas as pd
import pymoo.problems.many
import pymoo.problems.multi
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from algorithms.moead import MOEAD
from algorithms.moeadw import MOEADW
from algorithms.wselection import WSelection


def get_algorithm(
        name: str,
        n_obj: int,
        n_partitions: int,
        seed: Optional[int] = None
) -> GeneticAlgorithm:
    ref_dirs = get_reference_directions(
        "uniform",
        n_obj,
        n_partitions=n_partitions,
        seed=seed
    )
    if name == "MOEAD":
        return MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            seed=seed
        )
    if name == "MOEADW":
        return MOEADW(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            seed=seed
        )
    if name == "NSGA2":
        return NSGA2(
            pop_size=len(ref_dirs),
            seed=seed
        )
    if name == "NSGA2W":
        return NSGA2(
            pop_size=len(ref_dirs),
            selection=WSelection(),
            seed=seed
        )


def get_problem(
        name: str,
        n_var: Optional[int] = None,
        n_obj: Optional[int] = None,
        difficulty: Optional[int] = None
) -> Problem:
    if ("DTLZ" in name) or ("WFG" in name):
        if (n_var is None) or (n_obj is None):
            raise ValueError("For DLTZ and WFG families, n_var and n_obj must be set!")
        return getattr(pymoo.problems.many, name)(n_var, n_obj)
    if "DASCMOP" in name:
        if difficulty is None:
            raise ValueError("For DASCMOP family, difficulty must be set!")
        return getattr(pymoo.problems.multi, name)(difficulty)
    raise NotImplementedError(f"Problem {name} not supported yet!")


def run_experiment(
        problem: Problem,
        algo: GeneticAlgorithm,
        termination: Tuple[str, float],
        seed: Optional[int] = None,
) -> pd.DataFrame:
    res = minimize(
        problem,
        algo,
        termination,
        callback=OptCallback(),
        save_history=False,
        seed=seed,
        verbose=True
    )
    res_df = pd.DataFrame(res.algorithm.callback.data)
    res_df["gen"] = range(len(res_df))
    res_df = res_df.explode(["X", "F", "CV"]).reset_index(drop=True)
    # x_data = pd.DataFrame(res_df["X"].tolist())
    # x_data.columns = [f"x_{col}" for col in x_data.columns]
    y_data = pd.DataFrame(res_df["F"].tolist())
    y_data.columns = [f"y_{col}" for col in y_data.columns]
    cv_data = pd.DataFrame(res_df["CV"].tolist())
    cv_data.columns = [f"cv_{col}" for col in cv_data.columns]
    cv_data[cv_data != 0] = 1.0
    return pd.concat([res_df.drop(columns=["X", "F", "CV"]), y_data, cv_data], axis=1)


class OptCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["X"] = []
        self.data["F"] = []
        self.data["CV"] = []
        self.data["runtime"] = []

    def notify(self, algorithm):
        self.data["X"].append(algorithm.pop.get("X").tolist())
        self.data["F"].append(algorithm.pop.get("F").tolist())
        self.data["CV"].append(algorithm.pop.get("CV").tolist())
        self.data["runtime"].append(time.perf_counter())
