import math
import numpy as np
import random

from itertools import permutations
from pymoo.core.selection import Selection
from pymoo.util.misc import random_permuations
from scipy.stats import wasserstein_distance


def identify_pareto(objectives):
    # Count number of items
    population_size = objectives.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


class WSelection(Selection):

    def __init__(self, cloud_key="F"):
        # Init usefull variable to optimize the computations
        super().__init__()
        self.cloud_key = cloud_key

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        # Number of random individuals needed
        n_random = n_select * 2 * n_parents
        # Get pareto indeces
        F = np.array([individual.F for individual in pop])
        PF_indeces = identify_pareto(F)
        # Get random permutations
        P = list(permutations(PF_indeces, 2))
        # Sample from pareto front
        if len(P) >= n_random / 2:
            random.shuffle(P)
            P = P[:int(n_random / 2)]
            # Convert list of tuple in list of list
            P = [list(ele) for ele in P]
        else:
            # Sample from the entire population
            # Number of permutations needed
            n_perms = math.ceil(n_random / len(pop))
            # Get random permutations
            P = random_permuations(n_perms, len(pop))[:n_random]
        # Reshape the permutations
        P = np.reshape(P, (n_select, n_parents, 2))
        # Start the tournament
        S = self.binary_tournament(pop, P)
        return S

    def binary_tournament(self, pop, P):
        if P.shape[1] != 2:
            raise ValueError("Only implemented for binary tournament!")
        if P.shape[2] != 2:
            raise ValueError("Only implemented for two parents case!")

        S = []
        for i in range(P.shape[0]):
            parents1, parents2 = P[i, 0], P[i, 1]
            # If at least one solution is infeasible
            parents1_CV = pop[parents1[0]].CV + pop[parents1[1]].CV
            parents2_CV = pop[parents2[0]].CV + pop[parents2[1]].CV
            if parents1_CV > 0.0 or parents2_CV > 0.0:
                S.append(compare(parents1, parents1_CV, parents2, parents2_CV,
                                 method='smaller_is_better', return_random_if_equal=True))
            # Both solutions are feasible
            else:
                parents1_cloud = pop[parents1].get(self.cloud_key)
                parents2_cloud = pop[parents2].get(self.cloud_key)
                parents1_ws = wasserstein_distance(parents1_cloud[0], parents1_cloud[1])
                parents2_ws = wasserstein_distance(parents2_cloud[0], parents2_cloud[1])
                # Compare the two pair of parents and get the best (the one with higher distance)
                winner = compare(parents1, parents1_ws, parents2, parents2_ws, method='larger_is_better',
                                 return_random_if_equal=True)
                S.append(winner)
        return np.array(S)


def compare(parents1, parents1_val, parents2, parents2_val, method, return_random_if_equal=False):
    if method == 'larger_is_better':
        if parents1_val > parents2_val:
            return parents1
        elif parents1_val < parents2_val:
            return parents2
        else:
            if return_random_if_equal:
                if np.random.choice([0, 1]) == 0:
                    return parents1
                else:
                    return parents2
            else:
                return None
    elif method == 'smaller_is_better':
        if parents1_val < parents2_val:
            return parents1
        elif parents1_val > parents2_val:
            return parents2
        else:
            if return_random_if_equal:
                if np.random.choice([0, 1]) == 0:
                    return parents1
                else:
                    return parents2
            else:
                return None
    else:
        raise Exception("Unknown method.")
