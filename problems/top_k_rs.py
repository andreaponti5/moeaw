import math

import numpy as np
from pymoo.core.problem import ElementwiseProblem


class TopKRS(ElementwiseProblem):

    def __init__(self, rating_matrix, ratings_per_item, k: int):
        self.rating_matrix = rating_matrix.to_numpy()
        self.ratings_per_item = np.array(ratings_per_item)
        self.n_user = int(self.rating_matrix.shape[0])
        self.n_items = int(self.rating_matrix.shape[1])
        n_var = self.n_user * k
        self.k = k
        self.id_max_film = int(self.rating_matrix.shape[1] - 1)

        self._user_indices = np.arange(self.n_user)[:, None]
        self._items_self_information = np.log2(self.n_user / self.ratings_per_item)
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_constr=0,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, self.id_max_film),
                         type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(x)
        x = np.reshape(x, (self.n_user, self.k))
        f1, sf1 = self.accuracy(x)
        f2, sf2 = self.accuracy(x)
        f3 = self.coverage(x)
        out["F"] = [-f1, -f2, -f3]
        out["SF1"] = sf1
        out["SF2"] = sf2

    def accuracy(self, rec_list):
        average_per_list = self.rating_matrix[self._user_indices, rec_list].mean(axis=1)
        return average_per_list.mean(), average_per_list

    def coverage(self, rec_list):
        return len(np.unique(rec_list)) / self.n_items

    def novelty(self, rec_list):
        average_per_list = self._items_self_information[rec_list].mean(axis=1)
        return average_per_list.mean(), average_per_list
