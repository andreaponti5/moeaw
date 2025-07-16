import numpy as np

from pymoo.core.problem import ElementwiseProblem


class OSP2(ElementwiseProblem):

    def __init__(self, impact_matrix, budget):
        self.impact_matrix = impact_matrix
        self.budget = budget

        self.impact_matrix = self.impact_matrix.drop_duplicates(subset=["node", "inj_node"])
        self.impact_matrix = self.impact_matrix.pivot(index="inj_node", columns="node", values="time").fillna(9e4)
        self.sensors = np.array([node for node in self.impact_matrix.columns])
        n_var = len(self.sensors)
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, 1),
                         type_var=int)

    def _evaluate(self, solution, out, *args, **kwargs):
        if sum(solution) == 0:
            out['F'] = [9e4, 4.5e4]
            out['SF'] = [9e4] * self.impact_matrix.shape[0]
        else:
            sensors = self.sensors[np.where(solution)[0]]
            impact_distr = self.impact_matrix[sensors].min(axis=1)
            out['SF'] = impact_distr.tolist()
            out['F'] = [np.mean(impact_distr), np.std(impact_distr)]
        out['G'] = [self.budget_constraint(solution)]

    def budget_constraint(self, sp):
        return sum(sp) - self.budget


class OSP4(ElementwiseProblem):

    def __init__(self, impact_matrix1, impact_matrix2, budget):
        self.impact_matrix1 = impact_matrix1
        self.impact_matrix2 = impact_matrix2
        self.budget = budget

        self.impact_matrix1 = self.impact_matrix1.drop_duplicates(subset=["node", "inj_node"])
        self.impact_matrix1 = self.impact_matrix1.pivot(index="inj_node", columns="node", values="time").fillna(9e4)
        self.sensors = np.array([node for node in self.impact_matrix1.columns])

        self.impact_matrix2 = self.impact_matrix2.drop_duplicates(subset=["node", "inj_node"])
        self.impact_matrix2 = self.impact_matrix2.pivot(index="inj_node", columns="node", values="volume").fillna(500)

        n_var = len(self.sensors)
        super().__init__(n_var=n_var,
                         n_obj=4,
                         n_ieq_constr=1,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, 1),
                         type_var=int)

    def _evaluate(self, solution, out, *args, **kwargs):
        if sum(solution) == 0:
            out['F'] = [9e4, 4.5e4, 500, 250]
            out['SF1'] = [9e4] * self.impact_matrix1.shape[0]
            out['SF2'] = [500] * self.impact_matrix1.shape[0]
        else:
            sensors = self.sensors[np.where(solution)[0]]
            impact_distr1 = self.impact_matrix1[sensors].min(axis=1)
            impact_distr2 = self.impact_matrix2[sensors].min(axis=1)
            out['SF1'] = impact_distr1.tolist()
            out['SF2'] = impact_distr2.tolist()
            out['F'] = [np.mean(impact_distr1), np.std(impact_distr1), np.mean(impact_distr2), np.std(impact_distr2)]
        out['G'] = [self.budget_constraint(solution)]

    def budget_constraint(self, sp):
        return sum(sp) - self.budget
