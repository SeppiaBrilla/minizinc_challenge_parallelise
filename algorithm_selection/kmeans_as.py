import numpy as np

portfolios = [
    [('cp-sat', 8)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 4), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 4), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('gecode', 2), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)]
]

portfolios_2025 = [
    [('cp-sat', 8)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 4), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 4), ('CPLEX', 2), ('chuffed', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('gecode', 2), ('CPLEX', 1), ('chuffed', 1), ('Picat', 1)],
]

portfolios_2024 = [
    [('cp-sat', 8)],
    [('cp-sat', 1), ('CPLEX', 1), ('CPLEX', 4), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 4), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('CPLEX', 4), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('CPLEX', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('gecode', 2), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
    [('cp-sat', 1), ('gecode', 1), ('CPLEX', 2), ('chuffed', 1), ('Picat', 1)],
]

class Kmeans_AS:
    def __init__(self, k:int) -> None:
        self.k = k

    def train(self, X:np.ndarray, portfolio_scores:np.ndarray) -> None:
        assert X.shape[0] == portfolio_scores.shape[0], f'X and the scores do not have the same shape along the first axes ({X.shape}, {portfolio_scores.shape})'
        self.X = X
        self.y = self.__portfolio_scores_to_y(portfolio_scores)

    def __portfolio_scores_to_y(self, y:np.ndarray) -> np.ndarray:
        return y.argmax(axis=1)

    def predict(self, X:np.ndarray) -> np.ndarray:
        preds = []
        for _x in X:
            dists = np.linalg.norm(self.X - _x, axis=1)
            k_dist = np.argpartition(dists, self.k)[:self.k]
            best_algs = self.y[k_dist]
            preds.append(np.bincount(best_algs).argmax())

        return np.array(preds)

if __name__ == '__main__':
    model = Kmeans_AS(5)
    X = np.array([[0, 1, 1, 2],
                  [2, 3, 3, 1],
                  [4, 5, 5, 3],
                  [7, 9, 9, 1],
                  [5, 4, 4, 9],
                  [1, 6, 6, 3],
                  [4, 2, 2, 1],
                  [3, 3, 2, 1],
                  [2, 2, 1, 5],
                  [3, 3, 1, 3]])

    y = np.array([[1,2],
                  [3,4],
                  [8,1],
                  [7,2],
                  [1,3],
                  [5,2],
                  [2,4],
                  [3,7],
                  [1,0],
                  [4,2]])

    model.train(X, y)
    test_X = np.array([[0, 1, 1, 2], [3, 3, 2, 1]])
    for pred in model.predict(test_X):
        print(pred)
