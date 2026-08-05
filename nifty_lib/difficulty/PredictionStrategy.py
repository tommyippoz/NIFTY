from enum import Enum

class PredictionStrategy(Enum):
    """
    This is to support the choice of a prediction strategy
    """
    # Requires parameter "ml_alg"
    IID_ML = 1,
    # Requires parameter "n_neighbours" and "neigh_strategy"
    IID_KNN = 2,
    # Requires parameter "reg_strategy"
    IID_REG = 3