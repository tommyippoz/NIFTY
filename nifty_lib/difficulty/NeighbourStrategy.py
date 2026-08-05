from enum import Enum

class NeighbourStrategy(Enum):
    """
    This is to support neighbour-based prediction strategy
    """
    MAX = 1,
    MEDIAN = 2,
    AVERAGE = 3
