import numpy as np


def MuInitializer(SearchAgents_no, dim, heightOfEnvironment, widthOfEnvironment, freedomRate):
    """
    Initialize the first population of search agents.

    Parameters
    ----------
    SearchAgents_no : int
        Number of search agents (wolves).
    dim : int
        Number of dimensions (2 * number of nodes, since each node has x and y).
    heightOfEnvironment : float
        Height of the environment.
    widthOfEnvironment : float
        Width of the environment.
    freedomRate : float
        Fraction of environment range to scatter agents around the center.

    Returns
    -------
    Positions : np.ndarray, shape (SearchAgents_no, dim)
        Initial positions of search agents. Odd columns (0, 2, 4, ...) are x-coords,
        even columns (1, 3, 5, ...) are y-coords.
    """
    Positions = np.zeros((SearchAgents_no, dim))

    for i in range(SearchAgents_no):
        for j in range(0, dim, 2):
            Positions[i, j]     = widthOfEnvironment  / 2 + (2 * np.random.rand() - 1) * widthOfEnvironment  * freedomRate * 0.5  # x
            Positions[i, j + 1] = heightOfEnvironment / 2 + (2 * np.random.rand() - 1) * heightOfEnvironment * freedomRate * 0.5  # y

    return Positions


# import numpy as np


# def MuInitializer(SearchAgents_no, dim, heightOfEnvironment, widthOfEnvironment, freedomRate):
#     """
#     Initialize the first population of search agents.

#     Parameters
#     ----------
#     SearchAgents_no : int
#         Number of search agents (wolves).
#     dim : int
#         Number of dimensions (2 * number of nodes, since each node has x and y).
#     heightOfEnvironment : float
#         Height of the environment.
#     widthOfEnvironment : float
#         Width of the environment.
#     freedomRate : float
#         Fraction of environment range for random placement.

#     Returns
#     -------
#     Positions : np.ndarray, shape (SearchAgents_no, dim)
#         Initial positions of search agents. Odd columns (0, 2, 4, ...) are x-coords,
#         even columns (1, 3, 5, ...) are y-coords.
#     """
#     Positions = np.zeros((SearchAgents_no, dim))

#     for i in range(SearchAgents_no):
#         for j in range(0, dim, 2):
#             Positions[i, j]     = np.random.rand() * widthOfEnvironment   # x
#             Positions[i, j + 1] = np.random.rand() * heightOfEnvironment  # y

#     return Positions