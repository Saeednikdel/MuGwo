import os
import csv
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from skimage.draw import disk

from MuDrawPlots import MuDrawPlots



def MuFitness(x, l, transmissionRange, sensingRange, resolution,
              heightOfEnvironment, widthOfEnvironment, isDraw, isPrint, pName, coverageImportance):
    """
    Compute the fitness of a node deployment configuration.

    Parameters
    ----------
    x : np.ndarray, shape (dim,)
        Flattened positions [x1, y1, x2, y2, ...] of all nodes.
    l : int
        Current iteration number.
    transmissionRange : float
        Maximum communication distance between nodes.
    sensingRange : float
        Sensing radius of each node.
    resolution : float
        Map resolution (cells per unit).
    heightOfEnvironment : float
        Height of the environment.
    widthOfEnvironment : float
        Width of the environment.
    isDraw : int (0 or 1)
        Whether to draw and save plots.
    isPrint : int (0 or 1)
        Whether to print and log results to CSV.
    pName : str
        Base name for output files.

    Returns
    -------
    o : float
        Fitness value (lower is better for minimization).
    """

    # Restriction penalty
    penalty = heightOfEnvironment * widthOfEnvironment
    totalDistance = 0.0

    # Map dimensions to nodes (2 dimensions per node: x and y)
    n = len(x) // 2
    nodes = np.zeros((n, 2))
    for i in range(n):
        nodes[i, 0] = x[i * 2]      # x
        nodes[i, 1] = x[i * 2 + 1]  # y

    # ── Generating the topology graph ─────────────────────────────────────────
    # Distance matrix between nodes (equivalent to MATLAB's dist(nodes'))
    distanceMatrix = cdist(nodes, nodes, metric='euclidean')

    # Adjacency: 1 if within transmission range
    adjacencyMatrix = (distanceMatrix <= transmissionRange).astype(float)
    np.fill_diagonal(adjacencyMatrix, 0)

    # Build graph
    G = nx.from_numpy_array(adjacencyMatrix)

    # Connected components (equivalent to MATLAB conncomp)
    components = list(nx.connected_components(G))
    sepGraphCount = len(components)
    dc = sepGraphCount - 1  # number of disconnected parts

    # Largest component size = mode(bins) count in MATLAB
    maxDegree = max(len(c) for c in components)
    degreeRate = maxDegree / n

    # Sum of edge distances (size(G.Edges,1) in MATLAB = number of undirected edges)
    edges = list(G.edges())
    for (u, v) in edges:
        totalDistance += distanceMatrix[u, v]

    # ── Calculating sensing coverage ──────────────────────────────────────────
    # Occupancy mapping via rasterization
    h_cells = int(round(heightOfEnvironment * resolution))
    w_cells = int(round(widthOfEnvironment  * resolution))
    occ_map = np.zeros((h_cells, w_cells), dtype=np.float64)

    sensing_cells = int(round(sensingRange * resolution))

    for i in range(n):
        # x -> col, y -> row
        cx = int(round(nodes[i, 0] * resolution))
        cy = int(round(nodes[i, 1] * resolution))
        rr, cc = disk((cy, cx), sensing_cells, shape=(h_cells, w_cells))
        occ_map[rr, cc] = 1.0

    # Coverage rate
    mapSize = np.sum(coverageImportance)
    occupation = occ_map * coverageImportance
    coverageReward = np.sum(occupation)
    cr = coverageReward / mapSize if mapSize > 0 else 0.0

    # ── Fitness function (Equation 11) ────────────────────────────────────────
    # fitness = (|edges| + dc*penalty) / (a*cr + b*degreeRate + c*totalDistance) - a*cr
    a = 40
    b = 5
    c = 1
    denom = a * cr + b * degreeRate + c * totalDistance
    if denom == 0:
        fitness = float('inf')
    else:
        fitness = (len(edges) + dc * penalty) / denom - a * cr

    # ── Logging ───────────────────────────────────────────────────────────────
    if isPrint == 1:
        # Match MATLAB disp format exactly (no fixed decimal places)
        print(f"Disconnected parts:{dc}, fitness: {fitness} , Coverage Rate:{cr * 100}%")
        N = [l, cr, totalDistance, dc, maxDegree, fitness]
        os.makedirs('Results', exist_ok=True)
        results_path = f'Results/{pName}_res.csv'
        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(N)

    # ── Drawing ───────────────────────────────────────────────────────────────
    if isDraw == 1:
        MuDrawPlots(x, l, adjacencyMatrix, occupation, None, G,
                    transmissionRange, sensingRange,
                    heightOfEnvironment, widthOfEnvironment, pName, coverageRate=cr)

    return fitness


def calculate_all_fitnesses_vectorized(Positions, transmissionRange, sensingRange, 
                                       resolution, heightOfEnvironment, widthOfEnvironment, 
                                     coverageImportance):
    """
    Calculate fitness for all wolves at once using vectorized operations.
    
    Parameters
    ----------
    Positions : np.ndarray, shape (SearchAgents_no, dim)
        All wolf positions.
    l : int
        Current iteration number.
    transmissionRange : float
        Maximum communication distance between nodes.
    sensingRange : float
        Sensing radius of each node.
    resolution : float
        Map resolution (cells per unit).
    heightOfEnvironment : float
        Height of the environment.
    widthOfEnvironment : float
        Width of the environment.
    isDraw : int (0 or 1)
        Whether to draw and save plots.
    isPrint : int (0 or 1)
        Whether to print and log results to CSV.
    pName : str
        Base name for output files.
    
    Returns
    -------
    fitnesses : np.ndarray, shape (SearchAgents_no,)
        Fitness values for all wolves.
    """
    
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    n = dim // 2  # Number of nodes
    
    penalty = heightOfEnvironment * widthOfEnvironment
    
    # Reshape all positions to (SearchAgents_no, n, 2) for vectorized operations
    nodes_all = Positions.reshape(SearchAgents_no, n, 2)
    
    # Initialize arrays for results
    edge_counts = np.zeros(SearchAgents_no)
    dc_counts = np.zeros(SearchAgents_no)
    total_distances = np.zeros(SearchAgents_no)
    coverage_rates = np.zeros(SearchAgents_no)
    degree_rates = np.zeros(SearchAgents_no)
    
    # Process each wolf's configuration
    for i in range(SearchAgents_no):
        nodes = nodes_all[i]
        
        # Distance matrix between nodes
        distanceMatrix = cdist(nodes, nodes, metric='euclidean')
        
        # Adjacency: 1 if within transmission range
        adjacencyMatrix = (distanceMatrix <= transmissionRange).astype(float)
        np.fill_diagonal(adjacencyMatrix, 0)
        
        # Build graph
        G = nx.from_numpy_array(adjacencyMatrix)
        
        # Connected components
        components = list(nx.connected_components(G))
        sepGraphCount = len(components)
        dc_counts[i] = sepGraphCount - 1
        
        # Largest component size
        maxDegree = max(len(c) for c in components)
        degree_rates[i] = maxDegree / n
        
        # Sum of edge distances
        edges = list(G.edges())
        edge_counts[i] = len(edges)
        for (u, v) in edges:
            total_distances[i] += distanceMatrix[u, v]
        
        # Calculate sensing coverage
        h_cells = int(round(heightOfEnvironment * resolution))
        w_cells = int(round(widthOfEnvironment * resolution))
        occ_map = np.zeros((h_cells, w_cells), dtype=np.float64)
        
        sensing_cells = int(round(sensingRange * resolution))
        
        for j in range(n):
            cx = int(round(nodes[j, 0] * resolution))
            cy = int(round(nodes[j, 1] * resolution))
            rr, cc = disk((cy, cx), sensing_cells, shape=(h_cells, w_cells))
            occ_map[rr, cc] = 1.0
        
        # Coverage rate
        mapSize = np.sum(coverageImportance)
        occupation = occ_map * coverageImportance
        coverageReward = np.sum(occupation)
        coverage_rates[i] = coverageReward / mapSize if mapSize > 0 else 0.0
    
    # Calculate fitness for all wolves at once (vectorized)
    a = 40
    b = 5
    c = 1
    denom = a * coverage_rates + b * degree_rates + c * total_distances
    
    # Handle division by zero
    denom = np.where(denom == 0, np.inf, denom)
    
    fitnesses = (edge_counts + dc_counts * penalty) / denom - a * coverage_rates
    
    return fitnesses
