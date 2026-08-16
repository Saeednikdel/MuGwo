import numpy as np
from MuFitness import calculate_all_fitnesses_vectorized
from MuInitializer import MuInitializer


def MuGWO(SearchAgents_no, Max_iter, dim, fitnessFnc, mutationCount,
          mutationDimensions, transmissionRange, sensingRange,
          heightOfEnvironment, widthOfEnvironment, resolution,
          initFreedom, pName, darwPlotsPer, coverageImportance):
    """
    Modified Grey Wolf Optimizer (Mu-GWO) with mutation on a subset of wolves.

    Parameters
    ----------
    SearchAgents_no : int
        Total number of search agents (wolves).
    Max_iter : int
        Maximum number of iterations.
    dim : int
        Number of dimensions (2 * number of nodes).
    fitnessFnc : callable
        Fitness function handle.
    mutationCount : int
        Number of wolves at the end of the pack that are "mutant" copies of alpha.
    mutationDimensions : int
        Maximum number of dimensions to mutate per mutant wolf per iteration.
    transmissionRange : float
        Communication range of nodes.
    sensingRange : float
        Sensing range of nodes.
    heightOfEnvironment : float
        Height of the environment.
    widthOfEnvironment : float
        Width of the environment.
    resolution : float
        Map resolution.
    initFreedom : float
        Initialization freedom rate (fraction of environment around center).
    pName : str
        Base name for output files.

    Returns
    -------
    Alpha_score : float
        Best fitness value found.
    Alpha_pos : np.ndarray, shape (dim,)
        Position (solution) corresponding to the best fitness.
    Convergence_curve : np.ndarray, shape (Max_iter,)
        Best fitness value at each iteration.
    """

    ub = heightOfEnvironment
    lb = 0.0

    # Initialize alpha, beta, delta
    Alpha_pos   = np.zeros(dim)
    Alpha_score = float('inf')   # change to -inf for maximization

    Beta_pos   = np.zeros(dim)
    Beta_score = float('inf')

    Delta_pos   = np.zeros(dim)
    Delta_score = float('inf')

    # Initialize positions of all search agents
    Positions = MuInitializer(SearchAgents_no, dim, heightOfEnvironment,
                              widthOfEnvironment, initFreedom)


    l = 0  # Loop counter
    
    # Cache frequently used values for speed
    half_dim = dim // 2
    ordinary_count = SearchAgents_no - mutationCount
    mutant_start = ordinary_count
    height = heightOfEnvironment
    width = widthOfEnvironment

    # ── Main loop ─────────────────────────────────────────────────────────────
    while l < Max_iter:

        # Calculate all fitnesses at once using vectorized function
        fitnesses = calculate_all_fitnesses_vectorized(
            Positions, transmissionRange, sensingRange,
            resolution, height, width,
            coverageImportance
        )

        # Update Alpha, Beta, Delta using vectorized sorting
        sorted_indices = np.argsort(fitnesses)
        
        # Update Alpha
        alpha_candidate_idx = sorted_indices[0]
        if fitnesses[alpha_candidate_idx] < Alpha_score:
            Alpha_score = fitnesses[alpha_candidate_idx]
            Alpha_pos = Positions[alpha_candidate_idx, :].copy()
        
        # Update Beta (find first solution bigger than Alpha but smaller than current Beta)
        beta_found = False
        for idx in sorted_indices:
            if fitnesses[idx] > Alpha_score and fitnesses[idx] < Beta_score:
                Beta_score = fitnesses[idx]
                Beta_pos = Positions[idx, :].copy()
                beta_found = True
                break
        
        # Update Delta (find first solution bigger than Beta but smaller than current Delta)
        if beta_found:
            for idx in sorted_indices:
                if fitnesses[idx] > Beta_score and fitnesses[idx] < Delta_score:
                    Delta_score = fitnesses[idx]
                    Delta_pos = Positions[idx, :].copy()
                    break

        # a decreases linearly from 2 to 0
        a = 2 - l * (2 / Max_iter)

        # ── Vectorized update for ordinary wolves ──────────────────────────
        if ordinary_count > 0:
            # Generate all random numbers at once for all ordinary wolves
            # Shape: (ordinary_count, dim) for each random matrix
            r_alpha = np.random.rand(ordinary_count, dim, 2)
            r_beta = np.random.rand(ordinary_count, dim, 2)
            r_delta = np.random.rand(ordinary_count, dim, 2)
            
            # Extract r1 and r2 for each leader
            r1_alpha = r_alpha[:, :, 0]
            r2_alpha = r_alpha[:, :, 1]
            r1_beta = r_beta[:, :, 0]
            r2_beta = r_beta[:, :, 1]
            r1_delta = r_delta[:, :, 0]
            r2_delta = r_delta[:, :, 1]
            
            # Calculate A and C coefficients (Eq. 3.3 and 3.4)
            A1 = 2 * a * r1_alpha - a
            C1 = 2 * r2_alpha
            A2 = 2 * a * r1_beta - a
            C2 = 2 * r2_beta
            A3 = 2 * a * r1_delta - a
            C3 = 2 * r2_delta
            
            # Get positions of ordinary wolves
            ordinary_positions = Positions[:ordinary_count, :]
            
            # Calculate distances and new positions (Eq. 3.5 and 3.6)
            D_alpha = np.abs(C1 * Alpha_pos - ordinary_positions)
            X1 = Alpha_pos - A1 * D_alpha
            
            D_beta = np.abs(C2 * Beta_pos - ordinary_positions)
            X2 = Beta_pos - A2 * D_beta
            
            D_delta = np.abs(C3 * Delta_pos - ordinary_positions)
            X3 = Delta_pos - A3 * D_delta
            
            # Update positions (Eq. 3.7)
            Positions[:ordinary_count, :] = (X1 + X2 + X3) / 3.0

        # ── Vectorized mutant wolf updates ─────────────────────────────────
        if mutationCount > 0:
            # Copy alpha position to all mutants
            Positions[mutant_start:, :] = Alpha_pos.copy()
            
            # Generate number of mutations for each mutant wolf
            num_mutations = np.random.randint(1, mutationDimensions + 1, size=mutationCount)
            
            # Process each mutant wolf (still need loop over mutants but operations are vectorized)
            for i in range(mutationCount):
                wolf_idx = mutant_start + i
                nm = num_mutations[i]
                
                # Randomly select which dimension pairs to mutate
                mutation_pairs = np.random.choice(half_dim, size=nm, replace=False)
                
                # Calculate indices for x and y coordinates
                x_indices = mutation_pairs * 2
                y_indices = mutation_pairs * 2 + 1
                
                # Generate random positions (vectorized)
                Positions[wolf_idx, x_indices] = np.random.rand(nm) * width
                Positions[wolf_idx, y_indices] = np.random.rand(nm) * height

        l += 1
        print(f"*************Iteration:{l} , Best:{Alpha_score}")

        # Draw plots at specified intervals
        if (l - 1) % darwPlotsPer == 0 or l == Max_iter:
            fitnessFnc(Alpha_pos, l - 1,
                       transmissionRange, sensingRange, resolution,
                       height, width,
                       1, 0, pName, coverageImportance)

        # Print/log at every iteration
        fitnessFnc(Alpha_pos, l - 1,
                   transmissionRange, sensingRange, resolution,
                   height, width,
                   0, 1, pName, coverageImportance)

    return Alpha_score, Alpha_pos