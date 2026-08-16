import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx


# Global state (mirrors MATLAB globals used in MuDrawPlots)
coverageImportanceMap = None
isCustomMap = 0


def MuDrawPlots(x, l, adjacencyMatrix, occupation, map_obj, G,
                transmissionRange, sensingRange,
                heightOfEnvironment, widthOfEnvironment, pName,
                coverageRate=0.0):
    """
    Draw and save visualization plots for the node deployment and coverage map.

    Parameters
    ----------
    x : np.ndarray, shape (dim,)
        Flattened positions [x1, y1, x2, y2, ...] of all nodes.
    l : int
        Current iteration number.
    adjacencyMatrix : np.ndarray, shape (n, n)
        Boolean adjacency matrix of the communication graph.
    occupation : np.ndarray
        Weighted occupancy map (coverage * importance).
    map_obj : object
        (Unused in Python version; retained for API compatibility.)
    G : networkx.Graph
        The communication graph.
    transmissionRange : float
        Radius of transmission circles.
    sensingRange : float
        Radius of sensing circles.
    heightOfEnvironment : float
        Height of the environment grid.
    widthOfEnvironment : float
        Width of the environment grid.
    pName : str
        Base name for output plot files.
    coverageRate : float
        Fraction of weighted map area covered (0–1).
    """
    global coverageImportanceMap, isCustomMap

    occupation = (255 * occupation).astype(np.uint8)
    drawGraph = False
    drawOccupancy = False

    # Map dimensions to nodes
    n = len(x) // 2
    nodes = np.zeros((n, 2))
    for i in range(n):
        nodes[i, 0] = x[i * 2]
        nodes[i, 1] = x[i * 2 + 1]

    margin = 20

    os.makedirs('Plots', exist_ok=True)

    # ── Figure 1: Node deployment map ─────────────────────────────────────────
    # MATLAB draw order (each hold on command adds a layer on top):
    # 1. scatter magenta (first, underneath everything)
    # 2. imshow rgbImage (on top of magenta scatter)
    # 3. sensing fill rectangles
    # 4. transmission ring outlines
    # 5. sensing ring outlines
    # 6. edge lines
    # 7. scatter black (topmost)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    # Step 1: magenta scatter (drawn first, underneath)
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='magenta', zorder=1)

    # Step 2: background image
    if isCustomMap == 0 and coverageImportanceMap is not None:
         # Flip the importance map vertically
        flipped_map = np.flipud(coverageImportanceMap)
        
        rgb_r = flipped_map.astype(np.float32) / 255.0
        # imadjust([0,1] -> [0.5,1]): new = 0.5 + old * 0.5
        rgb_r = 0.5 + rgb_r * 0.5
        rgb_r = np.clip(rgb_r, 0, 1)
        rgb_image = np.stack([rgb_r, rgb_r, rgb_r], axis=-1)
        # imshow with origin='lower' matches MATLAB's image orientation
        ax1.imshow(rgb_image, origin='lower',
                   extent=[0, widthOfEnvironment, 0, heightOfEnvironment],
                   aspect='auto', zorder=2)

    # Steps 3-6: sensing fill, rings, edges
    for i in range(n):
        # Sensing filled circle
        node_x = nodes[i, 0] - sensingRange
        node_y = nodes[i, 1] - sensingRange
        r = sensingRange * 2
        sensing_circle = plt.Circle(
            (nodes[i, 0], nodes[i, 1]), 
            radius=sensingRange,
            facecolor=(1, 1, 0.8, 0.6), 
            edgecolor='none', 
            zorder=3
        )
        ax1.add_patch(sensing_circle)


    for i in range(n):
        # Transmission ring outline
        tx = nodes[i, 0] - transmissionRange
        ty = nodes[i, 1] - transmissionRange
        tr = transmissionRange * 2
        trans_circle = plt.Circle(
            (nodes[i, 0], nodes[i, 1]),
            radius=transmissionRange,
            facecolor='none', 
            edgecolor=(0, 0, 1.0, 0.8),
            linewidth=0.1, 
            zorder=4
        )
        ax1.add_patch(trans_circle)

        # Sensing ring outline
        sx = nodes[i, 0] - sensingRange
        sy = nodes[i, 1] - sensingRange
        sr = sensingRange * 2
        sens_circle = plt.Circle(
            (nodes[i, 0], nodes[i, 1]),
            radius=sensingRange,
            facecolor='none', 
            edgecolor=(0.8, 0.8, 0.6),
            linewidth=0.5, 
            zorder=5
        )
        ax1.add_patch(sens_circle)


        # Communication edges
        for j in range(i + 1, n):
            if adjacencyMatrix[i, j] == 1:
                ax1.plot(
                    [nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    '-r.', zorder=5
                )

    # Step 7: black scatter on top
    ax1.scatter(nodes[:, 0], nodes[:, 1], color='black', zorder=7)

    ax1.set_ylim([-margin, heightOfEnvironment + margin])
    ax1.set_xlim([-margin, widthOfEnvironment + margin])
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.text(
        0.02, 0.98,
        f'Iteration: {l+1}\nCoverage Rate: {coverageRate * 100:.2f}%',
        transform=ax1.transAxes,
        fontsize=12,
        fontweight='bold',
        color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.65),
        zorder=8,
    )

    fig1.savefig(f'Plots/{pName}_NDMap_{l}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ── Figure 4: Occupancy / coverage map ────────────────────────────────────
    # fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))

    # # MATLAB:
    # #   occMap = imcomplement(flip(occupation,1))   -> 255 - flipud(occupation)
    # #   occMap = imadjust(occMap,[0,1],[0,0.5])     -> occMap * 0.5
    # #   imshow(255 - occMap)                        -> show 255 - (occMap*0.5)
    # occ_flipped = np.flipud(occupation)                              # flip(occupation,1)
    # occ_comp    = (255 - occ_flipped).astype(np.float32)            # imcomplement
    # occ_adj     = occ_comp / 255.0 * 0.5                            # imadjust [0,1]->[0,0.5]
    # occ_final   = np.clip(255 - occ_adj * 255, 0, 255).astype(np.uint8)  # imshow(255-occMap)

    # ax4.imshow(occ_final, cmap='gray', vmin=0, vmax=255)
    # ax4.axis('off')
    # ax4.text(
    #     0.02, 0.98,
    #     f'Iteration: {l}\nCoverage Rate: {coverageRate * 100:.2f}%',
    #     transform=ax4.transAxes,
    #     fontsize=12,
    #     fontweight='bold',
    #     color='white',
    #     verticalalignment='top',
    #     bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.65),
    #     zorder=8,
    # )

    # fig4.savefig(f'Plots/{pName}_occMap_{l}.png', dpi=300, bbox_inches='tight')
    # plt.close(fig4)

    # ── Optional: Graph visualization ─────────────────────────────────────────
    if drawGraph:
        fig2, ax2 = plt.subplots()
        pos = {i: (nodes[i, 0], nodes[i, 1]) for i in range(n)}
        nx.draw(G, pos=pos, ax=ax2)
        plt.show()
        plt.close(fig2)

    # ── Optional: Occupancy map (binary) ──────────────────────────────────────
    if drawOccupancy:
        pass  # map_obj show() not applicable in Python; skipped
