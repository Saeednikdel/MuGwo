import numpy as np
from PIL import Image
import time

import MuDrawPlots as _MuDrawPlotsModule
from sepblockfun import sepblockfun
from MuFitness   import MuFitness
from MuGWO       import MuGWO


SearchAgents_no    = 100    # Number of search agents
mutationCount = 30
mutationDimensions = 2      # Number of dimensions for mutation
Max_iter           = 500    # Maximum number of iterations

widthOfEnvironment  = 50    # Bounds of environment
heightOfEnvironment = 50
resolution          = 1
n = 8
dim = n * 2         # Number of nodes and corresponding dimensions
fitnessFnc = MuFitness      # Fitness function
sensingRange       = 100      # Range of sensor
transmissionRange  = sensingRange * 2   # Range of transmitter
degFree = 0.3
runs = 1
darwPlotsPer = 100
isCustomMap = 0


# ==============================================================================
# Global flags (shared across modules via their module-level globals)
# ==============================================================================
_MuDrawPlotsModule.isCustomMap = isCustomMap

# ==============================================================================
# Coverage importance map
# ==============================================================================
if isCustomMap == 0:
    img = Image.open('Maps/geo_1.bmp').convert('L')
    coverageImportanceMap = np.array(img, dtype=np.uint8)
    _MuDrawPlotsModule.coverageImportanceMap = coverageImportanceMap

    coverageImportance = sepblockfun(
        coverageImportanceMap.astype(np.float64) / 255.0,
        [int(1 / resolution), int(1 / resolution)],
        'mean'
    )
    # flip(coverageImportance, 1) → flip along axis 0
    coverageImportance = np.flipud(coverageImportance)

    widthOfEnvironment  = coverageImportance.shape[1]
    heightOfEnvironment = coverageImportance.shape[0]
else:
    coverageImportance = np.ones(
        (int(heightOfEnvironment * resolution), int(widthOfEnvironment * resolution))
    )


start_time_total = time.time()

for i in range(1, runs + 1):
    pName = f'mugwo_{sensingRange}_{transmissionRange}_{n}_{i}'
    
    start_time_run = time.time()
    MuGWO(SearchAgents_no, Max_iter, dim, fitnessFnc, mutationCount,
          mutationDimensions, transmissionRange, sensingRange,
          heightOfEnvironment, widthOfEnvironment, resolution,
          degFree, pName, darwPlotsPer, coverageImportance)
    end_time_run = time.time()
    
    print(f'GA Run {i} completed in: {end_time_run - start_time_run:.2f} seconds')

end_time_total = time.time()
print(f'\nTotal GA execution time: {end_time_total - start_time_total:.2f} seconds')
print(f'Average time per run: {(end_time_total - start_time_total)/runs:.2f} seconds')