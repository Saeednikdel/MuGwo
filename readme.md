# Mu-GWO: Mutant Grey Wolf Optimizer for Wireless Sensor Network Node Deployment

A Python implementation of the Mutant Grey Wolf Optimizer (Mu-GWO) algorithm for optimal node deployment in Wireless Sensor Networks (WSNs). This project optimizes the placement of sensor nodes to maximize coverage and connectivity while minimizing energy consumption.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Algorithm Details](#algorithm-details)
- [File Descriptions](#file-descriptions)
- [References](#references)
- [License](#license)

## 🎯 Overview

Wireless Sensor Networks consist of spatially distributed sensor nodes that monitor environmental conditions. The optimal deployment of these nodes is crucial for maximizing coverage area, maintaining network connectivity, and extending network lifetime. This project implements a modified Grey Wolf Optimizer that introduces mutation operators to enhance exploration and prevent premature convergence.

### Key Objectives
- Maximize sensor coverage area
- Ensure network connectivity
- Minimize energy consumption through optimal node placement
- Balance exploration and exploitation using mutation operators

## ✨ Features

- **Grey Wolf Optimization**: Implements the GWO algorithm with alpha, beta, and delta wolves
- **Mutation Mechanism**: Mutates a subset of wolves to maintain diversity and prevent local optima
- **Coverage Map Support**: Uses importance maps to prioritize coverage in critical areas
- **Connectivity Analysis**: Ensures nodes remain connected via communication graph analysis
- **Visualization**: Generates plots showing node deployment, coverage maps, and connectivity
- **Vectorized Operations**: Optimized for performance using NumPy
- **CSV Logging**: Records fitness and convergence data for analysis

## 🏗️ Architecture

```
MuMain.py (Main entry point)
    │
    ├── MuGWO.py (Optimizer core)
    │   ├── MuInitializer.py (Population initialization)
    │   └── MuFitness.py (Fitness calculation)
    │       ├── MuDrawPlots.py (Visualization)
    │       └── sepblockfun.py (Block operations)
    │
    └── Results & Plots directories
```

## 💻 Installation

### Prerequisites
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- NetworkX
- Pillow (PIL)
- scikit-image

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mugwo-wsn-deployment.git
cd mugwo-wsn-deployment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements File
Create `requirements.txt`:
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
networkx>=2.6.0
Pillow>=8.3.0
scikit-image>=0.18.0
```

## 🚀 Usage

### Basic Usage

```python
python MuMain.py
```

### Configuration Parameters

Edit `MuMain.py` to customize the optimization parameters:

```python
# Algorithm Parameters
SearchAgents_no = 100      # Number of wolves
mutationCount = 30         # Number of mutant wolves
mutationDimensions = 2     # Max dimensions to mutate
Max_iter = 500             # Maximum iterations

# Environment Parameters
widthOfEnvironment = 50    # Width in meters
heightOfEnvironment = 50   # Height in meters
resolution = 1             # Map resolution (cells per unit)

# Network Parameters
n = 8                      # Number of sensor nodes
sensingRange = 100         # Sensing radius
transmissionRange = 200    # Communication radius (2x sensing)

# Other Parameters
degFree = 0.3             # Initialization freedom
runs = 1                  # Number of independent runs
darwPlotsPer = 100        # Plot frequency
isCustomMap = 0           # Use custom importance map
```

### Custom Maps

To use a custom importance map:
1. Place a grayscale `.bmp` image in the `Maps/` directory
2. Set `isCustomMap = 0` (uses `Maps/geo_1.bmp` by default)
3. The map weights determine coverage priorities

## 📊 Output

### Results Directory
- `Results/{pName}_res.csv`: Fitness history with metrics
  - Iteration
  - Coverage rate
  - Total distance
  - Disconnected components
  - Maximum degree
  - Fitness value

### Plots Directory
- `Plots/{pName}_NDMap_{iteration}.png`: Node deployment visualization
  - Node positions (black dots)
  - Communication links (red lines)
  - Sensing coverage (yellow circles)
  - Transmission range (blue circles)
  - Coverage rate overlay

### Visualization Components
| Element | Description |
|---------|-------------|
| Black dots | Sensor nodes |
| Yellow circles | Sensing coverage |
| Blue circles | Transmission range |
| Red lines | Communication links |
| Background | Coverage importance map |

## 🧮 Algorithm Details

### Grey Wolf Optimization (GWO)

The algorithm simulates the social hierarchy and hunting behavior of grey wolves:

1. **Social Hierarchy**:
   - Alpha (α): Best solution
   - Beta (β): Second best
   - Delta (δ): Third best
   - Omega (ω): Remaining wolves

2. **Position Update**:
   ```
   D = |C·X_p(t) - X(t)|
   X(t+1) = X_p(t) - A·D
   ```
   Where A and C are coefficient vectors.

3. **Mutation Mechanism**:
   - Selects `mutationCount` wolves as mutants
   - Copies alpha position as base
   - Mutates `mutationDimensions` random pairs
   - Enhances exploration and prevents stagnation

### Fitness Function

The fitness function evaluates node deployments based on:
1. **Coverage Rate**: Area covered by sensors
2. **Connectivity**: Network connectivity and graph properties
3. **Energy Efficiency**: Communication distances

```
Fitness = (|E| + dc·penalty) / (a·CR + b·DR + c·TD) - a·CR
```

Where:
- `|E|`: Number of communication edges
- `dc`: Disconnected components
- `CR`: Coverage rate
- `DR`: Degree rate
- `TD`: Total edge distance
- `a, b, c`: Weighting coefficients

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `MuMain.py` | Main entry point with configuration |
| `MuGWO.py` | Core optimization algorithm |
| `MuFitness.py` | Fitness evaluation and logging |
| `MuInitializer.py` | Population initialization |
| `MuDrawPlots.py` | Visualization generation |
| `sepblockfun.py` | Separable block operations |
| `BlockMean.py` | 2D block mean calculation |

## 📚 References

- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
- Original MATLAB implementation by Jan Simon (2009-2010)
- Python adaptation with mutation mechanism for WSN deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, consider optimizing for specific sensor models and environmental conditions.