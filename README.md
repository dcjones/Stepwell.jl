
# Stepwell

Analysis of cell proximity using random walk expected hitting time.

## Installing

This is not yet a registered package, so for now, clone this repo somewhere then, run in the julia repl

```
]dev /path/to/Stepwell
```


## Usage

```
using Muon, Stepwell

# Read dataset on AnnData format
adata = readh5ad("spatial-data.h5ad")

# Build a neighborhood graph
G = CellularNeighborhoodGraph(adata)

# Define some subset of cells that we wish to measure proximity to
absorbing_cells = adata.obs.celltype .== "Tumor"

# Measure expected random walk lengths from each cell
eat = expected_absorption_time(G, absorbing_cells)

# Measure expected absorption time normalized for local cell type composition.
# `k` here determines the spatial scale at which we are normalizing for local
# cell type composition. If `k` is small we are measuring highly specific co-location,
# if `k` is large we are measuring broader co-location.
normalized_eat = local_shuffled_expected_absorption_time(G, absorbing_cells, k=100)
```


## Theory


Consider the graph $G = (V, E)$ where cells are nodes and edges are between neighboring
cells (where we are agnostic to the definition of "neighbor"). Suppose we have some subset
of the cells $U \subset V$, for example cells of a particular type. We want to measure the
overall aggregate proximity of each individual cell in $v \in V$ to $U$.

We measure aggregate proximity as the expected number of steps in a random walk
on $G$ from $v$ before some cell in $U$ is encountered. This can be estimated
efficiently using by solving a sparse linear system using off the shelf solvers in 
[LinearSolve](https://github.com/SciML/LinearSolve.jl)

We only consider (in the `local_shuffled_expected_absorption_time` function)
proximity normalized for local cell type composition to measure true
co-location, independent of incidental co-location that occurs purely to due
abundance of particular cell types. This works by first sending the cell $v$
on a $k$-step random walk before then measuring the expected hitting time.

