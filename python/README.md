# CLUES-ABM Python Implementation Base

This subdirectory contains the high-performance Python implementation of the **CLUES-ABM** core engine. 

---

## 📂 Subdirectory Architecture

* **`clues_abm/`**: The standard Python module library.
  * `WorldOfMatrix_GPU4.py`: Core computational matrix model governing large-system agent communications.
* **`data/`**: Input file repository.
  * `MRIOExample.json`: Sample Multi-Regional Input-Output dataset formatted for tensor bootstrapping.
* **`output/`**: Execution matrix cache.
  * `TestResults_ReductionInProductionCapacityExample.npz`: Multi-dimensional simulation state arrays.
* **`demo.ipynb`**: **[Highly Recommended]** Textbook-style interactive notebook providing complete parameter sweeps and visual charting.
* **`example_1_basic_run.py`**: A lightweight, headless batch-processing Python script optimized for server side deployments.

---

## ⚙️ Dependency & Environment Installation

Ensure your local compute environment runs Python $\ge 3.8$. 

### 1. Core Dependency Alignment
To install the foundational matrix and data data science dependencies, execute the following command within this directory:

```bash
pip install -r requirements.txt
```

### 2. Activating GPU Hardware Acceleration (Optional)
The computation engine (WorldOfMatrix_GPU4.py) uses vectorized array structures natively compatible with parallel architectures. If your node features an NVIDIA GPU environment, please install the specific cupy package matching your local CUDA toolkit version to bypass CPU bottlenecks:

```bash
# Example: For CUDA 12.x Toolkit environments
pip install cupy-cuda12x

# Example: For CUDA 11.x Toolkit environments
pip install cupy-cuda11x
```

## 🚀 Execution & Pathway Decoupling

To minimize reproduction friction, we provide two distinctive execution pathways tailored for different academic research scenarios:

* Pathway A: Interactive Exploration via Jupyter Notebook (`demo.ipynb`)
Highly recommended for Peer Reviewers and exploratory researchers. This notebook breaks down the entire simulation pipeline into atomic cells—ranging from data schema validation, exogenous shock injection, vectorized loop execution, to high-frequency visual charting. It retains pre-rendered output graphics, enabling immediate performance evaluation directly via the GitHub web interface without executing local dependencies.

* Pathway B: Headless Batch Running via Command Line (`example_1_basic_run.py`)
Tailored for cluster batch tasks and deep multi-scenario analysis. An un-blocked, lightweight Python script optimized for server-side deployments and High-Performance Computing (HPC) environments. It bypasses GUI graphic rendering and dumps multi-dimensional tracking metrics (e.g., `S0_Evolution_ValueAdded_Region`) directly into the compressed `.npz` storage in the `output/` folder.

>⚠️ Note: Please ensure the output/ directory exists locally before running the headless script.





