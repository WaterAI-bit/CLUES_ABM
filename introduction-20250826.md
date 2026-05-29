
# CLUES ABM Model



# 1. Overview

The CLUES ABM model (Climate-resilient & Low-carbon Unfolding Economic Scenarios – An Agent-Based Model) simulates the short-term impact of abrupt external shocks on an equilibrium economic–environmental system, containing the process of adaptation and mitigation of agents within the system. The model unfolds economic and environmental system changes over time, allowing high-temporal and spatial resolution simulations for short and medium-term periods. It couples global environmental changes with China's socio-economic system at a high spatial-temporal precision, simulating the diffusion of environmental risks, natural disasters, and policy changes. This model helps to assess economic and environmental impacts more accurately, identify risk nodes, and design adaptive policies.

Figure 1 illustrates the simulation process of the CLUES-ABM model. Within a predefined world (such as China or globally), based on the supply-demand in the industrial chain, production agents and production/consumption agents transfer information or material via transportation agents. The transmission of material and information flows is determined by the initial world and the adaptive behavior of various agents, spreading and diffusing through the supply chain network.

<p align="center">
  <img src="https://github.com/WaterAI-bit/CLUES_ABM/raw/main/figures/clues_abm_flow.png" alt="CLUES-ABM model schematic diagram" width="60%">
  <br>
  <em>Figure 1: Schematic framework for the CLUES-ABM model.</em>
</p>



**To be supplemented: 1 link points to the second library**

# 2. Function Introduction

## 2.1 Core Capabilities

The platform constructs an economic system model based on the interacting adaptive subjects in the industrial network. Through the integration of multi-source data, subject behavior rules and parallel computing technology，The platform is able to simulate the diffusion process of environmental risks, natural disasters, policy adjustments and other sudden events in the economic system at a **high spatial and temporal resolution**, and to identify risk nodes and key transmission paths. Compared with traditional models, the platform can better reflect the real adaptive behavior and complexity characteristics of economic agents, thus providing a scientific basis for policy formulation.


## 2.2 Key Application Scenarios

* **Climate Change and Disaster Response:** Forecasting the impact of typhoons, floods, droughts and other disasters on the industrial chain and regional economy, and assessing the effects of emergency dispatch and recovery programs.
* **Green Transformation and Emission Reduction Policy Evaluation:** simulate the impact of carbon tax, emission trading and other policies on industrial structure and enterprise adaptation behavior, and optimize the green transformation path.
* **Public Health and Emergency Management:** analyze the dynamic impacts of epidemics and other public emergencies on production, logistics, and trade, and formulate collaborative response strategies.
* **Macroeconomic Risk Early Warning:** Construct a "sensing system" for environmental and economic risks to support risk prevention, resilience enhancement and cross-sectoral collaborative governance.

The platform can be used as an important tool for scientific research, governmental decision-making and corporate strategic analysis, and provides support for enhancing the adaptability and sustainability of China's socio-economic systems in the face of environmental change.


## 2.3 Selected Publications

Below is a curated list of research articles demonstrating the validation, application, and methodology of the CLUES-ABM framework in leading academic journals.

<details>
<summary><b>📊 Click to expand the publication list (7 papers)</b></summary>
<br>

1. **Qi Zhou**, Shen Qu, Miaomiao Liu, Jianxun Yang, Jia Zhou, Yunlei She, Zhouyi Liu, Jun Bi. Enhancing the Efficiency of Enterprise Shutdowns for Environmental Protection: An Agent-Based Modeling Approach with High Spatial–Temporal Resolution Data, *Engineering*, 2024, 42: 295-307. [https://doi.org/10.1016/j.eng.2024.02.006](https://doi.org/10.1016/j.eng.2024.02.006)
2. Wen Wen, Yang Su, Ying-er Tang, Xingman Zhang, Yuchen Hu, Yawen Ben, Shen Qu. Evaluating carbon emissions reduction compliance based on 'dual control' policies of energy consumption and carbon emissions in China, *Journal of Environmental Management*, 2024, 367: 121990. [https://doi.org/10.1016/j.jenvman.2024.121990](https://doi.org/10.1016/j.jenvman.2024.121990)
3. Qianzi Wang, **Qi Zhou**, Jin Lin, Sen Guo, Yunlei She, Shen Qu. Risk assessment of power outages to inter-regional supply chain networks in China, *Applied Energy*, 2023, 353: 122100. [https://doi.org/10.1016/j.apenergy.2023.122100](https://doi.org/10.1016/j.apenergy.2023.122100)
4. Liping Wang, Zhouyi Liu, Yunlei She, Yiyi Cao, Mimi Gong, Meng Wang, Shen Qu. Exploring the network structure of virtual water trade among China's cities. *Journal of Environmental Management*, 2025, 388: 125968. [https://doi.org/10.1016/j.jenvman.2025.125968](https://doi.org/10.1016/j.jenvman.2025.125968)
5. Yunlei She, Jiayang Chen, **Qi Zhou**, Liping Wang, Kai Duan, Ranran Wang, Shen Qu. Evaluating losses from water scarcity and benefits of water conservation measures to intercity supply chains in China, *Environmental Science & Technology*, 2024, 58(2): 1119-1130. [https://doi.org/10.1021/acs.est.3c09015](https://doi.org/10.1021/acs.est.3c09015)
6. Yiyi Cao, Yunlei She, Qianzi Wang, Jin Lin, Weiming Chen, Shen Qu, Zhouyi Liu. Redefining virtual water allocation in China based on economic welfare gains from environmental externalities, *Journal of Cleaner Production*, 2024, 434: 140243. [https://doi.org/10.1016/j.jclepro.2023.140243](https://doi.org/10.1016/j.jclepro.2023.140243)
7. Kun Zhang, Yiyi Cao, Zhouyi Liu, **Qi Zhou**, Shen Qu, Yi-Ming Wei. Allocation of carbon emission responsibility among Chinese cities guided by economic welfare gains: Case study based on multi-regional input-output analysis, *Applied Energy*, 2024, 376: 124252. [https://doi.org/10.1016/j.apenergy.2024.124252](https://doi.org/10.1016/j.apenergy.2024.124252)

</details>


# 3. Quick Start

## 3.1 Environmental Preparation

You can choose either the **Python version** or the **MATLAB version** depending on your preference.

### 3.1.1 Python Version

* **Requirements:** Python ≥ 3.9 (Recommend 3.9–3.11)
* **Environment:** Recommended to use virtual environments (`conda` or `venv`)

The commands are as follows:

```bash
# Create a virtual environment (either one)
conda create -n clues-abm python=3.10 -y && conda activate clues-abm
# Or
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install Dependencies:

```bash
pip install solara numpy matplotlib
# If the repository has requirements.txt:
pip install -r requirements.txt
```

### 3.1.2 MATLAB Version

* **Requirements:** MATLAB R2022a or later (Recommended)
* **Required Toolboxes:** 
  * `Parallel Computing Toolbox` (for parallel acceleration)
  * `Optimization Toolbox` (optional, depending on scenario settings)

**Setup Instructions:**

1. Launch MATLAB and change the current folder to the project root directory `CLUES_ABM/`.
2. Verify that all required toolboxes are properly installed and activated.
3. *(Optional)* Run the following command in the MATLAB Command Window to check your environment and verify the setup:

```matlab
% Run this in the MATLAB Command Window to check installed toolboxes
ver
```


## 3.2 Get Code

The code is as follows:

```bash
git clone https://github.com/WaterAI-bit/CLUES_ABM.git
cd CLUES_ABM
```

## 3.3 Model Background & Core Architecture

CLUES ABM is a **Dynamic Complex Network Model** based on adaptive agents. It simulates the evolution of multi-regional economic systems under external shocks and is particularly well-suited for assessing the supply chain impacts of climate change adaptation and mitigation.

* **Behavioral Network Data**
  * Multi-regional input-output tables (e.g., `MRIOExample.json`)
  * Enterprise Network Business Database
  * Custom/User-compiled trade network data

* **Agent Types**
  * Production agent
  * Consumption agent
  * Transportation agent

* **Adaptive Behaviors**
  * Overproduction capacity utilization
  * Inventory management
  * Trade structure adaptation
  * Production process adaptation
  * Post-disaster reconstruction

* **Program Structure**
  * Object-Oriented Programming (OOP) architecture
  * Core Classes:
    * `World`
    * `AgentProduction`
    * `AgentConsumption`
    * `AgentTransportation`

* **Simulation Cycle** (e.g., 365 days a year)
  1. Update environmental and policy constraints.
  2. Connect subjects and establish interactions.
  3. Execute core agent actions:
     * **Production Agents:** Production → Product preparation → Order placement → Adaptation → Memory update
     * **Consumption Agents:** Consumption → Order placement → Memory update
  4. Advance to the next time step (day).

> **Note:** Understanding this underlying core architecture is essential for customizing the model configurations and properly setting up scenario impacts.


## 3.4 Script Running

Run the script in the project root directory:

```bash
# Note the space in the filename
python "example 1 basic run.py"
```

Script Execution Flow:

1. Load the data (e.g. `cMRIOExample.json`)
2. Set simulation parameters (time step `delta_t`, total days `day_total`, target days `ndays_target_default`)
3. Initialize agents (production, consumption, transport)
4. Injecting scenario shocks (modifying `AgentsP_Theta`to simulate capacity declines)
5. Advancing through the cycle (flow of goods → subject decisions → memory update)
6. Record results (value added changes, network flows, scarcity, etc.)
7. Plotting/saving (line graphs + `.npz` result files)

Expected Output:

* A daily Gross Value Added (GVA) line chart window will automatically pop up to visualize the economic impact.
* Optionally, save the results to `TestResults_ReductionInProductionCapacityExample.npz` for post-processing.


## 3.5 Adjustable Configurations and Variables

This section outlines the primary configuration files, adjustable simulation parameters, and core output variables available for customization.

### 3.5.1 Input Data & Model Dimensions
* **`MRIOExample.json`:** Represents the multi-regional input-output data structure. Key dimensions and matrices include:
  * `MRIO_R`: Total number of regions.
  * `MRIO_S`: Total number of economic sectors.
  * `MRIO_Z`: Intermediate demand matrix between regions and sectors.
  * `MRIO_C`: Final consumption matrix.
  * `MRIO_VA`: Value-added matrix for production nodes.

### 3.5.2 Temporal Parameters
* `delta_t`: Time step size for the simulation loop (default is `1/365`, representing a single day).
* `day_total`: Total duration of the simulation run (default is `365` days).
* `ndays_target_default`: Target days for inventory buffer or production forward-planning (default is `3`).

### 3.5.3 Scenario Shock & Policy Impact Settings
* **Capacity Shocks (`AgentsP_Theta`):** Modified directly within the example script. By setting targeted values in the capacity reduction vector/matrix over specified day intervals, users can endogenously simulate sudden drops in manufacturing or supply capacity.
* **Transportation & Logistics (`AgentsT_*`):** Controls the cargo flow propulsion velocity and unloading mechanisms. The structural mapping and connectivity between networks are managed using the indexing arrays `k_NetPP` (Production-to-Production) and `k_NetPC` (Production-to-Consumption).

### 3.5.4 Primary Output Variables
The model dynamically tracks and exports the following metrics for systemic risk assessment:
* **Change in value added at regional/agent level:** Temporal changes in gross value added at both the sub-national regional level and individual agent level.
* **Changes in cross-regional product flows:** Spatiotemporal updates in cross-regional and inter-sectoral product flows.
* **Scarcity indicators:** Supply chain bottlenecks and localized product deficits across regions.
* **Percentage loss of value added:** Cumulative percentage loss of value added relative to the baseline equilibrium state.


# 4. Output Description

## 4.1 Output Variables

1. **`S0_Evolution_ValueAdded_ProductionAgents`**: Evolution of value added by each production agent each step for one simulation period.
   - **Shape**: `(model.N_P, day_total)` where `model.N_P` represents the total number of production agents, and `day_total` is the total simulation days.

2. **`S0_ProductInNetwork_Region`**: Tracks product flow between each region every step. This variable is a 3D array where `S0_ProductInNetwork_Region[i, j, t]` represents the number of products flowing from region `i` to region `j` on day `t`.
   - **Shape**: `(model.MRIO_R, model.MRIO_R, day_total)` where `model.MRIO_R` represents the total number of regions.

3. **`S0_ProductInNetwork_Region_Change`**: Tracks changes in product flow between regions relative to the initial state `SS_ProductInNetwork_Region`. `S0_ProductInNetwork_Region_Change[i, j, t]` represents the change in product flow from region `i` to region `j` on day `t` compared to the initial state.
   - **Shape**: `(model.MRIO_R, model.MRIO_R, day_total)`

4. **`S0_Evolution_Scarcity_RegionsProducts`**: Tracks the scarcity level of each product in each region every step. `S0_Evolution_Scarcity_RegionsProducts[i, k, t]` represents the scarcity level of product `k` in region `i` on day `t`.
   - **Shape**: `(model.MRIO_R, model.Sa, day_total)` where `model.Sa` represents the total number of products.

## 4.2 Post-Processing Variables (Calculated Post-Loops)

1. **`S0_Evolution_ValueAdded_Region`**  
   Tracks the total Value Added evolution for each region every step. It is derived by aggregating `S0_Evolution_ValueAdded_ProductionAgents` by region.
   - **Shape**: `(model.MRIO_R, day_total)`  
   Where `model.MRIO_R` represents the total number of regions, and `day_total` is the total number of simulation days.

2. **`S0_LossPerc_ProductionAgents`**  
   Tracks the loss percentage in Value Added for each production agent relative to its initial state. Only production agents with initial Value Added greater than 1e-4 are considered for the loss percentage.
   - **Shape**: `(model.N_P, 1)`  
   Where `model.N_P` represents the total number of production agents.

3. **`S0_LossPerc_Region`**  
   Tracks the loss percentage in Value Added for each region relative to its initial state. Only regions with initial Value Added greater than 1e-4 are considered for the loss percentage.
   - **Shape**: `(model.MRIO_R, 1)`  
   Where `model.MRIO_R` represents the total number of regions.

4. **`S0_ProductInNetwork_Region_Change_Mean`**  
   Tracks the average product flow change between regions, calculated by averaging `S0_ProductInNetwork_Region_Change` over the time dimension.
   - **Shape**: `(model.MRIO_R, model.MRIO_R)`  
   Where `model.MRIO_R` represents the total number of regions.

> **Note**: Based on the above variables, we can deduce the total and indirect losses (i.e., total losses minus direct losses) for each production agent and each region, both daily and on average.


# 5. Model Principles

The model is an agent-based model for simulating the evolution of an input-output system which can either be monetary or physical, single- or multi-regional. The modeled scenarios can unfold at relatively fine temporal scales (such as days).

Given enough data and computing power, this agent-based model can be applied to any input-output system with many producers, consumers, and transporters. Below we assume, as in Multi-Regional Input-Output (MRIO) models, there are $r = 1, \dots, R$ regions and $s = 1, \dots, S$ sectors in each region. Each region-sector is represented by a production agent `⟨P⟩(r,s)`. The final consumption in each region is represented by a consumption agent `⟨C⟩(r)`. The transportation agent from production agents `⟨P⟩(r₁,s₁)` to `⟨P⟩(r₂,s₂)` is `⟨T⟩^(→⟨P⟩)(r₁,s₁,r₂,s₂)`, shipping the relevant intermediate product, and the transportation agent from the production agent `⟨P⟩(r₁,s₁)` to the consumption agent `⟨C⟩(r₂)` is `⟨T⟩^(→⟨C⟩)(r₁,s₁,r₂)`, shipping the final product to consumers in region `r₂`.

The specific behaviors and corresponding micro-foundations of the these agents including production agents, consumption agents, and transportation agents are detailed here. According to the classification, the behavior mechanism of each agent is introduced in detail.


## 5.1 Production agent

***Production agent*** `⟨P⟩(r,s)` carries out the production and sends products and orders to other connected agents in the supply network in each simulation step. If external shocks (such as shortage of raw materials and/or loss of production capacity) occur, they can show certain adaptive behaviors, such as replenishing inventory, adjusting the order shares of upstream suppliers, using idle production capacity, adjusting production technology, post-disaster reconstruction, etc. Below we detail each type of behavior of this agent `⟨P⟩(r,s)` in one simulation step (e.g., one day).

### 1) Producing goods using the Leontief production function

The production is based on the Leontief production function and is limited by the total order, production capacity, and raw material supply:

$$
X^{a} = \min\{O^{tot}, X^{cap}, \min\{X^{s'}\}, \min\{X^{r', s'}\}\} \quad (1)
$$


$$
X^{a} = \min\left\{O^{tot}, X^{cap}, \min\left\{X^{s'}\right\}, \min\left\{X^{r', s'}\right\}\right\}
\tag{1}
$$

$$
\begin{equation}
X^{a} = \min \left\{ O^{tot}, X^{cap}, \min \left\{ X^{s^{\prime}} \right\}, \min \left\{ X^{r^{\prime}, s^{\prime}} \right\} \right\}
\end{equation}
$$