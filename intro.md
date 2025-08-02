
# CLUES ABM Model

## Summary

The CLUES model is an agent-based model for simulating the evolution of an input-output system which can either be monetary or physical, single- or multi-regional. The abbreviation “CLUES” stands for “Climate-resilient and Low-carbon Unfolding Economic Scenarios”, which implies the model can be used to model the supply chain impacts of adaptation and mitigation for global environmental change and the modeled scenarios can unfold at relatively fine temporal scales (such as days).

## How to Run

To run the model interactively, in this directory, run the following command

```
    $ solara run app.py
```

## Files

* ``mat_to_python.py``: Contains the Schelling model class
*  ``example 1 basic run.py``: Contains the Schelling agent class
* ``WorldOfMatrix_GPU4.py``: Code for the interactive visualization.
* ``TestResults_ReductionInProductionCapacityExample.npz``: Notebook demonstrating how to run experiments and parameter sweeps on the model.
* ``TMRIOExample.json``:

## Results

![中国每日和累积的间接经济损失](https://github.com/WaterAI-bit/CLUES_ABM/blob/main/results/%E4%B8%AD%E5%9B%BD%E6%AF%8F%E6%97%A5%E5%92%8C%E7%B4%AF%E7%A7%AF%E7%9A%84%E9%97%B4%E6%8E%A5%E7%BB%8F%E6%B5%8E%E6%8D%9F%E5%A4%B1.jpg)

## Model

```python
print("hello world")
```


## Examples

```python
from WorldOfMatrix_GPU4 import WorldOfMatrixGPU
import json
import numpy as np

# ======================= Load and initialize model =======================
model = WorldOfMatrixGPU()
with open("MRIOExample.json") as f:
    MRIOdata = json.load(f)

# Set simulation parameters
delta_t = 1 / 365
ndays_target_default = 3
day_total = 365

# ==== Basic Input-Output variables ====
model.MRIO_R = int(MRIOdata["MRIO_R"])
model.MRIO_S = int(MRIOdata["MRIO_S"])
model.MRIO_Z = np.array(MRIOdata["MRIO_Z"])
model.MRIO_C = np.array(MRIOdata["MRIO_C"])
model.MRIO_VA = np.array(MRIOdata["MRIO_VA"]).flatten().reshape(1, -1)
model.OpenEcon = False
model.S2Sa = np.eye(model.MRIO_S)

model.delta_t = delta_t
model.ndays_Target_Default = ndays_target_default

# Set MRIO_Dist: default 3 with diagonals = 1
model.MRIO_Dist = 3 * np.ones((model.MRIO_R, model.MRIO_S), dtype=int)
np.fill_diagonal(model.MRIO_Dist, 1)
model.MRIO_Dist = model.MRIO_Dist - 1

# Water intensity: placeholder
model.AgentsP_WaterIntensity = np.ones((model.MRIO_R * model.MRIO_S,1))


# Initialize
model.initialize_variables()
model.initialize_production_agents()
model.initialize_consumption_agents()
model.initialize_transportation_agents()

# ======================= Prepare recording arrays =======================
S0_Evolution_ValueAdded_ProductionAgents = np.zeros((model.N_P, day_total))
RegionSectors2Regions = np.kron(np.eye(model.MRIO_R), np.ones((model.MRIO_S, 1)))
SS_AgentsP_VA = model.AgentsP_VA
SS_Region_VA = RegionSectors2Regions.T @ SS_AgentsP_VA
S0_ProductInNetwork_Region = np.zeros((model.MRIO_R, model.MRIO_R, day_total))
S0_ProductInNetwork_Region_Change = np.zeros_like(S0_ProductInNetwork_Region)
SS_ProductInNetwork_Region = RegionSectors2Regions.T @ model.AgentsP_ProductInP @ RegionSectors2Regions
S0_Evolution_Scarcity_RegionsProducts = np.zeros((model.MRIO_R, model.Sa, day_total))


# ======================= Run simulation =======================
for day in range(1,day_total+1):
    print(day)
    # Example shock: production loss for first month
    if 1 <= day <= 30:
        ind = np.arange(0, 20)
        model.AgentsP_Theta[ind] = 0.2
    if 100 <= day <= 130:
        ind = np.arange(49, 56)
        model.AgentsP_Theta[ind] = 0.3
    if 200 <= day <= 230:
        ind = np.arange(29, 46)
        model.AgentsP_Theta[ind] = 0.4



    # Tranportation lines:
    # Load, move, and unload goods in the transportation chains.
    # Move one step forward (creating augumented transportation lines), for P2P.
    model.AgentsT_P2P = np.hstack([np.zeros((model.nl_NetPP, 1)), model.AgentsT_P2P])
    # Calculate products loaded to each transportation lines.
    flat_P2P = model.AgentsT_P2P.ravel(order='F').copy()
    flat_P2P[model.AgentsT_P2P_StartLinInd] = model.AgentsP_ProductOutP.ravel(order='F')[model.k_NetPP]
    model.AgentsT_P2P = flat_P2P.reshape(model.AgentsT_P2P.shape, order='F')
    
    
    # Move one step forward (creating augumented transportation lines), for P2C.
    model.AgentsT_P2C = np.hstack([np.zeros((model.nl_NetPC, 1)), model.AgentsT_P2C])
    # Calculate products loaded to each transportation lines.
    flat_P2C = model.AgentsT_P2C.ravel(order='F').copy()
    flat_P2C[model.AgentsT_P2C_StartLinInd] = model.AgentsP_ProductOutC.ravel(order='F')[model.k_NetPC]  
    model.AgentsT_P2C = flat_P2C.reshape(model.AgentsT_P2C.shape, order='F')
    


    # Calculate products unloaded to each production agent from transportation lines.
    temp2 = model.AgentsP_ProductInP.T.copy()
    flat_temp = temp2.ravel(order='F').copy()
    flat_temp[model.k_NetPP] = model.AgentsT_P2P[:, -1]
    temp2 = flat_temp.reshape(temp2.shape, order='F')
    model.AgentsP_ProductInP = temp2.T.copy()
    model.AgentsT_P2P = model.AgentsT_P2P[:, :-1]
    
    
    # Calculate products unloaded to each consumption agent from transportation lines.
    temp2 = model.AgentsC_ProductInP.T.copy()
    flat_temp = temp2.ravel(order='F').copy()
    flat_temp[model.k_NetPC] =  model.AgentsT_P2C[:, -1]
    temp2 = flat_temp.reshape(temp2.shape, order='F')
    model.AgentsC_ProductInP = temp2.T.copy()
    model.AgentsT_P2C = model.AgentsT_P2C[:, :-1]
    del temp2, flat_temp




    # Agent actions
    model.agents_communicate()
    model.update_inventories()
    model.update_export_orders()
    model.update_shares()
    model.production_agents_produce()
    model.production_agents_prepare_product_out()
    model.production_agents_prepare_order_out()
    model.production_agents_adapt_to_shocks()
    model.production_agents_adapt_to_shortages()
    model.production_agents_remember()
    model.consumption_agents_consume()
    model.consumption_agents_prepare_order_out()
    model.consumption_agents_remember()

    # Record
    S0_Evolution_ValueAdded_ProductionAgents[:, day-1] = model.AgentsP_VA.flatten()
    flow = RegionSectors2Regions.T @ model.AgentsP_ProductInP @ RegionSectors2Regions
    S0_ProductInNetwork_Region[:, :, day-1] = flow
    S0_ProductInNetwork_Region_Change[:, :, day-1] = flow - SS_ProductInNetwork_Region
    S0_Evolution_Scarcity_RegionsProducts[:, :, day-1] = model.Scarcity_RegionsProducts


# ======================= Post-processing =======================
# Evolution of value-added of all Regions.
S0_Evolution_ValueAdded_Region = RegionSectors2Regions.T @ S0_Evolution_ValueAdded_ProductionAgents

# Percentage of value-added reduction of each Region-sector.
S0_LossPerc_ProductionAgents = np.zeros((model.N_P, 1))

# Select sectors with positive value-added in the beginning.
ind = (SS_AgentsP_VA > 1e-4).flatten()

# Compute % loss for those with positive initial VA
S0_LossPerc_ProductionAgents[ind] = (
    100 * (SS_AgentsP_VA[ind] - np.mean(S0_Evolution_ValueAdded_ProductionAgents[ind, :], axis=1).reshape(-1,1))
    / SS_AgentsP_VA[ind]
)

# Percentage of value-added reduction of each region.
S0_LossPerc_Region = np.zeros((model.MRIO_R, 1))

# Select regions with positive value-added in the beginning.
ind = (SS_Region_VA > 1e-4).flatten()

# Compute % loss for each region
S0_LossPerc_Region[ind] = (
    100 * (SS_Region_VA[ind] - np.mean(S0_Evolution_ValueAdded_Region[ind, :], axis=1).reshape(-1,1))
    / SS_Region_VA[ind]
)

# Average change of inter-region trade network each simulation step.
S0_ProductInNetwork_Region_Change_Mean = np.mean(S0_ProductInNetwork_Region_Change, axis=2)




# ======================= Save results =======================
# np.savez("TestResults_ReductionInProductionCapacityExample.npz",
#     S0_Evolution_ValueAdded_ProductionAgents = S0_Evolution_ValueAdded_ProductionAgents,
#     S0_Evolution_ValueAdded_Region = S0_Evolution_ValueAdded_Region,
#     S0_ProductInNetwork_Region = S0_ProductInNetwork_Region,
#     S0_ProductInNetwork_Region_Change = S0_ProductInNetwork_Region_Change,
#     S0_ProductInNetwork_Region_Change_Mean = S0_ProductInNetwork_Region_Change_Mean,
#     S0_LossPerc_ProductionAgents = S0_LossPerc_ProductionAgents,
#     S0_LossPerc_Region = S0_LossPerc_Region,
#     S0_Evolution_Scarcity_RegionsProducts = S0_Evolution_Scarcity_RegionsProducts,
#     SS_AgentsP_VA = SS_AgentsP_VA,
#     SS_Region_VA = SS_Region_VA,
#     SS_ProductInNetwork_Region = SS_ProductInNetwork_Region,
#     RegionSectors2Regions = RegionSectors2Regions
# )

# ======================= Plot =======================
import matplotlib.pyplot as plt
# 每天的总 ValueAdded：对代理维度求和（axis=0 表示按行方向）
daily_total = np.sum(S0_Evolution_ValueAdded_ProductionAgents, axis=0)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(range(1, day_total+1), daily_total, marker='o')
plt.xlabel('Day')
plt.ylabel('Total Value Added')
plt.title('Daily Total Value Added of Production Agents')
plt.grid(True)
plt.tight_layout()
plt.show()

```
