
# CLUES ABM Model



# 1. Overview

The CLUES ABM model (Climate-resilient & Low-carbon Unfolding Economic Scenarios – An Agent-Based Model) simulates the short-term impact of abrupt external shocks on an equilibrium economic–environmental system, containing the process of adaptation and mitigation of agents within the system. The model unfolds economic and environmental system changes over time, allowing high-temporal and spatial resolution simulations for short and medium-term periods. It couples global environmental changes with China's socio-economic system at a high spatial-temporal precision, simulating the diffusion of environmental risks, natural disasters, and policy changes. This model helps to assess economic and environmental impacts more accurately, identify risk nodes, and design adaptive policies.

Figure 1 illustrates the simulation process of the CLUES-ABM model. Within a predefined world (such as China or globally), based on the supply-demand in the industrial chain, production agents and production/consumption agents transfer information or material via transportation agents. The transmission of material and information flows is determined by the initial world and the adaptive behavior of various agents, spreading and diffusing through the supply chain network.
![CLUES-ABM model schematic diagram](https://i-blog.csdnimg.cn/direct/76f743b097f04ca3b764e26f00b6c936.gif#pic_center)
**To be supplemented: 1 link points to the second library**

# 2. Function Introduction
The platform constructs an economic system model based on the interacting adaptive subjects in the industrial network. Through the integration of multi-source data, subject behavior rules and parallel computing technology，The platform is able to simulate the diffusion process of environmental risks, natural disasters, policy adjustments and other sudden events** in the economic system at a **high spatial and temporal resolution**, and to identify risk nodes and key transmission paths. Compared with traditional models, the platform can better reflect the real adaptive behavior and complexity characteristics of economic agents, thus providing a scientific basis for policy formulation.


**Application**：

(1) **Climate Change and Disaster Response** - Forecasting the impact of typhoons, floods, droughts and other disasters on the industrial chain and regional economy, and assessing the effects of emergency dispatch and recovery programs.<br>
(2) **Green Transformation and Emission Reduction Policy Evaluation** - simulate the impact of carbon tax, emission trading and other policies on industrial structure and enterprise adaptation behavior, and optimize the green transformation path.<br>
(3) **Public Health and Emergency Management** - analyze the dynamic impacts of epidemics and other public emergencies on production, logistics, and trade, and formulate collaborative response strategies.<br>
(4) **Macroeconomic Risk Early Warning** - Construct a "sensing system" for environmental and economic risks to support risk prevention, resilience enhancement and cross-sectoral collaborative governance.<br>

The platform can be used as an important tool for scientific research, governmental decision-making and corporate strategic analysis, and provides support for enhancing the adaptability and sustainability of China's socio-economic systems in the face of environmental change.

**Publications**：<br>
1，Qi Zhou, Shen Qu, Miaomiao Liu, Jianxun Yang, Jia Zhou, Yunlei She, Zhouyi Liu, Jun Bi, Enhancing the Efficiency of Enterprise Shutdowns for Environmental Protection: An Agent-Based Modeling Approach with High Spatial–Temporal Resolution Data, *Engineering*, **https://doi.org/10.1016/j.eng.2024.02.006**<br>
2，Wen Wen, Yang Su, Ying-er Tang, Xingman Zhang, Yuchen Hu, Yawen Ben, Shen Qu, Evaluating carbon emissions reduction compliance based on 'dual control' policies of energy consumption and carbon emissions in China, *Journal of Environmental Management*, **https://doi.org/10.1016/j.jenvman.2024.121990**.<br>
3，Qianzi Wang, Qi Zhou, Jin Lin, Sen Guo, Yunlei She, Shen Qu,
Risk assessment of power outages to inter-regional supply chain networks in China, *Applied Energy*, **https://doi.org/10.1016/j.apenergy.2023.122100**.<br>
4，Liping Wang, Zhouyi Liu, Yunlei She, Yiyi Cao, Mimi Gong, Meng Wang, Shen Qu. Exploring the network structure of virtual water trade among China's cities. *Journal of Environmental Management* 2025, 388 , 125968. https://doi.org/10.1016/j.jenvman.2025.125968 <br>
5，Y She, J Chen, Q Zhou, L Wang, K Duan, R Wang , Evaluating losses from water scarcity and benefits of water conservation measures to intercity supply chains in China，*Environmental science & technology*, 2024<br>
6，Yiyi Cao, Yunlei She, Qianzi Wang, Jin Lin, Weiming Chen, Shen Qu, Zhouyi Liu,
Redefining virtual water allocation in China based on economic welfare gains from environmental externalities, *Journal of Cleaner Production*, **https://doi.org/10.1016/j.jclepro.2023.140243**.<br>
7，Kun Zhang, Yiyi Cao, Zhouyi Liu, Qi Zhou, Shen Qu, Yi-Ming Wei,
Allocation of carbon emission responsibility among Chinese cities guided by economic welfare gains: Case study based on multi-regional input-output analysis, Applied *Energy*, **https://doi.org/10.1016/j.apenergy.2024.124252.**<br>
...

# 3. Quick Start
## 3.1 Environmental Preparation
  •	  Python ≥ 3.9（Recommend 3.9–3.11）
  •	  Recommended for virtual environments（```conda```or ```venv```）

The code is as follows：

```c
# Create a virtual environment (either one)
conda create -n clues-abm python=3.10 -y && conda activate clues-abm
# Or
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
安装依赖：
```c
pip install solara numpy matplotlib
#If the repository has requirements.txt：
pip install -r requirements.txt
```
## 3.2 Get Code
The code is as follows：
```c
git clone https://github.com/WaterAI-bit/CLUES_ABM.git
cd CLUES_ABM
```
## 3.3 Model Background & Core Architecture
CLUES ABM is a **Dynamic Complex Network Model** based on adaptive agents that simulates the evolution of multiregional economic systems under external shocks, and is particularly suitable for assessing supply chain impacts of climate change adaptation and mitigation.

 - **Behavioral  network data**<br>
&emsp; &emsp; Multi-regional input-output tables（eg. ```MRIOExample.json```）<br>
&emsp; &emsp; Enterprise Network Business Database<br>
&emsp; &emsp; Trade network data compiled by individuals<br>

 - **Subject type**<br>
 &emsp; &emsp; Production node (enterprise)<br>
 &emsp; &emsp; Consumption node<br>
 &emsp; &emsp; Transport node<br>

 - **Adaptive behavior**<br>
&emsp; &emsp; Overproduction Capacity<br>
&emsp; &emsp; Inventories<br>
&emsp; &emsp; Trade Adaptation<br>
&emsp; &emsp; Production Adaptation<br>
&emsp; &emsp; Reconstruction<br>
 - **Program structure**<br>
&emsp; &emsp; object-oriented programming（OOP）<br>
&emsp; &emsp; core category：<br>
&emsp; &emsp; &emsp; &emsp;	World<br>
&emsp; &emsp; &emsp; &emsp;	AgentProduction<br>
&emsp; &emsp; &emsp; &emsp;	AgentConsumption<br>
&emsp; &emsp; &emsp; &emsp;	AgentTransportation<br>
 - **Simulation cycle** (for example, 365 days a year)：<br>
&emsp; &emsp;Renewal of environmental and policy constraints<br>
&emsp; &emsp;Connecting subjects (establishing interactions)<br>
&emsp; &emsp;Main actions：<br>
&emsp; &emsp; &emsp; &emsp;Production body: production → preparing products → preparing orders → adaptation → memorization<br>
&emsp; &emsp; &emsp; &emsp;Consumption subjects: consumption → preparation of orders → memorization<br>
&emsp; &emsp;Go to next day<br>

Understanding this structure helps to customize the configuration with scenario impact

## 3.4 Interactive operation
Run it in the project root directory:
```c
solara run app.py
```
The browser opens an interactive interface that supports a visual view of the simulation.

## 3.5 Script Running
Run in the project root directory：
```c
# Note the space in the filename
python "example 1 basic run.py"
```
Script flow:
<br>
(1) Load the data (e.g. ``cMRIOExample.json``)<br>
(2) Set simulation parameters (time step ```delta_t```, total days ```day_total```, target days ```ndays_target_default```)<br>
(3) Initialize subjects (production, consumption, transport)<br>
(4) Injecting scenario shocks (modifying ```AgentsP_Theta```to simulate capacity declines)<br>
(5) Advancing through the cycle (flow of goods → subject decisions → memory update)<br>
(6) Record results (value added changes, network flows, scarcity, etc.)<br>
(7) Plotting/saving (line graphs + ``.npz`` result files)<br>





After running: <br>
&emsp; &emsp;A daily Gross Value Added line chart will pop up <br>
&emsp; &emsp;Optionally, save the results to ```TestResults_ReductionInProductionCapacityExample.npz```.<br>
The browser will open an interactive interface that supports visual viewing of the simulation.

## 3.6 Adjustable Configurations and Variables<br>

  • 	```MRIOExample.json```：Multi-regional input-output data（```MRIO_R```、```MRIO_S```、```MRIO_Z```、```MRIO_C```、```MRIO_VA```etc.）<br>
   •	Simulation step and duration： ```delta_t = 1/365```，```day_total = 365```，```ndays_target_default = 3```<br>
   •	Impact settings (modified directly within the example script)： <br>
   •	Setting ``AgentsP_Theta`` for some production agents in a specified number of day intervals to simulate a drop in capacity<br>
   •	Transportation chain length/mapping:<br>
   •	Control of cargo flow propulsion and unloading positions through ``AgentsT_*`` with index ``k_NetPP / k_NetPC``.<br>
   •	Main output variables:<br>
&emsp; &emsp;Change in value added at regional/agent level<br>
&emsp; &emsp;Changes in cross-regional product flows<br>
&emsp; &emsp;Scarcity indicators<br>
&emsp; &emsp;Percentage loss of value added<br>

## 3.7 Results Visualization

   - Interactive mode: browser visualization<br>
   - Scripting mode: terminal output + pop-up charts + optional ``.npz`` data file<br>
   


# 4. Output Description

## 4.1 Output Variables

1. **S0_Evolution_ValueAdded_ProductionAgents**: Evolution of value added by each production agent each step for one simulation period.
   - **Shape**: `(model.N_P, day_total)` where `model.N_P` represents the total number of production agents, and `day_total` is the total simulation days.

2. **S0_ProductInNetwork_Region**: Tracks product flow between each region every step. This variable is a 3D array where `S0_ProductInNetwork_Region[i, j, t]` represents the number of products flowing from region `i` to region `j` on day `t`.
   - **Shape**: `(model.MRIO_R, model.MRIO_R, day_total)` where `model.MRIO_R` represents the total number of regions.

3. **S0_ProductInNetwork_Region_Change**: Tracks changes in product flow between regions relative to the initial state `SS_ProductInNetwork_Region`. `S0_ProductInNetwork_Region_Change[i, j, t]` represents the change in product flow from region `i` to region `j` on day `t` compared to the initial state.
   - **Shape**: `(model.MRIO_R, model.MRIO_R, day_total)`

4. **S0_Evolution_Scarcity_RegionsProducts**: Tracks the scarcity level of each product in each region every step. `S0_Evolution_Scarcity_RegionsProducts[i, k, t]` represents the scarcity level of product `k` in region `i` on day `t`.
   - **Shape**: `(model.MRIO_R, model.Sa, day_total)` where `model.Sa` represents the total number of products.


## 4.2 Post-Processing Variables (Calculated after Loops)

1. **S0_Evolution_ValueAdded_Region**  
   Tracks the total Value Added evolution for each region every step. It is derived by aggregating `S0_Evolution_ValueAdded_ProductionAgents` by region.
   - **Shape**: `(model.MRIO_R, day_total)`  
   Where `model.MRIO_R` represents the total number of regions, and `day_total` is the total number of simulation days.

2. **S0_LossPerc_ProductionAgents**  
   Tracks the loss percentage in Value Added for each production agent relative to its initial state. Only production agents with initial Value Added greater than 1e-4 are considered for the loss percentage.
   - **Shape**: `(model.N_P, 1)`  
   Where `model.N_P` represents the total number of production agents.

3. **S0_LossPerc_Region**  
   Tracks the loss percentage in Value Added for each region relative to its initial state. Only regions with initial Value Added greater than 1e-4 are considered for the loss percentage.
   - **Shape**: `(model.MRIO_R, 1)`  
   Where `model.MRIO_R` represents the total number of regions.

4. **S0_ProductInNetwork_Region_Change_Mean**  
   Tracks the average product flow change between regions, calculated by averaging `S0_ProductInNetwork_Region_Change` over the time dimension.
   - **Shape**: `(model.MRIO_R, model.MRIO_R)`  
   Where `model.MRIO_R` represents the total number of regions.

> **Note**: Based on the above variables, we can deduce the total and indirect losses (i.e., total losses minus direct losses) for each production agent and each region, both daily and on average.

# 5. Model Principles

The model is an agent-based model for simulating the evolution of an input-output system which can either be monetary or physical, single- or multi-regional. The modeled scenarios can unfold at relatively fine temporal scales (such as days).

Given enough data and computing power, this agent-based model can be applied to any input-output system with many producers, consumers, and transporters. Below we assume, as in Multi-Regional Input-Output (MRIO) models, there are r = 1, ⋯, R regions and s = 1, ⋯, S sectors in each region. Each region-sector is represented by a production agent `⟨P⟩(r,s)`. The final consumption in each region is represented by a consumption agent `⟨C⟩(r)`. The transportation agent from production agents `⟨P⟩(r₁,s₁)` to `⟨P⟩(r₂,s₂)` is `⟨T⟩^(→⟨P⟩)(r₁,s₁,r₂,s₂)`, shipping the relevant intermediate product, and the transportation agent from the production agent `⟨P⟩(r₁,s₁)` to the consumption agent `⟨C⟩(r₂)` is `⟨T⟩^(→⟨C⟩)(r₁,s₁,r₂)`, shipping the final product to consumers in region r₂.

The specific behaviors and corresponding micro-foundations of the these agents including production agents, consumption agents, and transportation agents are detailed here. According to the classification, the behavior mechanism of each agent is introduced in detail.

## 5.1 Production agent

***Production agent*** `⟨P⟩(r,s)` carries out the production and sends products and orders to other connected agents in the supply network in each simulation step. If external shocks (such as shortage of raw materials and/or loss of production capacity) occur, they can show certain adaptive behaviors, such as replenishing inventory, adjusting the order shares of upstream suppliers, using idle production capacity, adjusting production technology, post-disaster reconstruction, etc. Below we detail each type of behavior of this agent `⟨P⟩(r,s)` in one simulation step (e.g., one day).

### 1) Producing goods using the Leontief production function

The production is based on the Leontief production function and is limited by the total order, production capacity, and raw material supply:

$$
X^{a} = \min\{O^{tot}, X^{cap}, \min\{X^{s'}\}, \min\{X^{r', s'}\}\} \quad (1)
$$

where X^(a) is the actual production; O^(tot) denotes the total order; X^(cap) is production capacity; X^(s') is the production constraint due to the inventory of intermediate product s' (if s' is a homogeneous product such as rice), and X^(r', s') is the production constraint due to the inventory of intermediate product s' from region r' (if s' is specific such as a particular type of machine component). Note that if an intermediate product s' is specific, the products s' from different regions are assumed to be different. These limitation variables are defined in equations (2) to (5) below, respectively.

**Total order.** The total order received by this production agent is

$$
O^{tot} = \sum O^{\leftarrow \langle P \rangle}(r',s') + \sum O^{\leftarrow \langle C \rangle}(r') + O^{E} \quad (2)
$$

where O^(←⟨P⟩) denotes the order of other production agents; O^(←⟨C⟩) denotes the order from the consumer; O^(E) denotes export orders (for open economies).

**Production capacity.** The production capacity of this agent is

$$
X^{cap} = \alpha \times (1 - \theta) \times \overline{X} \quad (3)
$$

where α is overproduction capacity (the value is 1 by default); θ is the reduction in production capacity relative to pre-event level (the value is 0 by default); $\overline{X}$ is the production capacity at the pre-event level.

**Production constraint.** The production constraint due to the inventory of intermediate product is

$$
\begin{cases}
I^{\text{R}}(s') = n(s') \times X_{t - 1}^{a} \times a(s') \\
X^{s'} = \dfrac{I(s')}{a(s')} \times \min\{\,1,\dfrac{I(s')}{\psi\, I^{\text{R}}(s')}\,\}
\end{cases}
\quad \text{if } s' \text{ is a homogeneous product} \quad (4)
$$

$$
\begin{cases}
I^{R}(r', s') = n(s') \times X_{t - 1}^{a} \times a(r', s') \\
X^{\,{r', s'}} = \dfrac{I(r', s')}{a(r', s')} \times \min\{\,1,\dfrac{I(r', s')}{\Psi\, I^{R}(r', s')}\,\}
\end{cases}
\quad \text{if } s' \text{ is a specific product} \quad (5)
$$

where I^(R) is the required inventory level, n is the target usage days for different products; X^(a) is the actual production; a is the input requirement of raw materials for unitary production; X^(s') or X^(r', s') is possible production levels constrained by inventories of different products; I is the current inventory level; Ψ is the ratio of the required inventory level below which the production agent would use only part of the inventory in order to smooth production between simulation steps (the value of Ψ is 0 by default).

### 2) Sending out products onto transportation chains to other agents

The distribution of products is based on received orders and observed patterns of product distribution. When the actual production is equal to the total order, products are allocated according to the orders as described in equation (6). When the actual production is less than the total order, products are allocated according to the share of various orders as described in equation (7).

$$
\begin{cases}
Z^{\rightarrow \langle P \rangle}(r', s') = O^{\leftarrow \langle P \rangle}(r', s') \\
Z^{\rightarrow \langle C \rangle}(r') = O^{\leftarrow \langle C \rangle}(r') \\
Z^{\rightarrow E} = O^{E}
\end{cases}
\quad \text{if } X^{a} = O^{tot} \quad (6)
$$

$$
\begin{cases}
Z^{\rightarrow \langle P \rangle}(r', s') = X^{a} \times \dfrac{O^{\leftarrow \langle P \rangle}(r', s')}{O^{tot}} \\
Z^{\rightarrow \langle C \rangle}(r') = X^{a} \times \dfrac{O^{\leftarrow \langle C \rangle}(r')}{O^{tot}} \\
Z^{\rightarrow E} = X^{a} \times \dfrac{O^{E}}{O^{tot}}
\end{cases}
\quad \text{if } X^{a} < O^{tot} \quad (7)
$$

where Z^(→⟨P⟩), Z^(→⟨C⟩), and Z^(→E) denote products sent toward different production agents, consumption agents, and export (which is only for the open economy).

### 3) Sending orders to replenish intermediate product inventories

The evolution of inventory of an intermediate product s' is

$$
I_{t}(s') = I_{t-1}(s') - X_{t-1}^{a} \times a_{t-1}(s') + \sum Z^{\leftarrow \langle P \rangle}(r', s') \quad (8)
$$

where I_(t)(s') is the inventory of s' for the production agent in period t; I_(t-1)(s') is the inventory of s' for the production agent in period t-1; X_(t-1)^(a) is the actual production in t-1; a_(t-1)(s') is the input requirement of s' for unit production in t-1; Z^(←⟨P⟩) denotes the product inflows from production agents in different regions.

When the target inventory, defined in equation (9), is greater than the current inventory, the agent will increase the order so that the inventory level would gradually grow to the target level, as in equation (10).

$$
I^{T}(s') = n(s') \times \min\{O^{tot},X^{cap}\} \times a(s') \quad (9)
$$

$$
O^{\rightarrow \langle P \rangle}(s') = a(s') \times X^{a} + \bigl( I^{T}(s') - I(s') \bigr) \times \frac{\Delta t}{\tau_{I}} \quad (10)
$$

where I^(T)(s') is the target inventory; O^(→⟨P⟩) is orders sent toward different production agents; Δt is the time length of each simulation step (e.g., one day); τ_(I) is the timescale for adjusting to targeted inventory levels.

### 4) Adjusting upstream suppliers to alleviate the shortage of intermediate products

The order shares of a homogeneous intermediate product s' given to connected suppliers in different regions can be adjusted in each period, according to the difference between the share of product s' sent by a particular supplier in the total product s' sent by all suppliers and the previous order share of this supplier. Intuitively, if a supplier provided fewer raw materials than the order given to it in the previous period, the order share given to it in this period will decrease.

$$
o_{t}^{\rightarrow \langle P \rangle}(r',s') = o_{t - 1}^{\rightarrow \langle P \rangle}(r',s') + \left( \frac{Z^{\rightarrow \langle P \rangle}(r,s|\langle P \rangle(r',s'))}{\sum_{r'}{Z^{\rightarrow \langle P \rangle}(r,s|\langle P \rangle(r',s'))}} - \frac{O_{t - 1}^{\rightarrow \langle P \rangle}(r',s')}{O_{t - 1}^{\rightarrow \langle P \rangle}(s')} \right) \times \frac{\Delta t}{\tau_{O}} \quad (11)
$$

where Z^(→⟨P⟩)(r,s|⟨P⟩(r',s')) denotes the product sent (toward the production agent ⟨P⟩(r,s) in consideration) by production agent ⟨P⟩(r',s'); o_(t)^(→⟨P⟩)(r',s') denotes the share of order for intermediate product s' sent toward the supplier in region r' in the current time period t; τ_(O) denotes the timescale for adjusting to the target order distribution.

Therefore, the order to the supplier in the region r' is

$$
O^{\rightarrow \langle P \rangle}(r',s') = O^{\rightarrow \langle P \rangle}(s') \times o^{\rightarrow \langle P \rangle}(r',s') \quad (12)
$$

where O^(→⟨P⟩)(s') is the total order for the intermediate product s' determined in equation (10).

#### 5) Using overproduction capacity to ensure product supply

If the actual production level is smaller than the total order received, i.e., X^(a) < O^(tot), there is a scarcity. In this case, the production agent will gradually utilize idle production capacities (i.e., overproduction capacity) to increase supply. If X^(a) = O^(tot), there is no scarcity. The production agent will gradually reduce the overproduction capacity parameter α toward 1.

$$
\alpha_{t + 1} =
\begin{cases}
\alpha_{t} + ( \alpha^{\max} - \alpha_{t} ) \times \dfrac{O^{tot} - X^{a}}{O^{tot}} \times \dfrac{\Delta t}{\tau_{\alpha}} & \text{if } X^{a} < O^{tot} \\
\alpha_{t} - ( \alpha_{t} - 1 ) \times \dfrac{\Delta t}{\tau_{\alpha}} & \text{if } X^{a} = O^{tot}
\end{cases}
\quad (13)
$$

where α^(max) denotes the maximum possible overproduction capacity (with the default value of 1.2); τ_(α) is the timescale for adjusting to maximum production capacity.

#### 6) Adjusting production technology to meet the shortage of intermediate products

**Case of scarcity.** Faced with a shortage of an intermediate product, the production agent will gradually reduce the requirement for it to a certain degree. For the intermediate product s', if equation (14) is satisfied, the production agent detects a shortage. Thus, the production agent sees a shortage if the total amount of s' sent by all the connected suppliers falls short of the previous total order. In this case, it will update intermediate requirements for unitary production (i.e., adjusting production technology) as described in equation (15).

$$
O_{t-1}^{\rightarrow \langle P \rangle}(s') > Z^{\rightarrow \langle P \rangle}(r,s|\langle P \rangle(s')) \quad (14)
$$

$$
a_{t}(s') = a_{t - 1}(s') - \frac{O_{t - 1}^{\rightarrow \langle P \rangle}(s') - Z^{\rightarrow \langle P \rangle}(r,s|\langle P \rangle(s'))}{O_{t - 1}^{\rightarrow \langle P \rangle}(s')} \times a_{t - 1}(s') \times \frac{\Delta t}{\tau_{A}^{\downarrow}} \quad (15)
$$

where Z^(→⟨P⟩)(r,s|⟨P⟩(s')) denotes the product sent (toward the production agent ⟨P⟩(r,s) in consideration) by all connected suppliers and O_(t-1)^(→⟨P⟩)(s') is the total order for s' sent in the previous simulation period; a_(t)(s') denotes the intermediate requirement of s' for unitary production and τ_(A)^(↓) is the timescale parameter of this adjustment.

**Case of no scarcity.** If equation (16) is satisfied, there is no scarcity of intermediate product s'. This means the total amount of s' sent by all the connected suppliers satisfies the previous total order. Then the requirement of s' for unitary production will shift back towards the original value in the steady state as described in equation (17).

$$
O_{t-1}^{\rightarrow \langle P \rangle}(s') = Z^{\rightarrow \langle P \rangle}(r,s|\langle P \rangle(s')) \quad (16)
$$

$$
a_{t}(s') = a_{t - 1}(s') + \frac{\overline{a}(s') - a_{t - 1}(s')}{\overline{a}(s')} \times \bigl( \overline{a}(s') - a_{t - 1}(s') \bigr) \times \frac{\Delta t}{\tau_{A}^{\uparrow}} \quad (17)
$$

where τ_(A)^(↑) is the timescale for this technology adaptation and $\overline{a}(s')$ is the input requirement for unitary production in a steady state.

If the modeled input-output system is open, agents may resort to foreign supplies, and the import requirement for unitary production is:

$$
i_{t} = i_{t - 1} + \sum_{s'=1}^{s}\bigl( a_{t - 1}(s') - a_{t}(s') \bigr) \quad (18)
$$

that is, the agent may use imported goods to compensate for the decreased/increased requirements for scarce intermediate inputs.

#### 7) Post-disaster reconstruction (gradual restoration of production capacity)

If the production agent suffers from an external shock resulting in the reduction of production capacity (with θ representing the ratio of loss), after the event, it will slowly restore the capacity through reconstruction until the θ becomes 0. Therefore, the production agent will gradually recover after an external shock (such as a disaster) and the speed is governed by the timescale parameter τ_(θ).

$$
\theta_{t + 1} = \bigl( 1 - \frac{\Delta t}{\tau_{\theta}} \bigr) \times \theta_{t} \quad (19)
$$

where θ is the ratio of loss and τ_(θ) is the timescale for reconstruction.

#### 8) Record key state variables

The production agent stores the relevant state variables in the current simulation period, which will be used in computations in future periods.


### (2) Consumption agent


***Consumption agent*** `⟨C⟩(r)` consumes products s' = 1, 2, ⋯, S in each simulation period. To do this, it must send orders to different suppliers and adjust these orders according to the actual supply fluctuations. Specifically, the consumption agent acts in the following ways.

#### 1) Sending order outflows

In a simulation period, the order shares (for a homogeneous product s') given to producers in different regions will be adjusted according to the difference between the share of product s' sent by a particular supplier in the total product s' sent by all suppliers and the previous order share of this supplier. Intuitively, if a supplier provided fewer products than the order given to it in the previous period, the order share given to it in this period will decrease.

$$
o_{t}^{\rightarrow \langle P \rangle}(r',s') = o_{t - 1}^{\rightarrow \langle P \rangle}(r',s') + \left( \frac{Z^{\rightarrow \langle C \rangle}(r,s|\langle P \rangle(r', s'))}{\sum_{r'}{Z^{\rightarrow \langle C \rangle}(r,s|\langle P \rangle(r', s'))}} - \frac{O_{t - 1}^{\rightarrow \langle P \rangle}(r', s')}{O_{t - 1}^{\rightarrow \langle P \rangle}(s')} \right) \times \frac{\Delta t}{\tau_{O}} \quad (20)
$$

where Z^(→⟨C⟩)(r,s|⟨P⟩(r', s')) denotes the product sent toward the consumption agent ⟨C⟩(r) in consideration by production agent ⟨P⟩(r',s');  o_(t)^(→⟨P⟩)(r',s') denotes the share of order for product s' sent toward the connected supplier in region r' in the simulation period t; τ_(O) denotes the timescale for adjusting to the target order distribution.

Therefore, the consumption agent sends orders to all connected suppliers for s' in different regions:

$$
O^{\rightarrow \langle P \rangle}(r', s')= O^{\rightarrow \langle P \rangle}(s') \times o^{\rightarrow \langle P \rangle}(r', s') \quad (21)
$$

where O^(→⟨P⟩)(s') is the total order for this homogeneous product s', which is determined by the steady-state consumption level.

#### 2) Record key state variables

The consumption agent stores the relevant state variables in the current simulation period, which will be used in computations in future periods.

### (3) Transportation agent

Transportation agent is the transportation chain connecting a pair of agents. Each transportation chain can transport one type of product. It is represented by a row vector and the length of this vector equals the number of simulation periods for transporting the product. For example, if it takes 5 days to move a certain type of product from one production agent to another production/consumption agent and given the temporal resolution for the simulation is 1 day, the relevant transportation chain will have a length of 5. In each period, the sending agent put the product into the first grid of this transportation chain, each original element of this chain moves to the right by one grid, and the original last element of the chain is unloaded to the receiving agent. In this way, we can simulate the product transportation processes between different agents. Below we express the above image mathematically.

#### Transportation between production agents

##### 1) Transportation chain between production agents

`⟨T⟩^(→⟨P⟩)(r₁,s₁,r₂,s₂)` delivers the cargo produced by the production agent in `⟨P⟩(r₁,s₁)` to the production agent in `⟨P⟩(r₂,s₂)`. The length of the transportation chain is L and the cargo at each step of the transportation chain in period t is

$$
Z_{t}(\cdot \mid \langle T \rangle^{\rightarrow \langle P \rangle}(r_1,s_1,r_2,s_2)) \equiv (Z_{1,t}, Z_{2,t}, \cdots, Z_{L,t}) \quad (22)
$$

##### 2) Loading

In the period of \(t+1\), the cargo is loaded to the transportation agent.

$$
\begin{aligned}
\tilde{Z}_{t+1}(\cdot \mid \langle T \rangle^{\rightarrow \langle P \rangle}(r_1,s_1,r_2,s_2))
&= \bigl( Z_{t}^{\rightarrow \langle P \rangle}(r_2,s_2 \mid \langle P \rangle(r_1,s_1)),\, Z_{1,t}, Z_{2,t}, \cdots, Z_{L,t} \bigr) \\
&\equiv \bigl( \tilde{Z}_{1,t+1}, \tilde{Z}_{2,t+1}, \cdots, \tilde{Z}_{L+1,t+1} \bigr) \quad (23)
\end{aligned}
$$

where \(Z_{t}^{\rightarrow \langle P \rangle}(r_2,s_2 \mid \langle P \rangle(r_1,s_1))\) represents the cargo produced by `⟨P⟩(r₁,s₁)` and planned to be delivered to `⟨P⟩(r₂,s₂)` in \(t\).

##### 3) Blockage (optional, not in this study)

If this transportation agent `⟨T⟩^(→⟨P⟩)(r₁,s₁,r₂,s₂)` has \(b\) units of goods \(s_1\) blocked at step \(l\) (\(l=1,2,\ldots,L\)), then unit \(b\) of goods \(s_1\) on step \(l\) cannot be transported forward as modeled in equation (23) due to this blockage. If \(b>0\), the transportation chain is blocked; if \(b=0\), the transportation chain is not blocked. Therefore, we must adjust the flows from step \(l\) to \(l+1\) in the transportation chain:

$$
\begin{aligned}
\widetilde{\widetilde{Z}}_{t+1}
&= \bigl( \tilde{Z}_{1,t+1}, \ldots, \tilde{Z}_{l,t+1}+b,\, \tilde{Z}_{l+1,t+1}-b, \ldots, \tilde{Z}_{L+1,t+1} \bigr) \\
&\equiv \bigl( \widetilde{\widetilde{Z}}_{1,t+1}, \ldots, \widetilde{\widetilde{Z}}_{l,t+1}, \widetilde{\widetilde{Z}}_{l+1,t+1}, \ldots, \widetilde{\widetilde{Z}}_{L+1,t+1} \bigr) \quad (24)
\end{aligned}
$$

##### 4) Unloading

The cargo in the last step \(L+1\) of transportation chain is unloaded to `⟨P⟩(r₂,s₂)` in simulation period \(t\):

$$
Z_{t}^{\leftarrow \langle P \rangle}( r_{1}, s_{1} \mid \langle P \rangle( r_{2}, s_{2} ) ) = \widetilde{\widetilde{Z}}_{L+1,t+1} \quad (25)
$$

##### 5) Updating

After unloading the cargo in previous step, the cargo at each step of the transportation agent in \(t+1\) is:

$$
Z_{t+1}(\cdot \mid \langle T \rangle^{\rightarrow \langle P \rangle}( r_{1}, s_{1}, r_{2}, s_{2} ) )
= \bigl( \widetilde{\widetilde{Z}}_{1,t+1}, \ldots, \widetilde{\widetilde{Z}}_{L,t+1} \bigr) \quad (26)
$$

Therefore, we have completed updating the transportation chain connecting the two production agents `⟨P⟩(r₁,s₁)` and `⟨P⟩(r₂,s₂)`.

---

#### Transportation between production agent and consumption agents

##### 1) Transportation agent between production agent and consumption agents

`⟨T⟩^(→⟨C⟩)(r₁,s₁,r₂)` delivers the cargo produced by the production agent in `⟨P⟩(r₁,s₁)` to the consumption agent in `⟨C⟩(r₂)`. The length of the transportation agent is \(L\), and the cargo at each step of the transportation agent in \(t\) is:

$$
Z_{t}(\cdot \mid \langle T \rangle^{\rightarrow \langle C \rangle}(r_1,s_1,r_2)) \equiv (Z_{1,t}, Z_{2,t}, \cdots, Z_{L,t}) \quad (27)
$$

##### 2) Loading

In the period of \(t+1\), the cargo is loaded to the transportation agent.

$$
\begin{aligned}
\tilde{Z}_{t+1}(\cdot \mid \langle T \rangle^{\rightarrow \langle C \rangle}(r_1,s_1,r_2))
&= \bigl( Z_{t}^{\rightarrow \langle C \rangle}(r_2 \mid \langle P \rangle(r_1,s_1)),\, Z_{1,t}, Z_{2,t}, \cdots, Z_{L,t} \bigr) \\
&\equiv \bigl( \tilde{Z}_{1,t+1}, \tilde{Z}_{2,t+1}, \cdots, \tilde{Z}_{L+1,t+1} \bigr) \quad (28)
\end{aligned}
$$

where \(Z_{t}^{\rightarrow \langle C \rangle}(r_2 \mid \langle P \rangle(r_1,s_1))\) represents the cargo produced by production agent `⟨P⟩(r₁,s₁)` and planned to be delivered to consumption agent `⟨C⟩(r₂)` in \(t\).

##### 3) Blockage (Optional, not in this study)

If this transportation agent `⟨T⟩^(→⟨C⟩)(r₁,s₁,r₂)` has \(b\) units of goods \(s_1\) blocked at step \(l\) (\(l=1, 2, \ldots, L\)), then unit \(b\) of goods \(s_1\) on step \(l\) cannot be transported forward. If \(b>0\), the transportation chain is blocked; if \(b=0\), the transportation chain is not blocked. Similar to the previous case, the relevant production flows must be adjusted:

$$
\begin{aligned}
\widetilde{\widetilde{Z}}_{t+1}
&= \bigl( \tilde{Z}_{1,t+1}, \ldots, \tilde{Z}_{l,t+1}+b,\, \tilde{Z}_{l+1,t+1}-b, \ldots, \tilde{Z}_{L+1,t+1} \bigr) \\
&\equiv \bigl( \widetilde{\widetilde{Z}}_{1,t+1}, \ldots, \widetilde{\widetilde{Z}}_{l,t+1}, \widetilde{\widetilde{Z}}_{l+1,t+1}, \ldots, \widetilde{\widetilde{Z}}_{L+1,t+1} \bigr) \quad (29)
\end{aligned}
$$

##### 4) Unloading

The cargo on the last step \(L+1\) of the transportation chain is unloaded to the target consumption agent `⟨C⟩(r₂)` in simulation period \(t\):

$$
Z_{t}^{\leftarrow \langle P \rangle}( r_{1}, s_{1} \mid \langle C \rangle( r_{2} ) ) = \widetilde{\widetilde{Z}}_{L+1,t+1} \quad (30)
$$

##### 5) Updating

After unloading the cargo in the previous step, the cargo at each step of the transportation chain in \(t+1\) is:

$$
Z_{t+1}( \cdot \mid \langle T \rangle^{\rightarrow \langle C \rangle}( r_{1}, s_{1}, r_{2} ) )
= \bigl( \widetilde{\widetilde{Z}}_{1,t+1}, \cdots, \widetilde{\widetilde{Z}}_{L,t+1} \bigr) \quad (31)
$$

Therefore, we have completed updating the transportation chain connecting the production agent `⟨P⟩(r₁,s₁)` and the consumption agent `⟨C⟩(r₂)`.



