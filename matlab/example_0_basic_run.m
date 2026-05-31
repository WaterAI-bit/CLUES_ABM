% Load MRIO data.
load('data/MRIOExample.mat');% Note: This is an example input-output table we have provided and does not represent real-world data.



%% Setting the following variables for simulation:
% Length of each time step, as a fraction of input flows:
delta_t = 1/365;

% Total periods of simulation:
day_total = 365;

% Default targeted inventory periods:
% Note: This is (1 + the number of periods the remaining inputs would last after the current production ends).
... Therefore, it should be >= 1.
ndays_Target_Default = 3;

% Periods for transportion between regions (integers, at least 1):
MRIO_Dist = 3 * ones(R_MRIOExample,S_MRIOExample);
for i=1:R_MRIOExample
    % Transportation line is 1 step in the same region.
    MRIO_Dist(i,i) = 1;
end
clear i;
% In fact, trasportation line lengths in CLUES model are the lengths of
... lines that stays in transportation steps.
... Therefore, it is tranportation period - 1.
... For example, if the tranportation period is 1 (i.e., arriving next period),
... there will be no goods staying in the intermediate transportation steps,
... so, the trasportation line length is 1 - 1 = 0.
MRIO_Dist = MRIO_Dist - 1;

% Water required for unitary output of each production agent (i.e., region-sector):
% Note: At least 1 water intensity should be nonzero to avoid computational error.
AgentsP_WaterIntensity = ones(R_MRIOExample*S_MRIOExample, 1); % MRIO_R*MRIO_S column vector. 

%% Define China's economy and initialize the model.
GlobalEcon = clues_abm.WorldOfMatrix_GPU; % China's economy is our World.

% Basic Input-Output variables.
GlobalEcon.MRIO_R = R_MRIOExample; % Number of regions (Provinces).
GlobalEcon.MRIO_S = S_MRIOExample; % Number of sectors in each region.
GlobalEcon.MRIO_Z = Z_MRIOExample; % Intermediate flows (in a year) from each Province-region to each Province-region.
GlobalEcon.MRIO_C = C_MRIOExample; % Consumption of each region.
GlobalEcon.MRIO_VA = VA_MRIOExample; % Value added of each region-sector.
GlobalEcon.OpenEcon = false; % Global economy is closed.
S2Sa = eye(S_MRIOExample);

% Length of each time step, as a fraction of input flows.
GlobalEcon.delta_t = delta_t;
% Default targeted inventory days.
GlobalEcon.ndays_Target_Default = ndays_Target_Default;
% Distance (i.e., steps in the transportation line) between regions.
GlobalEcon.MRIO_Dist = MRIO_Dist;
% Resource required for unitary output of each production agent.
GlobalEcon.AgentsP_ResourceIntensity = AgentsP_WaterIntensity;
% A conversion matrix, each row is a product and each column is an aggregate product. 
% The value is 1 if the column is aggregate product for the row product, otherwise is 0.
GlobalEcon.S2Sa = S2Sa; % Use the imported conversion matrix.
% ChinaEcon.S2Sa = eye(ChinaEcon.MRIO_R*ChinaEcon.MRIO_S); % Use the indentity matrix, so aggregate sectors are the same as original sectors.

%% Initialize the World.
tic % Start timing.
GlobalEcon = GlobalEcon.InitializeBasicVariables_UsingMRIO;
MATLAB_RUNTIME_1_1_InitializeBasic = toc; % End timing.

tic % Start timing
GlobalEcon = GlobalEcon.InitializeProductionAgents_UsingMRIO;
MATLAB_RUNTIME_1_2_InitializeProduction = toc; % End timing.

tic % Start timing
GlobalEcon = GlobalEcon.InitializeConsumptionAgents_UsingMRIO;
MATLAB_RUNTIME_1_3_InitializeConsump = toc; % End timing.

tic % Start timing
GlobalEcon = GlobalEcon.InitializeTransportationAgents_UsingMRIO;
MATLAB_RUNTIME_1_4_InitializeTransport = toc; % End timing.

%% Define key variables to be recorded.
% Evolution of value-added of all Provinces-sectors from day 1 to day_total.
... Each row represents a production agent (Province-sector in this case).
S0_Evolution_ValueAdded_ProductionAgents = zeros(GlobalEcon.N_P, day_total);

% Steady-state value-added of Province-sectors (each simulation step).
SS_AgentsP_VA = GlobalEcon.AgentsP_VA;

% Matrix for converting Province-sector results to Province results.
RegionSectors2Regions = kron(eye(GlobalEcon.MRIO_R),ones(GlobalEcon.MRIO_S,1));

% Steady-state value-added of Province-sectors (each simulation step).
SS_Provinces_VA = RegionSectors2Regions' * SS_AgentsP_VA;

% Evolution of the trade network between Provinces (for products coming into the Provinces).
... Eeah layer is a square matrix of product flows, where the row represent the sending Province, and the column represents the receiving Province.
S0_ProductInNetwork_Provinces = zeros(GlobalEcon.MRIO_R, GlobalEcon.MRIO_R, day_total);
S0_ProductInNetwork_Provinces_Change = zeros(GlobalEcon.MRIO_R, GlobalEcon.MRIO_R, day_total); % Change relative to the steady state.
% Steady state trade network.
SS_ProductInNetwork_Provinces = RegionSectors2Regions' * GlobalEcon.AgentsP_ProductInP * RegionSectors2Regions;

% Evolution of Scarcity Indices for each product in each region.
... Eeah layer is a R * S matrix: rows are regions, and columns are sectors (i.e. products).
S0_Evolution_Scarcity_RegionsProducts = zeros(GlobalEcon.MRIO_R,GlobalEcon.Sa,day_total);

%% SIMULATE NETWORK DYNAMICS AND RECORD KEY VARIABLES.
tic % Start timing

for day=1:day_total % Days for simulation. 
    % Direct impact:
    % In the first month, since day 2, all sectors in China suffer 10% losses in production capacity.
    % After that, they recover gradually according to the default parameters in the model.
    disp(day);
    ind = 1:20; % The indices of the affected nation-sectors.
    ...China is the 8th nation in the WIOT2014 table, where each nation has 56 sectors.
    ...Therefore, China's sectors have indices from (56*7+1)=393 to 56*8=448.
    if (day>=1) && (day<=30)
        GlobalEcon.AgentsP_Theta(ind) = 0.2;  
    end
    
    if (day>=100) && (day<=130)
        GlobalEcon.AgentsP_Theta(77:96) = 0.3;
    end
    
    if (day>=200) && (day<=230)
        GlobalEcon.AgentsP_Theta(273:280) = 0.4;
    end
    

    % To simulate the impact of resource constraints:
    % We can revise the property obj.WaterConstraints of the WorldOfMatrix_Water object:
    % obj.MRIO_R*1 vector: Water constraints for each MRIO region, FOR THE CURRENT SIMULATION PERIOD.
%     % Setting up regions where water Scarcity occurs:
%       Regions_WaterScarcity=2;
%     if day==1 % For example, in the first period, water Scarcity occurs:
%         WaterConstraints_RatioInput=ones(313,1);
%         WaterConstraints_RatioInput(Regions_WaterScarcity)=WaterConstraints_Ratio(Regions_WaterScarcity);
%         ChinaEcon.WaterConstraints = ChinaEcon.WaterConstraints.* WaterConstraints_RatioInput;
%     else % Very abundant water. PLEASE USE THIS IN EACH SIMULATION PERIOD IS THERE IS NO WATER CONSTRAINT!
%         ChinaEcon.WaterConstraints = ones(size(ChinaEcon.WaterConstraints)) * (sum(ChinaEcon.AgentsP_SS_Xcap) * 10 + 1e10);
%     end
    %% This part calculates movements in transportation lines.
    % DON'T REVISE THIS SECTION IF THERE IS NO TRANSPORTATION LINE OBSTRUCTION.

    % Tranportation lines:
    % Load, move, and unload goods in the transportation chains.
    % Move one step forward (creating augumented transportation lines), for P2P.
    GlobalEcon.AgentsT_P2P = [ zeros(GlobalEcon.nl_NetPP,1), GlobalEcon.AgentsT_P2P ];
    % Calculate products loaded to each transportation lines.
    GlobalEcon.AgentsT_P2P(GlobalEcon.AgentsT_P2P_StartLinInd) = GlobalEcon.AgentsP_ProductOutP(GlobalEcon.k_NetPP);
    % Move one step forward (creating augumented transportation lines), for P2C.
    GlobalEcon.AgentsT_P2C = [ zeros(GlobalEcon.nl_NetPC,1), GlobalEcon.AgentsT_P2C ];
    % Calculate products loaded to each transportation lines.
    GlobalEcon.AgentsT_P2C(GlobalEcon.AgentsT_P2C_StartLinInd) = GlobalEcon.AgentsP_ProductOutC(GlobalEcon.k_NetPC);  

    % Transportation line obstruction (optional).
    % HERE WE DO NOTHING SINCE THERE IS NO TRANSPORTATION LINE OBSTRUCTION.

    % Calculate products unloaded to each production agent from transportation lines.
    temp = GlobalEcon.AgentsP_ProductInP';
    temp(GlobalEcon.k_NetPP) =  GlobalEcon.AgentsT_P2P(:,end);
    GlobalEcon.AgentsP_ProductInP = temp';
    GlobalEcon.AgentsT_P2P(:,end) = [];
    % Calculate products unloaded to each consumption agent from transportation lines.
    temp = GlobalEcon.AgentsC_ProductInP';
    temp(GlobalEcon.k_NetPC) =  GlobalEcon.AgentsT_P2C(:,end);
    GlobalEcon.AgentsC_ProductInP = temp';
    GlobalEcon.AgentsT_P2C(:,end) = [];
    clear temp;
    
    %% Other actions of the agents, which are automatically computed.
    % Agents commuicate.
    GlobalEcon = GlobalEcon.AgentsCommunicate;    
    % Update inventories of production agents.
    GlobalEcon = GlobalEcon.UpdateInventories;   
    % Update export orders (only for open economy).
    GlobalEcon = GlobalEcon.UpdateExportOrders;
    % Update shares.
    GlobalEcon = GlobalEcon.UpdateShares;
    % Production agents produce.
    GlobalEcon = GlobalEcon.ProductionAgentsProduce;
    % Production agents prepare product outflows.
    GlobalEcon = GlobalEcon.ProductionAgentsPrepareProductOut;
    % Production agents prepare order outflows.
    GlobalEcon = GlobalEcon.ProductionAgentsPrepareOrderOut;
    % Production agents adpat to shocks.
    GlobalEcon = GlobalEcon.ProductionAgentsAdaptToShocks;
    % Production agents adpat production modes to shortages.
    GlobalEcon = GlobalEcon.ProductionAgentsAdaptToShortages;
    % Production agents remember key state variables.
    GlobalEcon = GlobalEcon.ProductionAgentsRemember;
    % Consumption agents consume.
    GlobalEcon = GlobalEcon.ConsumptionAgentsConsume;
    % Consumption agents prepare order outflows.
    GlobalEcon = GlobalEcon.ConsumptionAgentsPrepareOrderOut;
    % Consumption agents remember key state variables
    GlobalEcon = GlobalEcon.ConsumptionAgentsRemember;
    
    %% Record the evolution of key variables.
    S0_Evolution_ValueAdded_ProductionAgents(:,day) =  GlobalEcon.AgentsP_VA;
    S0_ProductInNetwork_Provinces(:,:,day) = RegionSectors2Regions' *  GlobalEcon.AgentsP_ProductInP * RegionSectors2Regions;
    S0_ProductInNetwork_Provinces_Change(:,:,day) = S0_ProductInNetwork_Provinces(:,:,day) - SS_ProductInNetwork_Provinces;
    S0_Evolution_Scarcity_RegionsProducts(:,:,day) = GlobalEcon.Scarcity_RegionsProducts;

end
MATLAB_RUNTIME_2_Simulate = toc; % End timing.

%% Calculate other key variables, using the recorded.
% Evolution of value-added of all Provinces.
S0_Evolution_ValueAdded_Provinces = RegionSectors2Regions' * S0_Evolution_ValueAdded_ProductionAgents;

% Percentage of value-added reduction of each Province-sector.
S0_LossPerc_ProductionAgents = zeros(GlobalEcon.N_P,1);
ind = SS_AgentsP_VA > 10^(-4); % Select sectors with positive value-added in the beginning.
S0_LossPerc_ProductionAgents(ind) = 100 * (SS_AgentsP_VA(ind)-mean(S0_Evolution_ValueAdded_ProductionAgents(ind,:),2)) ./ SS_AgentsP_VA(ind);

% Percentage of value-added reduction of each region.
S0_LossPerc_Provinces = zeros(GlobalEcon.MRIO_R,1);
ind = SS_Provinces_VA > 10^(-4); % Select sectors with positive value-added in the beginning.
S0_LossPerc_Provinces(ind) = 100 * (SS_Provinces_VA(ind)-mean(S0_Evolution_ValueAdded_Provinces(ind,:),2)) ./ SS_Provinces_VA(ind);

% Average change of inter-region trade network each simulation step.
S0_ProductInNetwork_Provinces_Change_Mean = mean(S0_ProductInNetwork_Provinces_Change,3);

%% Saving.
save('output/TestResults_ReductionInProductionCapacityExample0.mat','S0_Evolution_ValueAdded_ProductionAgents', 'S0_Evolution_ValueAdded_Provinces', ...
   'S0_ProductInNetwork_Provinces', 'S0_ProductInNetwork_Provinces_Change', 'S0_ProductInNetwork_Provinces_Change_Mean', ...
   'S0_LossPerc_ProductionAgents', 'S0_LossPerc_Provinces', ...
   'S0_Evolution_Scarcity_RegionsProducts', ...
   'SS_AgentsP_VA', 'SS_Provinces_VA', 'SS_ProductInNetwork_Provinces', ...
   'RegionSectors2Regions')

%% Plot the evoluton of values added of all production agents.

%% Plot the evoluton of the Chemical Industry in China and the Paper Industry in US.
col_sum = sum(S0_Evolution_ValueAdded_ProductionAgents, 1);
%col_sum = col_sum - col_sum(1);
x = 1:length(col_sum);           % 横坐标：每列的索引
y = col_sum;                     % 纵坐标：每列的和
scatter(x, y, 'filled');        % 绘制散点图，'filled' 表示填充颜色
xlabel('Column Index');
ylabel('Sum of Each Column');
title('Scatter Plot of Column Sums');
