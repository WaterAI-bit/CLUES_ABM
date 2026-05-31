classdef WorldOfMatrix_GPU < handle
    properties (Constant=true) % About.
        Name = '(An agent-based model for) Climate-resilient and Low-carbon Unfolding Economic Scenarios';
        Acronym = 'CLUES-ABM';
        Institution = 'CEEP-BIT';
        Designer = 'Shen Qu';
        Version  = '4.0';
        Date = '2023/03/31';
    end
    properties % Values relating to time.
        delta_t = 1/12 % Length of each time step, as a fraction of input flows.
        % Characteristic times, measured in same units as above.
        tau_Theta = 1/2 % Timescale for reconstruction.
        tau_Alpha = 1/2 % Timescale for adjusting to maximum production capacity.
        tau_I = 1/2 % Timescale for adjusting to targeted inventory levels.
        tau_O = 1/2 % Timescale for adjusting to targeted order distributions.
        tau_A = 1/2 % Timescale for technology adaptation.
        tau_E = 1/2 % Timescale for export demand adaptation.
        tau = 1/2 % Timescale, generic.
    end
    properties % Basic variables of the world.
        N_P % Number of production agents. Eech production agents produces one product.
        N_C % Number of consumption agents.
        S % Number of products.
        Sa % Number of aggregated products.

        ndays_Target_Default % Default targeted inventory days
        ... for different intemediate products used by different production agents.
        
        Products % N_P*1 vector for the index of the product produced by each production agent.
        Products_Matrix % N_P*S logical matrix that indicates whether the row producer produces the column product.
        S2Sa % S * Sa matrix, each row is a product, 
        ... and the value is 1 if the column is aggregate product for the row product, otherwise is 0.
        
        % Network connecting producers.
        NetPP logical % N_P*N_P matrix: 1 means that row producer sends products to column producer; 0 means otherwise.
        nl_NetPP % Number of links in NetPP.
        k_NetPP % nl_NetPP*1 vector: Linear indices of non-zero elements in NetPP.
        CutOff_NetPP = 1 % Cutoff value for flows to be included in the network.
        DistPP % N_P*N_P matrix: tranportation line lengths (>=0) from row producer to column producer. 
        i_NetPP % nl_NetPP*1 vector: Indices of non-zero elements in NetPP.
        % THE BELOW VARIABLES ARE MRIO VARIABLES.
        SendingRegion_NetPP % nl_NetPP*1 vector: Sending regions of non-zero elements in NetPP.
        ReceivingRegion_NetPP % nl_NetPP*1 vector: Receiving Regions of non-zero elements in NetPP.
        SendingSector_NetPP % nl_NetPP*1 vector: Sending sectors of non-zero elements in NetPP.
        ReceivingSector_NetPP % nl_NetPP*1 vector: Receiving sectors of non-zero elements in NetPP.
        % The following is an nl_NetPP*6 table for all the Production-to-Production-Agent links.
        AllLinks_NetPP
        % Column 1: i_NetPP
        % Column 2: k_NetPP
        % Column 3: SendingRegion_NetPP
        % Column 4: ReceivingRegion_NetPP
        % Column 5: SendingSector_NetPP
        % Column 6: ReceivingSector_NetPP
        
        % Network connecting producers and consumers.
        NetPC logical % N_P*N_C matrix: 1 means that row producer sends products to column consumer; 0 means otherwise.
        nl_NetPC % Number of links in NetPC.
        k_NetPC % nl_NetPC*1 vector: Linear indices of non-zero elements in NetPC.
        CutOff_NetPC = 1 % Cutoff value for flows to be included in the network.
        DistPC % N_P*N_C matrix: tranportation line lengths (>=0) from row producer to column consumer. 
        i_NetPC % nl_NetPC*1 vector: Indices of non-zero elements in NetPC.
        % THE BELOW VARIABLES ARE MRIO VARIABLES.
        SendingRegion_NetPC % nl_NetPC*1 vector: Sending regions of non-zero elements in NetPC.
        ReceivingRegion_NetPC % nl_NetPC*1 vector: Receiving Regions of non-zero elements in NetPC.
        SendingSector_NetPC % nl_NetPC*1 vector: Sending sectors of non-zero elements in NetPC.
        % nl_NetPC*5 table for all the Production-to-Production-Agent links.
        AllLinks_NetPC
        % Column 1: i_NetPC
        % Column 2: k_NetPC
        % Column 3: SendingRegion_NetPC
        % Column 4: ReceivingRegion_NetPC
        % Column 5: SendingSector_NetPC
        
        OpenEcon = false % Whether this is an open economy.
        ...true means open; false means closed.

        % Scarcity Indices for each aggregated product in each region.
        % Defined as: (Product demanded - Product supplied) / (Product demanded). (It is 0 if Product demanded is 0.)
        Scarcity_RegionsProducts % Scarcity Indices. R * Sa matrix: rows are regions, and columns are aggregated products.
        RegionsProducts_Supplied % Products comming in (or supplied). R * Sa matrix: rows are regions, and columns are aggregated products.
        RegionsProducts_Demanded % Products demanded. R * Sa matrix: rows are regions, and columns are aggregated products.
    end
    properties % State variables of production agents.
        % Production Agents.
        AgentsP_Theta % N_P*1: Reduction in production capacity relative to pre-event level, in [0,1].
        AgentsP_Alpha % N_P*1: Overproduction capacity, in [1, inf).
        AgentsP_Alpha_max % N_P*1: Maximun possible overproduction capacity, in [1, inf).
        % Order inflows: Each row represents a order receiver, each column a order sender.
        AgentsP_OrderInP % N_P*N_P: Order from production agents.
        AgentsP_OrderInC % N_P*N_C: Order from consumption agents.
        AgentsP_OrderInE % N_P*1: Order from export (only for open economy).
        AgentsP_OrderTot % N_P*1: Total order, sum of the above.
        AgentsP_orderlnP % N_P*N_P: Share of order from other production agents in total order.
        AgentsP_orderInC % N_P*N_C: Share of order from consumption agents in  total order.
        AgentsP_orderInE % N_P*1: Share of order from export in total order (only for open economy).
        % Product inflows: Each row represents a product receiver, each column a product sender.
        AgentsP_ProductInP % N_P*N_P: Product inflow through transportation agents (from production agents);
        AgentsP_ProductSentP % N_P*N_P: Product sent (toward here) by production agents in the previous period.
        AgentsP_ProductSentS % N_P*S: Different products sent toward producers at (t-1).
        AgentsP_ProductSentSa % N_P*Sa: Different aggregate products sent toward producers at (t-1).
        AgentsP_productSentP % N_P*N_P: Share of product sent (toward here) by production agents in a sector.
        
        % Production technology: Each row represents a production agent, each column an type of input.
        AgentsP_a % N_P*Sa: Input (intermediate good) requirement for unitary production.
        AgentsP_va % N_P*1: Value-added requirement for unitary production.
        AgentsP_im % N_P*1: Import requirement for unitary production (only for open economy).
        % Inventories:
        AgentsP_n % N_P*Sa: Targeted days of use for different products. TO BE INITIALIZED!
        AgentsP_I % N_P*Sa: Current inventory level.
        % Production: Each row represents a production agent.
        AgentsP_Xcap % N_P*1: Production capacity.
        AgentsP_Xs % N_P*Sa: Possible production levels constrained by inventories of different products.
        AgentsP_Xa % N_P*1: Actual production.
        AgentsP_VA % N_P*1: Actual value added.
        AgentsP_ResourceIntensity % N_P*1: Resource required for unitary output of each production agent.
        % Product Outflows: Each row represents an product sender, each column a product receiver.
        AgentsP_ProductOutP % N_P*N_P: Product sent toward different production agents (through transportation agents).
        AgentsP_ProductOutC % N_P*N_C: Product sent toward different consumption agents (through transportation agents).
        AgentsP_ProductOutE % N_P*1: Product sent toward export (only for open economy).
        % Order Outflows: Each row represents an order sender, each column an order receiver.
        AgentsP_OrderOutP % N_P*N_P: Orders sent toward different production agents.
        AgentsP_orderOutP % N_P*N_P: Share of orders (in total order of the product) sent toward different production agents.
        AgentsP_OrderOutS % N_P*S: Orders sent for different products.
        AgentsP_orderOutS % N_P*S: Share of orders (in total order of the aggregate product) for different products.
        AgentsP_OrderOutSa % N_P*Sa: Orders sent for different aggregate products.
        
        % State variables in previous periods.
        AgentsP_PP_a
        AgentsP_PP_Xa
        AgentsP_PP_OrderOutP
        AgentsP_PP2_OrderOutP
        AgentsP_PP_orderOutP
        AgentsP_PP2_orderOutP
        AgentsP_PP_OrderOutS
        AgentsP_PP2_OrderOutS
        AgentsP_PP_OrderOutSa
        AgentsP_PP2_OrderOutSa
    end
    properties % State variables of consumption agents.
        % Product inflows: Each row represents a consumer agent.
        AgentsC_ProductInP % N_C*N_P: Product inflow through transportation agents (from production agents).
        AgentsC_ProductSentP % N_C*N_P: Product sent (toward here) by production agents.
        AgentsC_ProductSentS % N_C*S: Different products sent toward consumers at (t-1).
        AgentsC_ProductSentSa % N_C*Sa: Different aggregate products sent toward consumers at (t-1).
        % Shares in product inflows.
        AgentsC_productSentP % N_C*N_P: Share of product sent (toward here) by production agents in the previous period.
        % Import requirement (i.e., order) for consumption, only for open economy.
        AgentsC_IMO % N_C*1.
        % Order Outflows.
        AgentsC_OrderOutP % N_C*N_P: Orders sent toward different production agents.
        AgentsC_orderOutP % N_C*N_P: Share of orders (for a particular good) sent toward different production agents (that produces this good).
        AgentsC_OrderOutS % N_C*S: Orders sent for different products.
        AgentsC_orderOutS % N_C*S: Share of orders (in total order of the aggregate product) for different products.
        AgentsC_OrderOutSa % N_C*Sa: Orders sent for different aggregate products.
        
        % State variables in previous periods.
        AgentsC_PP_OrderOutP
        AgentsC_PP2_OrderOutP
        AgentsC_PP_orderOutP
        AgentsC_PP2_orderOutP
    end
    properties % State variables of transportation agents.
        % Production-Agent-To-Production-Agent trasportation lines.
        AgentsT_P2P % Trasportation lines: nl_NetPP * AgentsT_P2P_MaxLength matrix.
        ...Each row is a transportation line, placed at the end of the row, with zeros at the beginning if
        ...this line's length is smaller than the max length of all transportation lines.
        AgentsT_P2P_Lengths % Trasportation line lengths: nl_NetPP*1 vector.
        AgentsT_P2P_MaxLength % Max length of all nl_NetPP transportation lines.
        AgentsT_P2P_StartLinInd % Linear indices of the starting point of each nl_NetPP lines in AgentsT_P2P.
        % Production-Agent-To-Consumption-Agent trasportation lines.
        AgentsT_P2C % Trasportation lines: nl_NetPC * AgentsT_P2C_MaxLength matrix.
        ...Each row is a transportation line, placed at the end of the row, with zeros at the beginning if
        ...this line's length is smaller than the max length of all transportation lines.
        AgentsT_P2C_Lengths % Trasportation line lengths: nl_NetPC*1 vector.
        AgentsT_P2C_MaxLength % Max length of all nl_NetPP transportation lines.
        AgentsT_P2C_StartLinInd % Linear indices of the starting point of each nl_NetPC lines in AgentsT_P2C.
        % Max length of all transportation agents.
        AgentsT_MaxLength
    end
    properties % Steady state variables of production agents.
        AgentsP_SS_Xcap
        AgentsP_SS_a
        AgentsP_SS_OrderInP
        AgentsP_SS_OrderInC
        AgentsP_SS_OrderInE
        AgentsP_SS_orderlnP
        AgentsP_SS_orderInC
        AgentsP_SS_orderInE
    end
    properties % Steady state variables of consumption agents.
        AgentsC_SS_OrderOutP
        AgentsC_SS_OrderOutS
    end
    properties % MRIO variables (optional, NEEDED for initializaton using MRIO data).
        MRIO_R % Number of regions.
        MRIO_S % Number of sectors in each region. Each sector produces one proudct.
        MRIO_Z % Intermediate flows. (RS)*(RS) matrix; Z(i,j) is the product flow from sector i to sector j.
        MRIO_C % Consumption. (RS)*(R) matrix; C(i,j) is the is the consumption of product i by region j.
        MRIO_VA % Value added. 1*(RS) vector; VA(i) is the value added of sector i.
        % The below variables are only defined for the open economy.
        MRIO_E  % Exports. (RS)*1 vector.
        MRIO_IP % Imports by sectors. 1*(RS) vector.
        MRIO_IC % Imports by consumption. 1*R vector.
        % Distance between regions.
        MRIO_Dist % MRIO_R*MRIO_R matrix: simulation steps for transportation from row region to column region.        
    end
    properties % MRIO variables (which could be initialized automatically)
        % Resource constraints IN EACH SIMULATION PERIOD, may or maynot needed to be provided externally.
        ResourceConstraints % MRIO_R*1 vector: Resource constraints for each region in MRIO table.

        % Conversion matrices for MRIO computations
        Regions_Matrix % MRIO_R*N_P matrix that converts rows of region-sectors to rows of regions
        Regions_Matrix_ResourceIntensity % MRIO_R*N_P matrix 
        ... where each row contains the resource intensities of the sectors in the corresponding region.
    end
    methods % Methods for initialization using data from Multi-Regional Input-Output Tables.
        function obj = InitializeBasicVariables_UsingMRIO(obj) % Initialize Basic variables of the world.
            obj.N_P = obj.MRIO_R * obj.MRIO_S;
            obj.N_C = obj.MRIO_R;
            obj.S = obj.MRIO_S;
            obj.Products = kron(ones(obj.MRIO_R,1),(1:obj.MRIO_S)'); % According to the convention of MRIO tables.
            obj.Products_Matrix = kron(ones(1,obj.S), obj.Products)==(1:obj.S);
            % Determine Network Structure.
            % Network connecting producers.
            obj.NetPP = obj.MRIO_Z>obj.CutOff_NetPP;
            obj.nl_NetPP = nnz(obj.NetPP);
            obj.k_NetPP = find(obj.NetPP); 
            obj.i_NetPP = (1:obj.nl_NetPP)';
            % MRIO variables.
            Col_NetPP = ceil(obj.k_NetPP / obj.N_P); % Colunms in NetPP.
            Row_NetPP = obj.k_NetPP - (Col_NetPP - 1) * obj.N_P; % Rows in NetPP.
            obj.SendingRegion_NetPP = ceil(Row_NetPP / obj.MRIO_S);
            obj.ReceivingRegion_NetPP = ceil(Col_NetPP / obj.MRIO_S);
            obj.SendingSector_NetPP = Row_NetPP - (obj.SendingRegion_NetPP - 1) * obj.MRIO_S;
            obj.ReceivingSector_NetPP = Col_NetPP - (obj.ReceivingRegion_NetPP - 1) * obj.MRIO_S;
            % Creating the table.
            obj.AllLinks_NetPP = table(obj.i_NetPP, obj.k_NetPP, obj.SendingRegion_NetPP, obj.ReceivingRegion_NetPP, obj.SendingSector_NetPP, obj.ReceivingSector_NetPP);
            obj.AllLinks_NetPP.Properties.VariableNames = {'i','k','SendingRegion','ReceivingRegion','SendingSector','ReceivingSector'};
            
            % Network connecting producers and consumers.
            obj.NetPC = obj.MRIO_C>obj.CutOff_NetPC;
            obj.nl_NetPC = nnz(obj.NetPC);
            obj.k_NetPC = find(obj.NetPC);
            obj.i_NetPC = (1:obj.nl_NetPC)';
            % MRIO variables.
            Col_NetPC = ceil(obj.k_NetPC / obj.N_P); % Colunms in NetPC.
            Row_NetPC = obj.k_NetPC - (Col_NetPC - 1) * obj.N_P; % Rows in NetPC.
            obj.SendingRegion_NetPC = ceil(Row_NetPC / obj.MRIO_S);
            obj.ReceivingRegion_NetPC = ceil(Col_NetPC);
            obj.SendingSector_NetPC = Row_NetPC - (obj.SendingRegion_NetPC - 1) * obj.MRIO_S;
            % Creating the table.
            obj.AllLinks_NetPC = table(obj.i_NetPC, obj.k_NetPC, obj.SendingRegion_NetPC, obj.ReceivingRegion_NetPC, obj.SendingSector_NetPC);
            obj.AllLinks_NetPC.Properties.VariableNames = {'i','k','SendingRegion','ReceivingRegion','SendingSector'};
            
            % Delete very small flows (i.e., below the cutoff value) in the networks.
            obj.MRIO_Z(obj.MRIO_Z<=obj.CutOff_NetPP) = 0;
            obj.MRIO_C(obj.MRIO_C<=obj.CutOff_NetPC) = 0;
            
            % MRIO_R*N_P matrix that converts rows of region-sectors to rows of regions.
            obj.Regions_Matrix = kron(eye(obj.N_P/obj.S,obj.N_P/obj.S),ones(1,obj.S));
            % MRIO_R*N_P matrix where each row contains the resource intensities of the sectors in the corresponding region.
            obj.Regions_Matrix_ResourceIntensity = obj.Regions_Matrix .* obj.AgentsP_ResourceIntensity';

            % Number of aggregated products.
            obj.Sa = size(obj.S2Sa,2);
        end
        function obj = InitializeProductionAgents_UsingMRIO(obj) % Initialize production agents.
            % State variables of production agents.
            obj.AgentsP_Theta = zeros(obj.N_P,1); % N_P*1: Reduction in production capacity relative to pre-event level, in [0,1].
            obj.AgentsP_Alpha = ones(obj.N_P,1); % N_P*1: Overproduction capacity, in [1, inf).
            obj.AgentsP_Alpha_max = 1.2 * ones(obj.N_P,1); % N_P*1: Maximun possible overproduction capacity, in [1, inf). 
            
            if obj.OpenEcon==false
                obj.MRIO_E = zeros(obj.N_P,1);
                obj.MRIO_IP = zeros(1,obj.N_P);
                obj.MRIO_IC = zeros(1,obj.N_C);
            end
            % Total production, N_P*1.
            MRIO_X = sum([obj.MRIO_Z, obj.MRIO_C, obj.MRIO_E],2);
            % Input requirement for unitary production, (RS)*(RS) matrix.
            MRIO_A = zeros(obj.N_P,obj.N_P);
            ind = MRIO_X>0;
            MRIO_A(:,ind) = obj.MRIO_Z(:,ind) ./ MRIO_X(ind)';
            % Order inflows: Each row represents a order receiver, each column a order sender.
            obj.AgentsP_OrderInP = obj.MRIO_Z * obj.delta_t; % N_P*N_P: Order from production agents.
            obj.AgentsP_OrderInC = obj.MRIO_C * obj.delta_t; % N_P*N_C: Order from consumption agents.
            obj.AgentsP_OrderInE = obj.MRIO_E * obj.delta_t; % N_P*1: Order from export (only for open economy).
            obj.AgentsP_OrderTot = sum([obj.AgentsP_OrderInP, obj.AgentsP_OrderInC, obj.AgentsP_OrderInE], 2); % N_P*1: Total order, sum of the above.
            ind = obj.AgentsP_OrderTot>0;
            obj.AgentsP_orderlnP = zeros(obj.N_P,obj.N_P); % N_P*N_P: Share of order from other production agents in total order.
            obj.AgentsP_orderlnP(ind,:) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInP(ind,:);
            obj.AgentsP_orderInC = zeros(obj.N_P,obj.N_C); % N_P*N_C: Share of order from consumption agents in total order.
            obj.AgentsP_orderInC(ind,:) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInC(ind,:);
            obj.AgentsP_orderInE = zeros(obj.N_P,1); % N_P*1: Share of order from export in total order (only for open economy).
            obj.AgentsP_orderInE(ind) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInE(ind);
            % Product inflows: Each row represents a product receiver, each column a product sender.
            obj.AgentsP_ProductInP = obj.AgentsP_OrderInP'; % N_P*N_P: Product inflow through transportation agents (from production agents);
            obj.AgentsP_ProductSentP = obj.AgentsP_ProductInP; % N_P*N_P: Product sent (toward here) by production agents in the previous period.
            obj.AgentsP_ProductSentS = obj.AgentsP_ProductSentP * obj.Products_Matrix; % N_P*S: Different products sent toward producers at (t-1).
            temp = obj.AgentsP_ProductSentS(:,obj.Products); ind = temp>0;
            obj.AgentsP_productSentP = zeros(obj.N_P,obj.N_P); % N_P*N_P: Share of product sent (toward here) by production agents in a sector.
            obj.AgentsP_productSentP(ind) = obj.AgentsP_ProductSentP(ind) ./ temp(ind); 
            % Production technology: Each row represents a production agent, each column an type of input.
            obj.AgentsP_a = MRIO_A' * obj.Products_Matrix; % N_P*S: Input (intermediate good) requirement for unitary production.
            obj.AgentsP_a = obj.AgentsP_a * obj.S2Sa; % N_P*Sa: Input (intermediate aggregate good) requirement for unitary production.
            obj.AgentsP_va = zeros(obj.N_P,1); % N_P*1: Value-added requirement for unitary production.
            ind = MRIO_X>0;
            obj.AgentsP_va(ind) = ( obj.MRIO_VA(ind) )' ./ MRIO_X(ind);
            obj.AgentsP_im = zeros(obj.N_P,1); % N_P*1: Import requirement for unitary production (only for open economy).
            obj.AgentsP_im(ind) = ( obj.MRIO_IP(ind) )' ./ MRIO_X(ind);
            
            % Production: Each row represents a production agent.
            obj.AgentsP_Xcap = MRIO_X * obj.delta_t; % N_P*1: Production capacity.
            obj.AgentsP_Xs = []; % N_P*S: Possible production levels constrained by inventories of different products.
            ...Did not initialize this variable, because it will be computed in the production stage.
            obj.AgentsP_Xa = obj.AgentsP_Xcap; % N_P*1: Actual production.
            obj.AgentsP_VA = obj.MRIO_VA' * obj.delta_t; % N_P*1: Actual value added.
            % Inventories.
            obj.AgentsP_n = obj.ndays_Target_Default * ones(obj.N_P,obj.Sa); % N_P*Sa: Targeted days of use for different aggregate products.
            ...We know that each row represents a production agent, and each column represents a type of aggregate product.
%             obj.AgentsP_n(:,13) = 2; % The inventory for electricity lasts for only 1 period.
            ...In China's MRIO table of EORA26 database, the electricity，gas,water sector is 25.
            obj.AgentsP_I = obj.AgentsP_n .* ( obj.AgentsP_Xa .* obj.AgentsP_a ); % N_P*Sa: Current inventory level.
            
            % Product Outflows: Each row represents an product sender, each column a product receiver.
            obj.AgentsP_ProductOutP = obj.AgentsP_OrderInP; % N_P*N_P: Product sent toward different production agents (through transportation agents).
            obj.AgentsP_ProductOutC = obj.AgentsP_OrderInC; % N_P*N_C: Product sent toward different consumption agents (through transportation agents).
            obj.AgentsP_ProductOutE = obj.AgentsP_OrderInE; % N_P*1: Product sent toward export (only for open economy).
            % Order Outflows: Each row represents an order sender, each column an order receiver.
            obj.AgentsP_OrderOutP = obj.AgentsP_OrderInP'; % N_P*N_P: Orders sent toward different production agents.
            obj.AgentsP_OrderOutS = obj.AgentsP_OrderOutP * obj.Products_Matrix; % N_P*S: Orders sent for different products.
            temp = obj.AgentsP_OrderOutS(:,obj.Products); ind = temp>0;
            obj.AgentsP_orderOutP = zeros(obj.N_P,obj.N_P); % N_P*N_P: Share of orders (in total order of the product) sent toward different production agents.
            obj.AgentsP_orderOutP(ind) = obj.AgentsP_OrderOutP(ind) ./ temp(ind);
            
            % State variables in previous periods.
            obj.AgentsP_PP_a = obj.AgentsP_a;
            obj.AgentsP_PP_Xa = obj.AgentsP_Xa;
            obj.AgentsP_PP_OrderOutP = obj.AgentsP_OrderOutP;
            obj.AgentsP_PP2_OrderOutP = obj.AgentsP_OrderOutP;
            obj.AgentsP_PP_orderOutP = obj.AgentsP_orderOutP;
            obj.AgentsP_PP2_orderOutP = obj.AgentsP_orderOutP;
            obj.AgentsP_PP_OrderOutS = obj.AgentsP_OrderOutS;
            obj.AgentsP_PP2_OrderOutS = obj.AgentsP_OrderOutS;
            
            % Steady state variables of production agents.
            obj.AgentsP_SS_Xcap = obj.AgentsP_Xcap;
            obj.AgentsP_SS_a = obj.AgentsP_a;
            obj.AgentsP_SS_OrderInP = obj.AgentsP_OrderInP;
            obj.AgentsP_SS_OrderInC = obj.AgentsP_OrderInC;
            obj.AgentsP_SS_OrderInE = obj.AgentsP_OrderInE;
            obj.AgentsP_SS_orderlnP = obj.AgentsP_orderlnP;
            obj.AgentsP_SS_orderInC = obj.AgentsP_orderInC;
            obj.AgentsP_SS_orderInE = obj.AgentsP_orderInE;

            % Variables related to the aggregate products.
            obj.AgentsP_ProductSentSa = obj.AgentsP_ProductSentS * obj.S2Sa; % N_P*Sa: Different aggregate products sent toward producers at (t-1).
            obj.AgentsP_OrderOutSa = obj.AgentsP_OrderOutS * obj.S2Sa; % N_P*Sa: Orders sent for different aggregate products.
            obj.AgentsP_orderOutS = zeros(size(obj.AgentsP_OrderOutS)); % N_P*S: Share of orders (in total order of the aggregate product) for different products.
            temp = obj.AgentsP_OrderOutSa * obj.S2Sa';
            ind = temp>0;
            obj.AgentsP_orderOutS(ind) = obj.AgentsP_OrderOutS(ind) ./ temp(ind);
            % Variables in previous periods.
            obj.AgentsP_PP_OrderOutSa = obj.AgentsP_OrderOutSa;
            obj.AgentsP_PP2_OrderOutSa = obj.AgentsP_OrderOutSa;

            % Initiate resource constraints. Higher than normal use.
            obj.ResourceConstraints = obj.Regions_Matrix_ResourceIntensity * obj.AgentsP_SS_Xcap + eps;
        end
        function obj = InitializeConsumptionAgents_UsingMRIO(obj) % Initialize consumption agents.
            % Product inflows: Each row represents a consumer agent.
            obj.AgentsC_ProductInP = obj.MRIO_C' * obj.delta_t; % N_C*N_P: Product inflow through transportation agents (from production agents).
            obj.AgentsC_ProductSentP = obj.AgentsC_ProductInP; % N_C*N_P: Product sent (toward here) by production agents in.
            obj.AgentsC_ProductSentS = obj.AgentsC_ProductSentP * obj.Products_Matrix; % N_C*S: Different products sent toward consumers at (t-1).
            % Shares in product inflows.
            obj.AgentsC_productSentP = zeros(obj.N_C,obj.N_P); % N_C*N_P: Share of product sent (toward here) by production agents in the previous period.
            temp = obj.AgentsC_ProductSentS(:,obj.Products); ind = temp>0;
            obj.AgentsC_productSentP(ind) = obj.AgentsC_ProductSentP(ind) ./ temp(ind);
            % Import requirement (i.e., order) for consumption, only for open economy.
            obj.AgentsC_IMO = obj.MRIO_IC' * obj.delta_t; % N_C*1.
            % Order Outflows.
            obj.AgentsC_OrderOutP = obj.AgentsC_ProductInP; % N_C*N_P: Orders sent toward different production agents.
            obj.AgentsC_orderOutP = obj.AgentsC_productSentP; % N_C*N_P: Share of orders (for a particular good) sent toward different production agents (that produces this good).
            obj.AgentsC_OrderOutS = obj.AgentsC_ProductSentS; % N_C*S: Orders sent for different products.
            
            % State variables in previous periods.
            obj.AgentsC_PP_OrderOutP = obj.AgentsC_OrderOutP;
            obj.AgentsC_PP2_OrderOutP = obj.AgentsC_OrderOutP;
            obj.AgentsC_PP_orderOutP = obj.AgentsC_orderOutP;
            obj.AgentsC_PP2_orderOutP = obj.AgentsC_orderOutP;
            
            % Steady state variables of consumption agents.
            obj.AgentsC_SS_OrderOutP = obj.AgentsC_OrderOutP;
            obj.AgentsC_SS_OrderOutS = obj.AgentsC_OrderOutS;
        end
        function obj = InitializeTransportationAgents_UsingMRIO(obj) % Initialize transportation agents.
            % Initialize Production-Agent-To-Production-Agent trasportation line lengths.
            [x, y] = ind2sub([obj.N_P, obj.N_P], obj.k_NetPP);
            x_R = ceil( x / obj.MRIO_S ); y_R = ceil( y / obj.MRIO_S );
            x_S = x - (x_R - 1) * obj.MRIO_S; y_S = y - (y_R - 1) * obj.MRIO_S;
            ind = sub2ind([obj.MRIO_R, obj.MRIO_R], x_R, y_R); % Linear indices in the R*R matrix.
            obj.AgentsT_P2P_Lengths = obj.MRIO_Dist(ind);
%             for i=1:obj.nl_NetPP % Adjusting transportation lengths for certain sectors.
%                 if (x_S(i)==25) % If for the ith link, the starting sector is electricity, then the transportation length is set to 0.
%                     obj.AgentsT_P2P_Lengths(i) = 0;
%                 end
%             end
            obj.AgentsT_P2P_MaxLength = max(obj.AgentsT_P2P_Lengths);        
            % Initialize Production-Agent-To-Consumption-Agent trasportation line lengths.
            [x, y] = ind2sub([obj.N_P, obj.N_C], obj.k_NetPC);
            x_R = ceil( x / obj.MRIO_S ); y_R = y;
            x_S = x - (x_R - 1) * obj.MRIO_S;
            ind = sub2ind([obj.MRIO_R, obj.MRIO_R], x_R, y_R); % Linear indices in the R*R matrix.
            obj.AgentsT_P2C_Lengths = obj.MRIO_Dist(ind);
%             for i=1:obj.nl_NetPC % Adjusting transportation lengths for certain sectors.
%                 if (x_S(i)==25) % If for the ith link, the starting sector is electricity, then the transportation length is set to 0.
%                     obj.AgentsT_P2C_Lengths(i) = 0;
%                 end
%             end
            obj.AgentsT_P2C_MaxLength = max(obj.AgentsT_P2C_Lengths);
            
            % Calculate the max length of the above two.
            obj.AgentsT_MaxLength = max(obj.AgentsT_P2P_MaxLength, obj.AgentsT_P2C_MaxLength);
            
            % Initialize Production-Agent-To-Production-Agent trasportation lines!
            obj.AgentsT_P2P = zeros(obj.nl_NetPP, obj.AgentsT_MaxLength);
            for i=1:obj.nl_NetPP
                obj.AgentsT_P2P(i, (end-obj.AgentsT_P2P_Lengths(i)+1):end) = obj.AgentsP_OrderInP(obj.k_NetPP(i));
            end
            obj.AgentsT_P2P_StartLinInd = sub2ind( size(obj.AgentsT_P2P)+[0,1], (1:obj.nl_NetPP)', obj.AgentsT_MaxLength - obj.AgentsT_P2P_Lengths + 1 );
            % Initialize Production-Agent-To-Consumption-Agent trasportation lines!
            obj.AgentsT_P2C = zeros(obj.nl_NetPC, obj.AgentsT_MaxLength);
            temp = obj.AgentsC_ProductInP';
            for i=1:obj.nl_NetPC
                obj.AgentsT_P2C(i, (end-obj.AgentsT_P2C_Lengths(i)+1):end) = temp(obj.k_NetPC(i));
            end
            obj.AgentsT_P2C_StartLinInd = sub2ind( size(obj.AgentsT_P2C)+[0,1], (1:obj.nl_NetPC)', obj.AgentsT_MaxLength - obj.AgentsT_P2C_Lengths + 1 );
        end
    end
    methods % Methods for step-by-step simulation.
        function obj = AgentsCommunicate(obj) % Agents communicate.
            % Production agents communicate.
            % Order information flows.
            obj.AgentsP_OrderInP = obj.AgentsP_OrderOutP';
            % Sent-products information flows.
            obj.AgentsP_ProductSentP = obj.AgentsP_ProductOutP';
            
            % Production and consumption agents communicate.
            % Order information flows.
            obj.AgentsP_OrderInC = obj.AgentsC_OrderOutP';
            % Sent-products information flows.
            obj.AgentsC_ProductSentP = obj.AgentsP_ProductOutC';
        end
        function obj = UpdateInventories(obj) % Update inventories of production agents.
            obj.AgentsP_I = obj.AgentsP_I - obj.AgentsP_PP_Xa .* obj.AgentsP_PP_a ...
                + obj.AgentsP_ProductInP * obj.Products_Matrix * obj.S2Sa;
        end
        function obj = UpdateExportOrders(obj) % Update export orders (only for open economy).
            if obj.OpenEcon
                ind1  = obj.AgentsP_OrderInE>obj.AgentsP_ProductOutE; % If supply does not meet demand.
                ind2  = (obj.AgentsP_OrderInE<=obj.AgentsP_ProductOutE) & (obj.AgentsP_SS_OrderInE>0); % If supply does meet demand.
                obj.AgentsP_OrderInE(ind1) = obj.AgentsP_OrderInE(ind1) ...
                    - (obj.AgentsP_OrderInE(ind1)-obj.AgentsP_ProductOutE(ind1))./obj.AgentsP_OrderInE(ind1) .* obj.AgentsP_OrderInE(ind1) * (obj.delta_t/obj.tau_E);
                obj.AgentsP_OrderInE(ind2) = obj.AgentsP_OrderInE(ind2) ...
                    + ( obj.AgentsP_SS_OrderInE(ind2) - obj.AgentsP_OrderInE(ind2) ) ./ obj.AgentsP_SS_OrderInE(ind2) ...
                    .* ( obj.AgentsP_SS_OrderInE(ind2) - obj.AgentsP_OrderInE(ind2) ) * (obj.delta_t/obj.tau_E);
            end
        end
        function obj = UpdateShares(obj) % Update shares.
            % Update shares of the consumption agents.
            % Different products sent toward consumers at (t-1).
            % N_C*S. Each row is a consumer; each column a product.
            obj.AgentsC_ProductSentS = obj.AgentsC_ProductSentP * obj.Products_Matrix;
            temp = obj.AgentsC_ProductSentS(:,obj.Products); ind = temp>0;
            obj.AgentsC_productSentP(ind) = obj.AgentsC_ProductSentP(ind) ./ temp(ind);
            
            % Update shares of the production agents.
            obj.AgentsP_OrderTot = sum( [obj.AgentsP_OrderInP, obj.AgentsP_OrderInC, obj.AgentsP_OrderInE], 2 );
            ind = obj.AgentsP_OrderTot>0;
            obj.AgentsP_orderlnP(ind,:) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInP(ind,:);
            obj.AgentsP_orderInC(ind,:) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInC(ind,:);
            obj.AgentsP_orderInE(ind) = obj.AgentsP_OrderTot(ind) .\ obj.AgentsP_OrderInE(ind);
            % Different products sent toward producers at (t-1).
            % N_P*S. Each row is a producer; each column a product.
            obj.AgentsP_ProductSentS = obj.AgentsP_ProductSentP * obj.Products_Matrix;
            temp = obj.AgentsP_ProductSentS(:,obj.Products); ind = temp>0;
            obj.AgentsP_productSentP(ind) = obj.AgentsP_ProductSentP(ind) ./ temp(ind);
            % N_P*Sa: Different aggregate products sent toward producers at (t-1).
            obj.AgentsP_ProductSentSa = obj.AgentsP_ProductSentS * obj.S2Sa;
        end
        function obj = ProductionAgentsProduce(obj) % Production agents produce.
            % Update production capacity.
            obj.AgentsP_Xcap = obj.AgentsP_Alpha .* (1- obj.AgentsP_Theta) .* obj.AgentsP_SS_Xcap;
            % Update possible production levels constrained by inventories of different products.
            obj.AgentsP_Xs = (obj.AgentsP_I + (obj.AgentsP_a==0)) ./ obj.AgentsP_a;
            % Calculate the upper and lower bounds of production.
            ub = min([obj.AgentsP_Xcap, obj.AgentsP_OrderTot, obj.AgentsP_Xs] ,[], 2); % Upper bound.
            ub(ub<0) = 0; % Upperbound must be >=0.
            lb = zeros(obj.N_P,1); % Lower bound.
            lb(obj.AgentsP_SS_Xcap>0) = obj.AgentsP_SS_Xcap(obj.AgentsP_SS_Xcap>0) * 1e-2;
            % Update actual production levels using linear optimazation.    
            obj.AgentsP_Xa = ... 
                linprog(-ones(obj.N_P,1),obj.Regions_Matrix_ResourceIntensity,obj.ResourceConstraints,[],[],lb,ub);
            % Update value added.
            obj.AgentsP_VA = obj.AgentsP_Xa .* obj.AgentsP_va;
           
            % Calculate scarcity Indices for each product in each region.
            % Products comming in (i.e., supplied).
            AgentsP_ProductInSa = obj.AgentsP_ProductInP * obj.Products_Matrix * obj.S2Sa;
            AgentsC_ProductInSa = obj.AgentsC_ProductInP * obj.Products_Matrix * obj.S2Sa;
            % R * Sa matrix: rows are regions, and columns are aggregate products.
            obj.RegionsProducts_Supplied = obj.Regions_Matrix * AgentsP_ProductInSa + AgentsC_ProductInSa;

            % Products demanded.
            AgentsP_Xa_1 = min([obj.AgentsP_Xcap, obj.AgentsP_OrderTot] ,[], 2); % Production levels under capacity and orders.
            AgentsP_ProductDemandedSa = AgentsP_Xa_1 .* obj.AgentsP_a;
            % R * Sa matrix: rows are regions, and columns are aggregate products.
            % Each demand is the sum of producer demand and consumer demand.
            obj.RegionsProducts_Demanded = obj.Regions_Matrix * AgentsP_ProductDemandedSa + obj.AgentsC_OrderOutS * obj.S2Sa;

            % Calculate scarcity of region sectors (with respect to consumption).
            obj.Scarcity_RegionsProducts = zeros(obj.N_P/obj.S,obj.Sa);
            idx = obj.RegionsProducts_Demanded>0;
            obj.Scarcity_RegionsProducts(idx) = (obj.RegionsProducts_Demanded(idx) - obj.RegionsProducts_Supplied(idx)) ./ obj.RegionsProducts_Demanded(idx);
        end
        function obj = ProductionAgentsPrepareProductOut(obj) % Production agents prepare product outflows.
            % If no scarcity.
            ind = obj.AgentsP_Xa>=obj.AgentsP_OrderTot;
            obj.AgentsP_ProductOutP(ind,:) = obj.AgentsP_OrderInP(ind,:);
            obj.AgentsP_ProductOutC(ind,:) = obj.AgentsP_OrderInC(ind,:);
            obj.AgentsP_ProductOutE(ind,:) = obj.AgentsP_OrderInE(ind,:);
            % Under scarcity.
            ind = obj.AgentsP_Xa<obj.AgentsP_OrderTot;
            obj.AgentsP_ProductOutP(ind,:) = obj.AgentsP_Xa(ind,:) .* obj.AgentsP_SS_orderlnP(ind,:);
            obj.AgentsP_ProductOutC(ind,:) = obj.AgentsP_Xa(ind,:) .* obj.AgentsP_SS_orderInC(ind,:);
            obj.AgentsP_ProductOutE(ind,:) = obj.AgentsP_Xa(ind,:) .* obj.AgentsP_SS_orderInE(ind,:);
        end
        function obj = ProductionAgentsPrepareOrderOut(obj) % Production agents prepare order outflows.
            % AgentsP_IT is targeted invetory levels: N_P*Sa.
            % Each row represents a production agent and column a type of aggregated product.
            temp = min([ obj.AgentsP_OrderTot obj.AgentsP_Xcap ], [], 2);
            AgentsP_IT = obj.AgentsP_n .* ( temp .* obj.AgentsP_a );
            % AgentsP_OrderOutSa is orders for different aggregate products: N_P*Sa.
            % AgentsP_OrderOutS is orders for different products: N_P*S.
            % Each row represents a production agent and each column a type of product.
            obj.AgentsP_OrderOutSa = obj.AgentsP_Xa .* obj.AgentsP_a + ( AgentsP_IT - obj.AgentsP_I ) * ( obj.delta_t / obj.tau_I );
            obj.AgentsP_OrderOutSa(obj.AgentsP_OrderOutSa<0) = 0;
            obj.AgentsP_OrderOutS = obj.AgentsP_OrderOutSa * obj.S2Sa' .* obj.AgentsP_orderOutS;
            obj.AgentsP_OrderOutS(obj.AgentsP_OrderOutS<0) = 0;
            
            % Share of orders sent toward different production agents.
            obj.AgentsP_orderOutP = obj.AgentsP_orderOutP + ( obj.AgentsP_productSentP - obj.AgentsP_PP2_orderOutP ) * (obj.delta_t/obj.tau_O);
            % Orders sent toward different production agents.
            obj.AgentsP_OrderOutP = obj.AgentsP_OrderOutS(:,obj.Products);
            obj.AgentsP_OrderOutP = obj.AgentsP_OrderOutP .* obj.AgentsP_orderOutP;
        end
        function obj = ProductionAgentsAdaptToShocks(obj) % Production agents adpat to shocks.
            % Reconstruct.
            obj.AgentsP_Theta = (1 - obj.delta_t/obj.tau_Theta) .* obj.AgentsP_Theta;
            
            % Adjust overproduction capacity.
            % If production capacity does not meet demand.
            ind = obj.AgentsP_Xcap < obj.AgentsP_OrderTot;
            obj.AgentsP_Alpha(ind)  = obj.AgentsP_Alpha(ind) + ...
                ( obj.AgentsP_Alpha_max(ind) - obj.AgentsP_Alpha(ind) ) .* ( (obj.AgentsP_OrderTot(ind) - obj.AgentsP_Xcap(ind) ) ./ obj.AgentsP_OrderTot(ind) ) * ( obj.delta_t/obj.tau_Alpha );
            % If production capacity can satisfy demand.
            ind = obj.AgentsP_Xcap >= obj.AgentsP_OrderTot;
            obj.AgentsP_Alpha(ind) = obj.AgentsP_Alpha(ind) - ( obj.AgentsP_Alpha(ind) - 1 ) *  ( obj.delta_t/obj.tau_Alpha );
        end
        function obj = ProductionAgentsAdaptToShortages(obj) % Production agents adpat production modes to shortages.
            % Remember the unadapted production technology.
            obj.AgentsP_PP_a = obj.AgentsP_a;
            
            % SCARCITY of different products perceived by producers.
            % N_P*Sa. Each row is a producer; each column an aggregate product.
            ScarcityIndex = zeros(obj.N_P,obj.Sa);
            % If supply cannot meet demand.
            ind = obj.AgentsP_PP2_OrderOutSa > (obj.AgentsP_ProductSentSa + 1e-8);
            ScarcityIndex(ind) = ( obj.AgentsP_PP2_OrderOutSa(ind) - obj.AgentsP_ProductSentSa(ind) ) ./ obj.AgentsP_PP2_OrderOutSa(ind);
            if obj.OpenEcon
                temp = zeros(obj.N_P,obj.Sa);
                temp(ind) = ScarcityIndex(ind) .* obj.AgentsP_a(ind) * (obj.delta_t/obj.tau_A); % Reduced input requirements.
                obj.AgentsP_im = obj.AgentsP_im + sum(temp,2);
            end
            obj.AgentsP_a(ind) = obj.AgentsP_a(ind) - ScarcityIndex(ind) .* obj.AgentsP_a(ind) * (obj.delta_t/obj.tau_A); % Reducing use of scarce inputs.
            % If supply can meet demand.
            ind = ( obj.AgentsP_PP2_OrderOutSa <= (obj.AgentsP_ProductSentSa + 1e-8) ) & ( obj.AgentsP_SS_a > 0 );
            if obj.OpenEcon
                temp = zeros(obj.N_P,obj.Sa);
                temp(ind) = ( obj.AgentsP_SS_a(ind)  - obj.AgentsP_a(ind) ) ./ ( obj.AgentsP_SS_a(ind) ) .* ( obj.AgentsP_SS_a(ind)  - obj.AgentsP_a(ind) ) * (obj.delta_t/obj.tau_A);
                obj.AgentsP_im = obj.AgentsP_im - sum(temp,2);
            end
            % Update intermediate requirement for unitary production.
            obj.AgentsP_a(ind) = obj.AgentsP_a(ind) + ...
                ( obj.AgentsP_SS_a(ind)  - obj.AgentsP_a(ind) ) ./ ( obj.AgentsP_SS_a(ind) ) .* ( obj.AgentsP_SS_a(ind)  - obj.AgentsP_a(ind) ) * (obj.delta_t/obj.tau_A);
        end
        function obj = ProductionAgentsRemember(obj) % Production agents remember key state variables.
            obj.AgentsP_PP_Xa = obj.AgentsP_Xa;
            obj.AgentsP_PP2_OrderOutP = obj.AgentsP_PP_OrderOutP;
            obj.AgentsP_PP_OrderOutP = obj.AgentsP_OrderOutP;
            obj.AgentsP_PP2_orderOutP = obj.AgentsP_PP_orderOutP;
            obj.AgentsP_PP_orderOutP = obj.AgentsP_orderOutP;
            obj.AgentsP_PP2_OrderOutS = obj.AgentsP_PP_OrderOutS;
            obj.AgentsP_PP_OrderOutS = obj.AgentsP_OrderOutS;
            obj.AgentsP_PP2_OrderOutSa = obj.AgentsP_PP_OrderOutSa;
            obj.AgentsP_PP_OrderOutSa = obj.AgentsP_OrderOutSa;
        end
        function obj = ConsumptionAgentsConsume(obj) % Consumption agents consume.
        end
        function obj = ConsumptionAgentsPrepareOrderOut(obj) % Consumption agents prepare order outflows.
            % Adjust domestic order shares.
            obj.AgentsC_orderOutP = obj.AgentsC_orderOutP + (obj.AgentsC_productSentP-obj.AgentsC_PP2_orderOutP) * ( obj.delta_t/obj.tau_O );
            % Orders sent toward different production agents.
            obj.AgentsC_OrderOutP = obj.AgentsC_SS_OrderOutS(:,obj.Products);
            obj.AgentsC_OrderOutP = obj.AgentsC_OrderOutP .* obj.AgentsC_orderOutP;
            obj.AgentsC_OrderOutS = obj.AgentsC_OrderOutP * obj.Products_Matrix;
        end
        function obj = ConsumptionAgentsRemember(obj) % Consumption agents remember key state variables
            obj.AgentsC_PP2_OrderOutP = obj.AgentsC_PP_OrderOutP;
            obj.AgentsC_PP_OrderOutP = obj.AgentsC_OrderOutP;
            obj.AgentsC_PP2_orderOutP = obj.AgentsC_PP_orderOutP;
            obj.AgentsC_PP_orderOutP = obj.AgentsC_orderOutP;
        end
    end
    methods % Transportation chains obstruction.
        function obj = TransportObstruct_MRIO(obj, r1, r2, s, position_fraction, product_blocked) % Model obstruction in a trasportation lines.
            % r1: origin region.
            % r2: destination region.
            % s: product (same as sector).
            % position_fraction: in (0, 1], the position in the tranporaton line where the obstruction occurs.
            % product_blocked: the amount of product blocked in the above position.
            
            % Find corresponding rows (i.e., link indices) in tables AllLinks_NetPP and AllLinks_NetPC.
            idx1 = obj.AllLinks_NetPP.i((obj.AllLinks_NetPP.SendingRegion == r1) & (obj.AllLinks_NetPP.ReceivingRegion == r2) & (obj.AllLinks_NetPP.SendingSector == s));      
            idx2 = obj.AllLinks_NetPC.i((obj.AllLinks_NetPC.SendingRegion == r1) & (obj.AllLinks_NetPC.ReceivingRegion == r2) & (obj.AllLinks_NetPC.SendingSector == s));
            % 
            SupplyChains = [obj.AgentsT_P2P(idx1,:); obj.AgentsT_P2C(idx2,:)];
            SupplyChains_Length = unique([obj.AgentsT_P2P_Lengths(idx1); obj.AgentsT_P2C_Lengths(idx2)]);
            SupplyChains_Position = obj.AgentsT_MaxLength - SupplyChains_Length + ceil(SupplyChains_Length * position_fraction);
            
            % Total product in the blocked position.
            total = sum(SupplyChains(:,SupplyChains_Position+1));
            product_blocked_true = min(total, product_blocked);
            % Blockage (i.e., obstruction) occurs!
            if product_blocked_true>0
                ratio_blocked = product_blocked_true / total;
                SupplyChains(:,SupplyChains_Position) = SupplyChains(:,SupplyChains_Position) + ratio_blocked * SupplyChains(:,SupplyChains_Position+1);
                SupplyChains(:,SupplyChains_Position+1) = SupplyChains(:,SupplyChains_Position+1) * (1 - ratio_blocked);
            end
            % Assign the total (length(idx1) + length(idx2)) lines back to the (augumented) AgentsT_P2P and AgentsT_P2C.
            obj.AgentsT_P2P(idx1,:) = SupplyChains( 1:length(idx1), : );
            obj.AgentsT_P2C(idx2,:) = SupplyChains( (length(idx1)+1) : (length(idx1) + length(idx2)), : );
        end
    end
end