import numpy as np
import pandas as pd


class WorldOfMatrixGPU:
    """
    An agent-based model for Climate-resilient and Low-carbon Unfolding Economic Scenarios (CLUES-ABM)
    Developed by Shen Qu, CEEP-BIT, Version 4.0, 2023/03/31
    """

    # ======= Static Class Info (originally Constant properties in MATLAB) =======
    Name = "(An agent-based model for) Climate-resilient and Low-carbon Unfolding Economic Scenarios"
    Acronym = "CLUES-ABM"
    Institution = "CEEP-BIT"
    Designer = "Shen Qu"
    Version = "4.0"
    Date = "2023/03/31"

    def __init__(self):
        # ======== properties # Values relating to time ========
        self.delta_t = 1/365 # Length of each time step, as a fraction of input flows.
        self.tau_Theta = 1/2 # Timescale for reconstruction.
        self.tau_Alpha = 1/2 # Timescale for adjusting to maximum production capacity.
        self.tau_I = 1/2 # Timescale for adjusting to targeted inventory levels.
        self.tau_O = 1/2 # Timescale for adjusting to targeted order distributions.
        self.tau_A = 1/2 # Timescale for technology adaptation.
        self.tau_E = 1/2 # Timescale for export demand adaptation.
        self.tau = 1/2 # Timescale, generic.
        
        # ======== Basic variables of the world ========
        self.N_P = None # Number of production agents. Eech production agents produces one product.
        self.N_C = None # Number of consumption agents.
        self.S = None # Number of products.
        self.Sa = None # Number of aggregated products.
        self.ndays_Target_Default = None # Default targeted inventory days for different intemediate products used by different production agents.
        self.Products = None # N_P*1 vector for the index of the product produced by each production agent.
        self.Products_Matrix = None # N_P*S logical matrix that indicates whether the row producer produces the column product.
        self.S2Sa = None # S * Sa matrix, each row is a product, and the value is 1 if the column is aggregate product for the row product, otherwise is 0.
        
        # ======== Network connecting producers ========
        self.NetPP = None # N_P*N_P matrix: 1 means that row producer sends products to column producer; 0 means otherwise.
        self.nl_NetPP = None # Number of links in NetPP.
        self.k_NetPP = None # nl_NetPP*1 vector: Linear indices of non-zero elements in NetPP.
        self.CutOff_NetPP = 1 # Cutoff value for flows to be included in the network.
        self.DistPP = None # N_P*N_P matrix: tranportation line lengths (>=0) from row producer to column producer. 
        self.i_NetPP = None # nl_NetPP*1 vector: Indices of non-zero elements in NetPP.
        self.SendingRegion_NetPP = None# nl_NetPP*1 vector: Sending regions of non-zero elements in NetPP.
        self.ReceivingRegion_NetPP = None # nl_NetPP*1 vector: Receiving Regions of non-zero elements in NetPP.
        self.SendingSector_NetPP = None # nl_NetPP*1 vector: Sending sectors of non-zero elements in NetPP.
        self.ReceivingSector_NetPP = None # nl_NetPP*1 vector: Receiving sectors of non-zero elements in NetPP.
        self.AllLinks_NetPP = None # An nl_NetPP*6 table for all the Production-to-Production-Agent links.
        # Column 1: i_NetPP
        # Column 2: k_NetPP
        # Column 3: SendingRegion_NetPP
        # Column 4: ReceivingRegion_NetPP
        # Column 5: SendingSector_NetPP
        # Column 6: ReceivingSector_NetPP
        
        # ======== Network connecting producers and consumers ========
        self.NetPC = None # N_P*N_C matrix: 1 means that row producer sends products to column consumer; 0 means otherwise.
        self.nl_NetPC = None # Number of links in NetPC.
        self.k_NetPC = None # nl_NetPC*1 vector: Linear indices of non-zero elements in NetPC.
        self.CutOff_NetPC = 1 # Cutoff value for flows to be included in the network.
        self.DistPC = None # N_P*N_C matrix: tranportation line lengths (>=0) from row producer to column consumer. 
        self.i_NetPC = None # nl_NetPC*1 vector: Indices of non-zero elements in NetPC.
        self.SendingRegion_NetPC = None # nl_NetPC*1 vector: Sending regions of non-zero elements in NetPC.
        self.ReceivingRegion_NetPC = None # nl_NetPC*1 vector: Receiving Regions of non-zero elements in NetPC.
        self.SendingSector_NetPC = None # nl_NetPC*1 vector: Sending sectors of non-zero elements in NetPC.
        self.AllLinks_NetPC = None # nl_NetPC*5 table for all the Production-to-Production-Agent links.
        # Column 1: i_NetPC
        # Column 2: k_NetPC
        # Column 3: SendingRegion_NetPC
        # Column 4: ReceivingRegion_NetPC
        # Column 5: SendingSector_NetPC
        
        # ======== Economy Type ========
        self.OpenEcon = False # Whether this is an open economy. true means open; false means closed.

        # ======== Scarcity Indices for each aggregated product in each region ========
        # Defined as: (Product demanded - Product supplied) / (Product demanded). (It is 0 if Product demanded is 0.)
        self.Scarcity_RegionsProducts = None # Scarcity Indices. R * Sa matrix: rows are regions, and columns are aggregated products.
        self.RegionsProducts_Supplied = None # Products comming in (or supplied). R * Sa matrix: rows are regions, and columns are aggregated products.
        self.RegionsProducts_Demanded = None # Products demanded. R * Sa matrix: rows are regions, and columns are aggregated products.
        
        # ======== State variables of production agents ========
        # Production Agents.
        self.AgentsP_Theta = None # N_P*1: Reduction in production capacity relative to pre-event level, in [0,1].
        self.AgentsP_Alpha = None # N_P*1: Overproduction capacity, in [1, inf).
        self.AgentsP_Alpha_max = None # N_P*1: Maximun possible overproduction capacity, in [1, inf).
        # Order inflows: Each row represents a order receiver, each column a order sender.
        self.AgentsP_OrderInP = None # N_P*N_P: Order from production agents.
        self.AgentsP_OrderInC = None # N_P*N_C: Order from consumption agents.
        self.AgentsP_OrderInE = None # N_P*1: Order from export (only for open economy).
        self.AgentsP_OrderTot = None # N_P*1: Total order, sum of the above.
        self.AgentsP_orderlnP = None # N_P*N_P: Share of order from other production agents in total order.
        self.AgentsP_orderInC = None # N_P*N_C: Share of order from consumption agents in  total order.
        self.AgentsP_orderInE = None # N_P*1: Share of order from export in total order (only for open economy).
        # Product inflows: Each row represents a product receiver, each column a product sender.
        self.AgentsP_ProductInP = None # N_P*N_P: Product inflow through transportation agents (from production agents);
        self.AgentsP_ProductSentP = None # N_P*N_P: Product sent (toward here) by production agents in the previous period.
        self.AgentsP_ProductSentS = None # N_P*S: Different products sent toward producers at (t-1).
        self.AgentsP_ProductSentSa = None # N_P*Sa: Different aggregate products sent toward producers at (t-1).
        self.AgentsP_productSentP = None # N_P*N_P: Share of product sent (toward here) by production agents in a sector.
        # Production technology: Each row represents a production agent, each column an type of input.
        self.AgentsP_a = None # N_P*Sa: Input (intermediate good) requirement for unitary production.
        self.AgentsP_va = None # N_P*1: Value-added requirement for unitary production.
        self.AgentsP_im = None # N_P*1: Import requirement for unitary production (only for open economy).
        # Inventories:
        self.AgentsP_n = None # N_P*Sa: Targeted days of use for different products. TO BE INITIALIZED!
        self.AgentsP_I = None # N_P*Sa: Current inventory level.
        # Production: Each row represents a production agent.
        self.AgentsP_Xcap = None # N_P*1: Production capacity.
        self.AgentsP_Xs = None # N_P*Sa: Possible production levels constrained by inventories of different products.
        self.AgentsP_Xa = None # N_P*1: Actual production.
        self.AgentsP_VA = None # N_P*1: Actual value added.
        self.AgentsP_WaterIntensity = None # N_P*1: Water required for unitary output of each production agent.
        # Product Outflows: Each row represents an product sender, each column a product receiver.
        self.AgentsP_ProductOutP = None # N_P*N_P: Product sent toward different production agents (through transportation agents).
        self.AgentsP_ProductOutC = None # N_P*N_C: Product sent toward different consumption agents (through transportation agents).
        self.AgentsP_ProductOutE = None # N_P*1: Product sent toward export (only for open economy).
        # Order Outflows: Each row represents an order sender, each column an order receiver.
        self.AgentsP_OrderOutP = None # N_P*N_P: Orders sent toward different production agents.
        self.AgentsP_orderOutP = None # N_P*N_P: Share of orders (in total order of the product) sent toward different production agents.
        self.AgentsP_OrderOutS = None # N_P*S: Orders sent for different products.
        self.AgentsP_orderOutS = None # N_P*S: Share of orders (in total order of the aggregate product) for different products.
        self.AgentsP_OrderOutSa = None # N_P*Sa: Orders sent for different aggregate products.
        # State variables in previous periods.
        self.AgentsP_PP_a = None
        self.AgentsP_PP_Xa = None
        self.AgentsP_PP_OrderOutP = None
        self.AgentsP_PP2_OrderOutP = None
        self.AgentsP_PP_orderOutP = None
        self.AgentsP_PP2_orderOutP = None
        self.AgentsP_PP_OrderOutS = None
        self.AgentsP_PP2_OrderOutS = None
        self.AgentsP_PP_OrderOutSa = None
        self.AgentsP_PP2_OrderOutSa = None
        
        # ======== State variables of consumption agents ========
        # Product inflows: Each row represents a consumer agent.
        self.AgentsC_ProductInP = None # N_C*N_P: Product inflow through transportation agents (from production agents).
        self.AgentsC_ProductSentP = None # N_C*N_P: Product sent (toward here) by production agents.
        self.AgentsC_ProductSentS = None # N_C*S: Different products sent toward consumers at (t-1).
        self.AgentsC_ProductSentSa = None # N_C*Sa: Different aggregate products sent toward consumers at (t-1).
        # Shares in product inflows.
        self.AgentsC_productSentP = None # N_C*N_P: Share of product sent (toward here) by production agents in the previous period.
        # Import requirement (i.e., order) for consumption, only for open economy.
        self.AgentsC_IMO = None # N_C*1.
        # Order Outflows.
        self.AgentsC_OrderOutP = None # N_C*N_P: Orders sent toward different production agents.
        self.AgentsC_orderOutP = None # N_C*N_P: Share of orders (for a particular good) sent toward different production agents (that produces this good).
        self.AgentsC_OrderOutS = None # N_C*S: Orders sent for different products.
        self.AgentsC_orderOutS = None # N_C*S: Share of orders (in total order of the aggregate product) for different products.
        self.AgentsC_OrderOutSa = None # N_C*Sa: Orders sent for different aggregate products.
        # State variables in previous periods.
        self.AgentsC_PP_OrderOutP = None
        self.AgentsC_PP2_OrderOutP = None
        self.AgentsC_PP_orderOutP = None
        self.AgentsC_PP2_orderOutP = None
        
        # ======== State variables of transportation agents ========
        # Production-Agent-To-Production-Agent trasportation lines.
        self.AgentsT_P2P = None # Trasportation lines: nl_NetPP * AgentsT_P2P_MaxLength matrix. Each row is a transportation line, placed at the end of the row, with zeros at the beginning if this line's length is smaller than the max length of all transportation lines.
        self.AgentsT_P2P_Lengths = None # Trasportation line lengths: nl_NetPP*1 vector.
        self.AgentsT_P2P_MaxLength = None # Max length of all nl_NetPP transportation lines.
        self.AgentsT_P2P_StartLinInd = None # Linear indices of the starting point of each nl_NetPP lines in AgentsT_P2P.
        # Production-Agent-To-Consumption-Agent trasportation lines.
        self.AgentsT_P2C = None # Trasportation lines: nl_NetPC * AgentsT_P2C_MaxLength matrix. Each row is a transportation line, placed at the end of the row, with zeros at the beginning if this line's length is smaller than the max length of all transportation lines.
        self.AgentsT_P2C_Lengths = None # Trasportation line lengths: nl_NetPC*1 vector.
        self.AgentsT_P2C_MaxLength = None # Max length of all nl_NetPP transportation lines.
        self.AgentsT_P2C_StartLinInd = None # Linear indices of the starting point of each nl_NetPC lines in AgentsT_P2C.
        # Max length of all transportation agents.
        self.AgentsT_MaxLength = None
        
        # ======== Steady state variables of production agents ========
        self.AgentsP_SS_Xcap = None
        self.AgentsP_SS_a = None
        self.AgentsP_SS_OrderInP = None
        self.AgentsP_SS_OrderInC = None
        self.AgentsP_SS_OrderInE = None
        self.AgentsP_SS_orderlnP = None
        self.AgentsP_SS_orderInC = None
        self.AgentsP_SS_orderInE = None
        
        # ======== Steady state variables of consumption agents ========
        self.AgentsC_SS_OrderOutP = None
        self.AgentsC_SS_OrderOutS = None
        
        # ======== MRIO variables (optional, NEEDED for initializaton using MRIO data) ========
        self.MRIO_R = None # Number of regions.
        self.MRIO_S = None # Number of sectors in each region. Each sector produces one proudct.
        self.MRIO_Z = None # Intermediate flows. (RS)*(RS) matrix; Z(i,j) is the product flow from sector i to sector j.
        self.MRIO_C = None # Consumption. (RS)*(R) matrix; C(i,j) is the is the consumption of product i by region j.
        self.MRIO_VA = None # Value added. 1*(RS) vector; VA(i) is the value added of sector i.
        # The below variables are only defined for the open economy.
        self.MRIO_E = None  # Exports. (RS)*1 vector.
        self.MRIO_IP = None # Imports by sectors. 1*(RS) vector.
        self.MRIO_IC = None # Imports by consumption. 1*R vector.
        # Distance between regions.
        self.MRIO_Dist = None # MRIO_R*MRIO_R matrix: simulation steps for transportation from row region to column region.        
        
        # ======== MRIO variables (which could be initialized automatically)========
        # Resource constraints IN EACH SIMULATION PERIOD, may or maynot needed to be provided externally.
        self.WaterConstraints = None # MRIO_R*1 vector: Water constraints for each region in MRIO table.
        # Conversion matrices for MRIO computations
        self.Regions_Matrix = None # MRIO_R*N_P matrix that converts rows of region-sectors to rows of regions
        self.Regions_Matrix_WaterIntensity = None # MRIO_R*N_P matrix where each row contains the water intensities of the sectors in the corresponding region.

    def initialize_variables(self): # Initialize Basic variables of the world.

        self.N_P = self.MRIO_R * self.MRIO_S
        self.N_C = self.MRIO_R
        self.S = self.MRIO_S
        self.Products = np.kron(np.ones((self.MRIO_R, 1)), np.arange(self.MRIO_S)).astype(int).flatten()
        self.Products_Matrix = np.zeros((self.N_P, self.S), dtype=int)
        self.Products_Matrix[np.arange(self.N_P), self.Products] = 1

        # Determine Network Structure.
        # Network connecting producers.
        self.NetPP = self.MRIO_Z > self.CutOff_NetPP
        self.nl_NetPP = np.count_nonzero(self.NetPP)
        self.k_NetPP = np.flatnonzero(self.NetPP.ravel(order='F').copy())
        self.i_NetPP = np.arange(0, self.nl_NetPP)

        # MRIO variables.
        Col_NetPP = self.k_NetPP // self.N_P
        Row_NetPP = self.k_NetPP % self.N_P
        self.SendingRegion_NetPP = Row_NetPP // self.MRIO_S
        self.ReceivingRegion_NetPP = Col_NetPP // self.MRIO_S
        self.SendingSector_NetPP = Row_NetPP % self.MRIO_S
        self.ReceivingSector_NetPP = Col_NetPP % self.MRIO_S

        # Creating the table.
        self.AllLinks_NetPP = pd.DataFrame([
            {
                "i": int(i),
                "k": int(k),
                "SendingRegion": int(sr),
                "ReceivingRegion": int(rr),
                "SendingSector": int(ss),
                "ReceivingSector": int(rs)
            }
            for i, k, sr, rr, ss, rs in zip(
                self.i_NetPP, self.k_NetPP,
                self.SendingRegion_NetPP, self.ReceivingRegion_NetPP,
                self.SendingSector_NetPP, self.ReceivingSector_NetPP
            )
        ])
        
        
        # Network connecting producers and consumers.
        self.NetPC = self.MRIO_C > self.CutOff_NetPC
        self.nl_NetPC = np.count_nonzero(self.NetPC)
        self.k_NetPC = np.flatnonzero(self.NetPC.ravel(order='F').copy())
        self.i_NetPC = np.arange(0,self.nl_NetPC)
        # MRIO variables.
        Col_NetPC = self.k_NetPC // self.N_P
        Row_NetPC = self.k_NetPC % self.N_P
        self.SendingRegion_NetPC = Row_NetPC // self.MRIO_S
        self.ReceivingRegion_NetPC = Col_NetPC
        self.SendingSector_NetPC = Row_NetPC % self.MRIO_S
        
        # Creating the table.
        self.AllLinks_NetPC = pd.DataFrame([
            {
                "i": int(i),
                "k": int(k),
                "SendingRegion": int(sr),
                "ReceivingRegion": int(rr),
                "SendingSector": int(ss)
            }
            for i, k, sr, rr, ss in zip(
                self.i_NetPC, self.k_NetPC,
                self.SendingRegion_NetPC, self.ReceivingRegion_NetPC,
                self.SendingSector_NetPC
            )
        ])
        
        # Delete very small flows (i.e., below the cutoff value) in the networks.
        self.MRIO_Z[self.MRIO_Z <= self.CutOff_NetPP] = 0
        self.MRIO_C[self.MRIO_C <= self.CutOff_NetPC] = 0
        # MRIO_R*N_P matrix that converts rows of region-sectors to rows of regions.
        self.Regions_Matrix = np.kron(np.eye(self.MRIO_R), np.ones((1, self.MRIO_S)))
        # MRIO_R*N_P matrix where each row contains the water intensities of the sectors in the corresponding region.
        self.Regions_Matrix_WaterIntensity = self.Regions_Matrix * self.AgentsP_WaterIntensity.T
        # Number of aggregated products.
        self.Sa = self.S2Sa.shape[1]


    def initialize_production_agents(self):
        # State variables of production agents.
        self.AgentsP_Theta = np.zeros((self.N_P, 1)) # N_P*1: Reduction in production capacity relative to pre-event level, in [0,1].
        self.AgentsP_Alpha = np.ones((self.N_P, 1)) # N_P*1: Overproduction capacity, in [1, inf).
        self.AgentsP_Alpha_max = 1.2 * np.ones((self.N_P, 1)) # N_P*1: Maximun possible overproduction capacity, in [1, inf). 

        if not self.OpenEcon:
            self.MRIO_E = np.zeros((self.N_P, 1))
            self.MRIO_IP = np.zeros((1, self.N_P))
            self.MRIO_IC = np.zeros((1, self.N_C))
        
        # Total production, N_P*1.
        MRIO_X = np.sum(np.hstack([self.MRIO_Z, self.MRIO_C, self.MRIO_E]), axis=1).flatten()
        # Input requirement for unitary production, (RS)*(RS) matrix.
        MRIO_A = np.zeros((self.N_P, self.N_P))
        ind = MRIO_X > 0
        MRIO_A[:, ind] = self.MRIO_Z[:, ind] / MRIO_X[ind].T
        # Order inflows: Each row represents a order receiver, each column a order sender.
        self.AgentsP_OrderInP = self.MRIO_Z * self.delta_t # N_P*N_P: Order from production agents.
        self.AgentsP_OrderInC = self.MRIO_C * self.delta_t # N_P*N_C: Order from consumption agents.
        self.AgentsP_OrderInE = self.MRIO_E * self.delta_t # N_P*1: Order from export (only for open economy).
        self.AgentsP_OrderTot = np.sum(np.hstack([self.AgentsP_OrderInP, self.AgentsP_OrderInC, self.AgentsP_OrderInE]), axis=1) # N_P*1: Total order, sum of the above.
    
        ind = self.AgentsP_OrderTot > 0
        self.AgentsP_orderlnP = np.zeros((self.N_P, self.N_P))  # N_P*N_P: Share of order from other production agents in total order.
        self.AgentsP_orderlnP[ind, :] = self.AgentsP_OrderInP[ind, :] / self.AgentsP_OrderTot[ind, np.newaxis]
        self.AgentsP_orderInC = np.zeros((self.N_P, self.N_C))  # N_P*N_C: Share of order from consumption agents in total order.
        self.AgentsP_orderInC[ind, :] = self.AgentsP_OrderInC[ind, :] / self.AgentsP_OrderTot[ind, np.newaxis]
        self.AgentsP_orderInE = np.zeros((self.N_P, 1))  # N_P*1: Share of order from export in total order (only for open economy).
        self.AgentsP_orderInE[ind, :] = self.AgentsP_OrderInE[ind, :] / self.AgentsP_OrderTot[ind, np.newaxis]
        
        # Product inflows: Each row represents a product receiver, each column a product sender.
        self.AgentsP_ProductInP = self.AgentsP_OrderInP.T.copy()  # N_P*N_P: Product inflow through transportation agents (from production agents);
        self.AgentsP_ProductSentP = self.AgentsP_ProductInP.copy()  # N_P*N_P: Product sent (toward here) by production agents in the previous period.
        self.AgentsP_ProductSentS = self.AgentsP_ProductSentP @ self.Products_Matrix  # N_P*S: Different products sent toward producers at (t-1).
        temp = self.AgentsP_ProductSentS[:, self.Products]
        ind = temp > 0
        self.AgentsP_productSentP = np.zeros((self.N_P, self.N_P))  # N_P*N_P: Share of product sent (toward here) by production agents in a sector.
        self.AgentsP_productSentP[ind] = self.AgentsP_ProductSentP[ind] / temp[ind] 
        
        # Production technology: Each row represents a production agent, each column an type of input.
        self.AgentsP_a = MRIO_A.T @ self.Products_Matrix # N_P*S: Input (intermediate good) requirement for unitary production.
        self.AgentsP_a = (self.AgentsP_a @ self.S2Sa) # N_P*Sa: Input (intermediate aggregate good) requirement for unitary production.
        
        
        self.AgentsP_va = np.zeros((self.N_P, 1))  # N_P*1: Value-added requirement for unitary production.
        ind = MRIO_X > 0
        self.AgentsP_va[ind] = self.MRIO_VA.T[ind] / MRIO_X.reshape(-1,1)[ind]
        self.AgentsP_im = np.zeros((self.N_P, ))  # N_P*1: Import requirement for unitary production (only for open economy).
        self.AgentsP_im[ind] = self.MRIO_IP[:, ind].T.flatten() / MRIO_X[ind]

        # Production: Each row represents a production agent.
        self.AgentsP_Xcap = MRIO_X * self.delta_t  # N_P*1: Production capacity.
        self.AgentsP_Xs = None  # N_P*S: Possible production levels constrained by inventories of different products. Did not initialize this variable, because it will be computed in the production stage.
        self.AgentsP_Xa = self.AgentsP_Xcap  # N_P*1: Actual production.
        self.AgentsP_VA = self.MRIO_VA.T * self.delta_t  # N_P*1: Actual value added.
        # Inventories.
        self.AgentsP_n = self.ndays_Target_Default * np.ones((self.N_P, self.Sa))  # N_P*Sa: Targeted days of use for different aggregate products.
        # ...We know that each row represents a production agent, and each column represents a type of aggregate product.
        # obj.AgentsP_n(:,13) = 2; % The inventory for electricity lasts for only 1 period.
        # ...In China's MRIO table of EORA26 database, the electricity，gas,water sector is 25.
        self.AgentsP_I = self.AgentsP_n * (self.AgentsP_Xa[:, np.newaxis] * self.AgentsP_a)  # N_P*Sa: Current inventory level.

        # Product Outflows: Each row represents an product sender, each column a product receiver.
        self.AgentsP_ProductOutP = self.AgentsP_OrderInP.copy()  # N_P*N_P: Product sent toward different production agents (through transportation agents).
        self.AgentsP_ProductOutC = self.AgentsP_OrderInC.copy()  # N_P*N_C: Product sent toward different consumption agents (through transportation agents).
        self.AgentsP_ProductOutE = self.AgentsP_OrderInE.copy()  # N_P*1: Product sent toward export (only for open economy).
        # Order Outflows: Each row represents an order sender, each column an order receiver.
        self.AgentsP_OrderOutP = self.AgentsP_OrderInP.T  # N_P*N_P: Orders sent toward different production agents.
        self.AgentsP_OrderOutS = self.AgentsP_OrderOutP @ self.Products_Matrix  # N_P*S: Orders sent for different products.
        temp = self.AgentsP_OrderOutS[:, self.Products]
        ind = temp > 0
        self.AgentsP_orderOutP = np.zeros((self.N_P, self.N_P))  # N_P*N_P: Share of orders (in total order of the product) sent toward different production agents.
        self.AgentsP_orderOutP[ind] = self.AgentsP_OrderOutP[ind] / temp[ind]

        # State variables in previous periods.
        self.AgentsP_PP_a = self.AgentsP_a.copy()
        self.AgentsP_PP_Xa = self.AgentsP_Xa.copy()
        self.AgentsP_PP_OrderOutP = self.AgentsP_OrderOutP.copy()
        self.AgentsP_PP2_OrderOutP = self.AgentsP_OrderOutP.copy()
        self.AgentsP_PP_orderOutP = self.AgentsP_orderOutP.copy()
        self.AgentsP_PP2_orderOutP = self.AgentsP_orderOutP.copy()
        self.AgentsP_PP_OrderOutS = self.AgentsP_OrderOutS.copy()
        self.AgentsP_PP2_OrderOutS = self.AgentsP_OrderOutS.copy()

        # Steady state variables of production agents.
        self.AgentsP_SS_Xcap = self.AgentsP_Xcap.copy()
        self.AgentsP_SS_a = self.AgentsP_a.copy()
        self.AgentsP_SS_OrderInP = self.AgentsP_OrderInP.copy()
        self.AgentsP_SS_OrderInC = self.AgentsP_OrderInC.copy()
        self.AgentsP_SS_OrderInE = self.AgentsP_OrderInE.copy()
        self.AgentsP_SS_orderlnP = self.AgentsP_orderlnP.copy()
        self.AgentsP_SS_orderInC = self.AgentsP_orderInC.copy()
        self.AgentsP_SS_orderInE = self.AgentsP_orderInE.copy()

        # Variables related to the aggregate products.
        self.AgentsP_ProductSentSa = self.AgentsP_ProductSentS @ self.S2Sa  # N_P*Sa: Different aggregate products sent toward producers at (t-1).
        self.AgentsP_OrderOutSa = self.AgentsP_OrderOutS @ self.S2Sa  # N_P*Sa: Orders sent for different aggregate products.
        self.AgentsP_orderOutS = np.zeros_like(self.AgentsP_OrderOutS)  # N_P*S: Share of orders (in total order of the aggregate product) for different products.
        temp = self.AgentsP_OrderOutSa @ self.S2Sa.T
        ind = temp > 0
        self.AgentsP_orderOutS[ind] = self.AgentsP_OrderOutS[ind] / temp[ind]
        # Variables in previous periods.
        self.AgentsP_PP_OrderOutSa = self.AgentsP_OrderOutSa.copy()
        self.AgentsP_PP2_OrderOutSa = self.AgentsP_OrderOutSa.copy()

        # Initiate resource constraints. Higher than normal use.
        self.WaterConstraints = self.Regions_Matrix_WaterIntensity @ self.AgentsP_SS_Xcap + np.finfo(float).eps


    def initialize_consumption_agents(self): # Initialize consumption agents.
        # Product inflows: Each row represents a consumer agent.
        self.AgentsC_ProductInP = self.MRIO_C.T * self.delta_t  # N_C*N_P: Product inflow through transportation agents (from production agents).
        self.AgentsC_ProductSentP = self.AgentsC_ProductInP  # N_C*N_P: Product sent (toward here) by production agents in.
        self.AgentsC_ProductSentS = self.AgentsC_ProductSentP @ self.Products_Matrix  # N_C*S: Different products sent toward consumers at (t-1).
    
        # Shares in product inflows.
        self.AgentsC_productSentP = np.zeros((self.N_C, self.N_P))  # N_C*N_P: Share of product sent (toward here) by production agents in the previous period.
        temp = self.AgentsC_ProductSentS[:, self.Products]  # 修正 MATLAB 到 Python 的索引
        ind = temp > 0
        self.AgentsC_productSentP[ind] = self.AgentsC_ProductSentP[ind] / temp[ind]
    
        # Import requirement (i.e., order) for consumption, only for open economy.
        self.AgentsC_IMO = self.MRIO_IC.T * self.delta_t  # N_C*1.
    
        # Order Outflows.
        self.AgentsC_OrderOutP = self.AgentsC_ProductInP  # N_C*N_P: Orders sent toward different production agents.
        self.AgentsC_orderOutP = self.AgentsC_productSentP  # N_C*N_P: Share of orders (for a particular good) sent toward different production agents (that produces this good).
        self.AgentsC_OrderOutS = self.AgentsC_ProductSentS  # N_C*S: Orders sent for different products.
    
        # State variables in previous periods.
        self.AgentsC_PP_OrderOutP = self.AgentsC_OrderOutP.copy()
        self.AgentsC_PP2_OrderOutP = self.AgentsC_OrderOutP.copy()
        self.AgentsC_PP_orderOutP = self.AgentsC_orderOutP.copy()
        self.AgentsC_PP2_orderOutP = self.AgentsC_orderOutP.copy()
    
        # Steady state variables of consumption agents.
        self.AgentsC_SS_OrderOutP = self.AgentsC_OrderOutP.copy()
        self.AgentsC_SS_OrderOutS = self.AgentsC_OrderOutS.copy()

    def initialize_transportation_agents(self): # Initialize Production-Agent-To-Production-Agent transportation line lengths.
        x, y = np.unravel_index(self.k_NetPP, (self.N_P, self.N_P),order='F')
        x_R = np.ceil((x + 1) / self.MRIO_S).astype(int)
        y_R = np.ceil((y + 1) / self.MRIO_S).astype(int)
        self.AgentsT_P2P_Lengths = self.MRIO_Dist[x_R - 1, y_R - 1]
        self.AgentsT_P2P_MaxLength = np.max(self.AgentsT_P2P_Lengths)

        # Initialize Production-Agent-To-Consumption-Agent transportation line lengths.
        x, y = np.unravel_index(self.k_NetPC, (self.N_P, self.N_C),order='F')
        x_R = np.ceil((x + 1) / self.MRIO_S).astype(int)
        y_R = y + 1
        self.AgentsT_P2C_Lengths = self.MRIO_Dist[x_R - 1, y_R - 1]
        self.AgentsT_P2C_MaxLength = np.max(self.AgentsT_P2C_Lengths)

        # Max transportation length
        self.AgentsT_MaxLength = max(self.AgentsT_P2P_MaxLength, self.AgentsT_P2C_MaxLength)
        
        # Initialize P2P transportation lines
        self.AgentsT_P2P = np.zeros((self.nl_NetPP, self.AgentsT_MaxLength))
        temp = self.AgentsP_OrderInP.ravel(order='F').copy()
        for i in range(self.nl_NetPP):
            start = self.AgentsT_MaxLength - self.AgentsT_P2P_Lengths[i]
            end = self.AgentsT_MaxLength
            self.AgentsT_P2P[i, start:end] = temp[self.k_NetPP[i]]
        
        rows = np.arange(self.nl_NetPP)
        #cols = (self.AgentsT_MaxLength - self.AgentsT_P2P_Lengths).clip(0, self.AgentsT_P2P.shape[1])
        cols = (self.AgentsT_MaxLength - self.AgentsT_P2P_Lengths)
        shape = (self.AgentsT_P2P.shape[0], self.AgentsT_P2P.shape[1] + 1)
        self.AgentsT_P2P_StartLinInd = np.ravel_multi_index((rows, cols), shape, order='F')
        
        # Initialize P2C transportation lines
        self.AgentsT_P2C = np.zeros((self.nl_NetPC, self.AgentsT_MaxLength))
        temp = self.AgentsC_ProductInP.T.ravel(order='F').copy()
        for i in range(self.nl_NetPC):
            start = self.AgentsT_MaxLength - self.AgentsT_P2C_Lengths[i]
            end = self.AgentsT_MaxLength
            self.AgentsT_P2C[i, start:end] = temp[self.k_NetPC[i]]

        rows = np.arange(self.nl_NetPC)
        #cols = (self.AgentsT_MaxLength - self.AgentsT_P2C_Lengths).clip(0, self.AgentsT_P2C.shape[1])
        cols = (self.AgentsT_MaxLength - self.AgentsT_P2C_Lengths)
        shape = (self.AgentsT_P2C.shape[0], self.AgentsT_P2C.shape[1] + 1)
        self.AgentsT_P2C_StartLinInd = np.ravel_multi_index((rows, cols), shape, order='F')

    def agents_communicate(self):
        # Agents communicate.

        # Production agents communicate.
        # Order information flows.
        self.AgentsP_OrderInP = self.AgentsP_OrderOutP.T.copy()
        # Sent-products information flows.
        self.AgentsP_ProductSentP = self.AgentsP_ProductOutP.T.copy()

        # Production and consumption agents communicate.
        # Order information flows.
        self.AgentsP_OrderInC = self.AgentsC_OrderOutP.T.copy()
        # Sent-products information flows.
        self.AgentsC_ProductSentP = self.AgentsP_ProductOutC.T.copy()

    def update_inventories(self):
        # Update inventories of production agents.
        self.AgentsP_I = self.AgentsP_I - self.AgentsP_PP_Xa[:, np.newaxis] * self.AgentsP_PP_a + self.AgentsP_ProductInP @ self.Products_Matrix @ self.S2Sa
        
    def update_export_orders(self):
        # Update export orders (only for open economy).
        if self.OpenEcon:
            ind1 = self.AgentsP_OrderInE > self.AgentsP_ProductOutE
            ind2 = (self.AgentsP_OrderInE <= self.AgentsP_ProductOutE) & (self.AgentsP_SS_OrderInE > 0)

            self.AgentsP_OrderInE[ind1] = self.AgentsP_OrderInE[ind1] - \
                (self.AgentsP_OrderInE[ind1] - self.AgentsP_ProductOutE[ind1]) / self.AgentsP_OrderInE[ind1] * \
                self.AgentsP_OrderInE[ind1] * (self.delta_t / self.tau_E)

            self.AgentsP_OrderInE[ind2] = self.AgentsP_OrderInE[ind2] + \
                (self.AgentsP_SS_OrderInE[ind2] - self.AgentsP_OrderInE[ind2]) / self.AgentsP_SS_OrderInE[ind2] * \
                (self.AgentsP_SS_OrderInE[ind2] - self.AgentsP_OrderInE[ind2]) * (self.delta_t / self.tau_E)

    def update_shares(self):
        # Update shares.

        # Update shares of the consumption agents.
        # Different products sent toward consumers at (t-1).
        # N_C*S. Each row is a consumer; each column a product.
        self.AgentsC_ProductSentS = self.AgentsC_ProductSentP @ self.Products_Matrix
        temp = self.AgentsC_ProductSentS[:, self.Products]
        ind = temp > 0
        self.AgentsC_productSentP[ind] = self.AgentsC_ProductSentP[ind] / temp[ind]

        # Update shares of the production agents.
        self.AgentsP_OrderTot = np.sum(
            np.column_stack([
                self.AgentsP_OrderInP,
                self.AgentsP_OrderInC,
                self.AgentsP_OrderInE
            ]),
            axis=1
        )
        ind = self.AgentsP_OrderTot > 0
        self.AgentsP_orderlnP[ind, :] = self.AgentsP_OrderInP[ind, :] / self.AgentsP_OrderTot[ind, np.newaxis]
        self.AgentsP_orderInC[ind, :] = self.AgentsP_OrderInC[ind, :] / self.AgentsP_OrderTot[ind, np.newaxis]
        self.AgentsP_orderInE[ind] = self.AgentsP_OrderInE[ind] / self.AgentsP_OrderTot[ind][:, np.newaxis]

        # Different products sent toward producers at (t-1).
        # N_P*S. Each row is a producer; each column a product.
        self.AgentsP_ProductSentS = self.AgentsP_ProductSentP @ self.Products_Matrix
        temp = self.AgentsP_ProductSentS[:, self.Products]
        ind = temp > 0
        self.AgentsP_productSentP[ind] = self.AgentsP_ProductSentP[ind] / temp[ind]

        # N_P*Sa: Different aggregate products sent toward producers at (t-1).
        self.AgentsP_ProductSentSa = self.AgentsP_ProductSentS @ self.S2Sa
        
        
    def production_agents_produce(self): # Production agents produce.
        from scipy.optimize import linprog    
        # Update production capacity.
        self.AgentsP_Xcap = (self.AgentsP_Alpha.flatten() * (1 - self.AgentsP_Theta.flatten()) * self.AgentsP_SS_Xcap)

        # Update possible production levels constrained by inventories of different products.
        self.AgentsP_Xs = (self.AgentsP_I + (self.AgentsP_a == 0)) / self.AgentsP_a

        # Calculate the upper and lower bounds of production.
        ub = np.min(np.column_stack([self.AgentsP_Xcap, self.AgentsP_OrderTot, self.AgentsP_Xs]), axis=1)
        ub[ub < 0] = 0
        lb = np.zeros(self.N_P)
        lb[self.AgentsP_SS_Xcap > 0] = self.AgentsP_SS_Xcap[self.AgentsP_SS_Xcap > 0] * 1e-2

        # Update actual production levels using linear optimization.
        self.AgentsP_Xa = linprog(
            c=-np.ones(self.N_P),
            A_ub=self.Regions_Matrix_WaterIntensity,
            b_ub=self.WaterConstraints,
            bounds=np.column_stack((lb, ub)),
            method='highs'
        ).x

        # Update value added.
        self.AgentsP_VA = self.AgentsP_Xa.reshape(-1,1) * self.AgentsP_va

        # Calculate scarcity Indices for each product in each region.
        AgentsP_ProductInSa = self.AgentsP_ProductInP @ self.Products_Matrix @ self.S2Sa
        AgentsC_ProductInSa = self.AgentsC_ProductInP @ self.Products_Matrix @ self.S2Sa
        self.RegionsProducts_Supplied = self.Regions_Matrix @ AgentsP_ProductInSa + AgentsC_ProductInSa

        # Products demanded.
        AgentsP_Xa_1 = np.min(np.column_stack([self.AgentsP_Xcap, self.AgentsP_OrderTot]), axis=1)
        AgentsP_ProductDemandedSa = AgentsP_Xa_1[:, np.newaxis] * self.AgentsP_a
        self.RegionsProducts_Demanded = self.Regions_Matrix @ AgentsP_ProductDemandedSa + self.AgentsC_OrderOutS @ self.S2Sa

        # Calculate scarcity of region sectors (with respect to consumption).
        self.Scarcity_RegionsProducts = np.zeros((self.N_P // self.S, self.Sa))
        idx = self.RegionsProducts_Demanded > 0
        self.Scarcity_RegionsProducts[idx] = (self.RegionsProducts_Demanded[idx] - self.RegionsProducts_Supplied[idx]) / self.RegionsProducts_Demanded[idx]



    def production_agents_prepare_product_out(self): # Production agents prepare product outflows.
        # If no scarcity.    
        ind = self.AgentsP_Xa >= self.AgentsP_OrderTot
        self.AgentsP_ProductOutP[ind, :] = self.AgentsP_OrderInP[ind, :]
        self.AgentsP_ProductOutC[ind, :] = self.AgentsP_OrderInC[ind, :]
        self.AgentsP_ProductOutE[ind, :] = self.AgentsP_OrderInE[ind, :]
        # Under scarcity.
        ind = self.AgentsP_Xa < self.AgentsP_OrderTot
        self.AgentsP_ProductOutP[ind, :] = self.AgentsP_Xa[ind, np.newaxis] * self.AgentsP_SS_orderlnP[ind, :]
        self.AgentsP_ProductOutC[ind, :] = self.AgentsP_Xa[ind, np.newaxis] * self.AgentsP_SS_orderInC[ind, :]
        self.AgentsP_ProductOutE[ind, :] = self.AgentsP_Xa[ind, np.newaxis] * self.AgentsP_SS_orderInE[ind, :]

    def production_agents_prepare_order_out(self): # Production agents prepare order outflows.
        # AgentsP_IT is targeted invetory levels: N_P*Sa.
        # Each row represents a production agent and column a type of aggregated product.    
        temp = np.minimum(self.AgentsP_OrderTot, self.AgentsP_Xcap.flatten()).reshape(-1,1)
        AgentsP_IT = self.AgentsP_n * (temp * self.AgentsP_a)
        
        # AgentsP_OrderOutSa is orders for different aggregate products: N_P*Sa.
        # AgentsP_OrderOutS is orders for different products: N_P*S.
        # Each row represents a production agent and each column a type of product.
        self.AgentsP_OrderOutSa = self.AgentsP_Xa[:, np.newaxis] * self.AgentsP_a + \
            (AgentsP_IT - self.AgentsP_I) * (self.delta_t / self.tau_I)
        self.AgentsP_OrderOutSa[self.AgentsP_OrderOutSa < 0] = 0

        self.AgentsP_OrderOutS = self.AgentsP_OrderOutSa @ self.S2Sa.T * self.AgentsP_orderOutS
        self.AgentsP_OrderOutS[self.AgentsP_OrderOutS < 0] = 0
        # Share of orders sent toward different production agents.
        self.AgentsP_orderOutP += (self.AgentsP_productSentP - self.AgentsP_PP2_orderOutP) * (self.delta_t / self.tau_O)
        # Orders sent toward different production agents.
        #self.AgentsP_OrderOutP = self.AgentsP_OrderOutS[:, self.Products] * self.AgentsP_orderOutP
        self.AgentsP_OrderOutP = self.AgentsP_OrderOutS[:, self.Products] * self.AgentsP_orderOutP

    def production_agents_adapt_to_shocks(self): # Production agents adpat to shocks.
        # Reconstruct.
        self.AgentsP_Theta *= (1 - self.delta_t / self.tau_Theta)
        # Adjust overproduction capacity.
        # If production capacity does not meet demand.
        ind = self.AgentsP_Xcap < self.AgentsP_OrderTot
        self.AgentsP_Alpha[ind] += (
            self.AgentsP_Alpha_max[ind] - self.AgentsP_Alpha[ind]
        ) * ((self.AgentsP_OrderTot[ind] - self.AgentsP_Xcap[ind]) / self.AgentsP_OrderTot[ind]).reshape(-1,1) * (self.delta_t / self.tau_Alpha)
        # If production capacity can satisfy demand.
        ind = self.AgentsP_Xcap >= self.AgentsP_OrderTot
        self.AgentsP_Alpha[ind] -= (
            self.AgentsP_Alpha[ind] - 1
        ) * (self.delta_t / self.tau_Alpha)

    def production_agents_adapt_to_shortages(self): # Production agents adapt production modes to shortages.
        # Remember the unadapted production technology.
        self.AgentsP_PP_a = self.AgentsP_a.copy()
        
        # SCARCITY of different products perceived by producers.
        # N_P*Sa. Each row is a producer; each column an aggregate product.
        ScarcityIndex = np.zeros((self.N_P, self.Sa))
        # If supply cannot meet demand.
        ind = self.AgentsP_PP2_OrderOutSa > (self.AgentsP_ProductSentSa + 1e-8)
        ScarcityIndex[ind] = (
            self.AgentsP_PP2_OrderOutSa[ind] - self.AgentsP_ProductSentSa[ind]
        ) / self.AgentsP_PP2_OrderOutSa[ind]

        if self.OpenEcon:
            temp = np.zeros((self.N_P, self.Sa))
            temp[ind] = ScarcityIndex[ind] * self.AgentsP_a[ind] * (self.delta_t / self.tau_A)
            self.AgentsP_im += np.sum(temp, axis=1)

        self.AgentsP_a[ind] -= ScarcityIndex[ind] * self.AgentsP_a[ind] * (self.delta_t / self.tau_A)
        # If supply can meet demand.
        ind = (self.AgentsP_PP2_OrderOutSa <= (self.AgentsP_ProductSentSa + 1e-8)) & (self.AgentsP_SS_a > 0)
        if self.OpenEcon:
            temp = np.zeros((self.N_P, self.Sa))
            temp[ind] = ((
                (self.AgentsP_SS_a[ind] - self.AgentsP_a[ind]) / self.AgentsP_SS_a[ind]
            ) * (self.AgentsP_SS_a[ind] - self.AgentsP_a[ind]) * (self.delta_t / self.tau_A))
            self.AgentsP_im -= np.sum(temp, axis=1)
        # Update intermediate requirement for unitary production.
        self.AgentsP_a[ind] += (
            (self.AgentsP_SS_a[ind] - self.AgentsP_a[ind]) / self.AgentsP_SS_a[ind]
        ) * (self.AgentsP_SS_a[ind] - self.AgentsP_a[ind]) * (self.delta_t / self.tau_A)

    def production_agents_remember(self): # Production agents remember key state variables.
        self.AgentsP_PP_Xa = self.AgentsP_Xa.copy()
        self.AgentsP_PP2_OrderOutP = self.AgentsP_PP_OrderOutP.copy()
        self.AgentsP_PP_OrderOutP = self.AgentsP_OrderOutP.copy()
        self.AgentsP_PP2_orderOutP = self.AgentsP_PP_orderOutP.copy()
        self.AgentsP_PP_orderOutP = self.AgentsP_orderOutP.copy()
        self.AgentsP_PP2_OrderOutS = self.AgentsP_PP_OrderOutS.copy()
        self.AgentsP_PP_OrderOutS = self.AgentsP_OrderOutS.copy()
        self.AgentsP_PP2_OrderOutSa = self.AgentsP_PP_OrderOutSa.copy()
        self.AgentsP_PP_OrderOutSa = self.AgentsP_OrderOutSa.copy()

    def consumption_agents_consume(self): # Consumption agents consume.
        pass

    def consumption_agents_prepare_order_out(self): # Consumption agents prepare order outflows.
        # Adjust domestic order shares.    
        self.AgentsC_orderOutP = self.AgentsC_orderOutP + (self.AgentsC_productSentP - self.AgentsC_PP2_orderOutP) * (self.delta_t / self.tau_O)
        # Orders sent toward different production agents.
        self.AgentsC_OrderOutP = self.AgentsC_SS_OrderOutS[:, self.Products] * self.AgentsC_orderOutP
        self.AgentsC_OrderOutS = self.AgentsC_OrderOutP @ self.Products_Matrix

    def consumption_agents_remember(self): # Consumption agents remember key state variables
        self.AgentsC_PP2_OrderOutP = self.AgentsC_PP_OrderOutP.copy()
        self.AgentsC_PP_OrderOutP = self.AgentsC_OrderOutP.copy()
        self.AgentsC_PP2_orderOutP = self.AgentsC_PP_orderOutP.copy()
        self.AgentsC_PP_orderOutP = self.AgentsC_orderOutP.copy()
        
    def transport_obstruct_mrio(self, r1, r2, s, position_fraction, product_blocked): # Model obstruction in a trasportation lines.
        # r1: origin region.
        # r2: destination region.
        # s: product (same as sector).
        # position_fraction: in (0, 1], the position in the tranporaton line where the obstruction occurs.
        # product_blocked: the amount of product blocked in the above position.
 
        # Find corresponding rows (i.e., link indices) in tables AllLinks_NetPP and AllLinks_NetPC.    
        idx1 = self.AllLinks_NetPP[(self.AllLinks_NetPP.SendingRegion == r1) &
                                   (self.AllLinks_NetPP.ReceivingRegion == r2) &
                                   (self.AllLinks_NetPP.SendingSector == s)].i.values
        idx2 = self.AllLinks_NetPC[(self.AllLinks_NetPC.SendingRegion == r1) &
                                   (self.AllLinks_NetPC.ReceivingRegion == r2) &
                                   (self.AllLinks_NetPC.SendingSector == s)].i.values

        SupplyChains = np.vstack([self.AgentsT_P2P[idx1, :], self.AgentsT_P2C[idx2, :]])
        SupplyChains_Length = np.unique(np.concatenate([self.AgentsT_P2P_Lengths[idx1], self.AgentsT_P2C_Lengths[idx2]]))
        SupplyChains_Position = self.AgentsT_MaxLength - SupplyChains_Length + np.ceil(SupplyChains_Length * position_fraction).astype(int)-1
        
        # Total product in the blocked position.
        total = np.sum(SupplyChains[:, SupplyChains_Position+1])
        product_blocked_true = min(total, product_blocked)
        # Blockage (i.e., obstruction) occurs!
        if product_blocked_true > 0:
            ratio_blocked = product_blocked_true / total
            SupplyChains[:, SupplyChains_Position] += ratio_blocked * SupplyChains[:, SupplyChains_Position+1]
            SupplyChains[:, SupplyChains_Position + 1] *= (1 - ratio_blocked)

        # Assign the total (length(idx1) + length(idx2)) lines back to the (augumented) AgentsT_P2P and AgentsT_P2C.
        self.AgentsT_P2P[idx1, :] = SupplyChains[:len(idx1), :]
        self.AgentsT_P2C[idx2, :] = SupplyChains[len(idx1):, :]
        
        
    