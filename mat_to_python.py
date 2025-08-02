import json

# ==== convert mat to json ====
from scipy.io import loadmat

mat = loadmat("MRIOExample.mat")

converted = {
    "MRIO_Z": mat["Z_MRIOExample"].tolist(),
    "MRIO_C": mat["C_MRIOExample"].tolist(),
    "MRIO_R": int(mat["R_MRIOExample"].item()),
    "MRIO_S": int(mat["S_MRIOExample"].item()),
    "MRIO_VA": mat["VA_MRIOExample"].flatten().tolist()
}

with open("MRIOExample.json", "w") as f:
    json.dump(converted, f, indent=2)
    
    
# ==== convert mat to json ====
from scipy.io import loadmat

mat = loadmat("CityLevelMRIO2017-python.mat")

converted = {
    "MRIO_Z": mat["MRIO_Z"].tolist(),
    "MRIO_C": mat["MRIO_C"].tolist(),
    "MRIO_R": int(mat["MRIO_R"].item()),
    "MRIO_S": int(mat["MRIO_S"].item()),
    "MRIO_VA": mat["MRIO_VA"].flatten().tolist()
}

with open("CityLevelMRIO2017-python.json", "w") as f:
    json.dump(converted, f, indent=2)
    