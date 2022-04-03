# example of function that reads simulated data from an hdf5 file
import h5py
import numpy as np

def readSim(file):
    
    with h5py.File(file, "r") as f:
        IDsGalaxies = list(f.keys())
        Np = len(f.keys())
        positions = []
        for i in range(Np):
            files = IDsGalaxies[i]
            tmpGWpos = np.asarray(f[files]["Pos_fin"])
            positions.extend(list(tmpGWpos))
        positions = np.array(positions)
    return np.array(positions)