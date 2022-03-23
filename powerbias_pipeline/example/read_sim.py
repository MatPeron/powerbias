# example of function that reads simulated data from an hdf5 file

import h5py
import numpy as np

def readSim(file):

   with h5py.File(file, "r") as f:
       positions = np.asarray(f["Coordinates"])

   return positions
