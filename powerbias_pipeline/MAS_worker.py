import numpy as np
from mpi4py import MPI
import pickle
import functions

# initialize MPI and get number of processors and processor rank
comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

parameters = None
parameters = comm.bcast(parameters, root=0)

L, Ng, Np, H = parameters["L"], parameters["Ng"], parameters["Np"], parameters["H"]

n = Np//size
r = Np%size

if rank<r:
    start = rank*(n+1)
    stop = start+n+1

else:
    start = rank*n+r
    stop = start+n

# read positions from catalog within the index range of the processor
local_pos = np.empty((stop-start, 3))
comm.Recv(local_pos, source=0, tag=42)

# initialize local variable
local_rho = np.zeros((Ng, Ng, Ng))

functions.printflush("starting MAS computation on {} particles (indexes {} to {})".format(len(local_pos), start, stop), rank)
local_pos, counts = np.unique(local_pos, return_counts=True, axis=0)

for pos, count in zip(local_pos, counts):
    weights, indexes = functions.assignDoubleGrid(pos, L, Ng)
    for i in range(len(weights)):
        local_rho[indexes[i]] += count*weights[i]/H**3

functions.printflush("has completed MAS computation, sending MAS results to root".format(rank), rank)

# send local result to 0 for summing
comm.Send(local_rho, dest=0, tag=35)

comm.Disconnect()
