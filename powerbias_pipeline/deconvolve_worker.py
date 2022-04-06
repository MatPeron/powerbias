import subprocess as sp
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

Ng, H, kf, kNyq = parameters["Ng"], parameters["H"], parameters["kf"], parameters["kNyq"]

n = Ng**3//size
r = Ng**3%size

if rank<r:
    start = rank*(n+1)
    stop = start+n+1

else:
    start = rank*n+r
    stop = start+n

functions.printflush("starting knorm and Wk computation".format(rank), rank)

local_knorms = np.zeros((Ng, Ng, Ng))
local_Wk = np.zeros((Ng, Ng, Ng))
for m in range(start, stop):
    index = (m//(Ng**2), m%(Ng**2)//Ng, m%Ng)
    local_knorms[index], local_Wk[index] = functions.computeDeconvolve(index, H, kf, kNyq)

functions.printflush("has completed knorm and Wk computation, sending results to root".format(rank), rank)

comm.Send(local_knorms, dest=0, tag=63)
comm.Send(local_Wk, dest=0, tag=24)

comm.Disconnect()
