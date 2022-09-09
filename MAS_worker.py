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

L, Ng, Np, H, isIntrl, MASOrder, verbose = (parameters[k] for k in ["L",
                                                                    "Ng",
                                                                    "Np",
                                                                    "H",
                                                                    "isIntrl",
                                                                    "MASOrder",
                                                                    "verbose"])

if verbose:
    from functions import printflush as pflush
    
else:
    from functions import fakePrintflush as pflush

n = Np//size
r = Np%size

if rank<r:
    start = rank*(n+1)
    stop = start+n+1

else:
    start = rank*n+r
    stop = start+n

# read positions from catalog within the index range of the processor
localPos = np.empty((stop-start, 3))
comm.Recv(localPos, source=0, tag=42)

# initialize local variable
localN = np.zeros((Ng, Ng, Ng))

pflush("starting MAS computation on {} particles (indexes {} to {})".format(len(localPos), start, stop), rank)
localPos, counts = np.unique(localPos, return_counts=True, axis=0)

for pos, count in zip(localPos, counts):
    weights, indexes = functions.assignDoubleGrid(pos, L, Ng, isIntrl, MASOrder)
    for i in range(len(weights)):
        localN[indexes[i]] += count*weights[i]

pflush("has completed MAS computation, sending MAS results to root".format(rank), rank)

# send local result to 0 for summing
comm.Send(localN, dest=0, tag=35)

comm.Disconnect()
