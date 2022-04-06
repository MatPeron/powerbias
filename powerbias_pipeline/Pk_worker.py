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

L, Ng, kf = parameters["L"], parameters["Ng"], parameters["kf"]
knorms, deltak1, deltak2 = parameters["knorms"], parameters["deltak1"], parameters["deltak2"] 

nbins = np.zeros(1, dtype=int)
comm.Recv(nbins, source=0, tag=73)

kbins = np.zeros(nbins[0])
comm.Recv(kbins, source=0, tag=99)

functions.printflush("starting P(k) computation on {} bins".format(nbins[0]), rank)

local_ks, local_Pks, local_sigmaPks = [], [], []
for k in kbins:
    singlek, singlePk, singlesigmaPk = functions.computeSinglePk(k, knorms, deltak1, deltak2, L, Ng, kf)

    local_ks.append(singlek)
    local_Pks.append(singlePk)
    local_sigmaPks.append(singlesigmaPk)

local_ks = np.array(local_ks)
local_Pks = np.array(local_Pks)
local_sigmaPks = np.array(local_sigmaPks)

functions.printflush("has completed P(k) computation, sending results to root", rank)

comm.Send(local_ks, dest=0, tag=89)
comm.Send(local_Pks, dest=0, tag=12)
comm.Send(local_sigmaPks, dest=0, tag=90)

comm.Disconnect()
