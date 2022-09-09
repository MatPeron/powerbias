import numpy as np
import sys
import time

def printflush(string, rank="ROOT"):

    caller = sys._getframe().f_back.f_code.co_filename
    if "/" in caller:
        caller = caller.split("/")[-1]
        
    t = time.time()
    print("[{:.0f}] cpu{}//{}: {}".format(t, rank, caller, string))
    sys.stdout.flush()
    
def fakePrintflush(string, rank="ROOT"):
    
    pass

def assignDoubleGrid(r, L, Ng, isIntrl, MASOrder):

    if isIntrl:
        NWeight, idx = assignDoubleGrid(r, L, Ng, False, MASOrder)
        
    else:
        NWeight, idx = [], []

    # particle position is adjusted for periodicity and normalized
    r = (r+L)%L
    r = r/L if np.all(r/L<=1) else np.ones(3)

    # position of the particle scaled to the number of points in the grid
    shift = 0.5 if isIntrl else 0
    vr = Ng*r+shift

    if MASOrder==2:
        iv = assignIndex(vr.astype(int), Ng, MASOrder)
        w1 = assignWeights2(vr)
        
    for x in range(MASOrder):
        for y in range(MASOrder):
            for z in range(MASOrder):
                NWeight.append(w1[0, x]*w1[1, y]*w1[2, z])
                idx.append((iv[0, x], iv[1, y], iv[2, z]))

    return NWeight, idx

# For CIC it finds the indices of the 2 cells that delimit the cube of 8 cells
# along the 3 directions
def assignIndex(v, Ng, MASOrder):

    idx = np.zeros((3, 4), dtype=int)

    for j in range(MASOrder):
        idx[:, j] = (v+j+Ng)%Ng

    return idx

# Computes the weights
def assignWeights2(v):

    w = np.zeros((3, 4))

    for i in range(3):
        h = v[i]-int(v[i])
        w[i, 0] = 1-h
        w[i, 1] = h

    return w
    
def logNormLikelihood(pars, y, yerr, bOps, isCross):
    
    b1, b2, b3, alpha = pars
    
    if not isCross:
        model = b1**2*bOps["b1b1"] +\
                b1*b2*bOps["b1b2"] +\
                b1*(-2/7)*(b1-1)*bOps["b1bs"] +\
                b2**2*bOps["b2b2"] +\
                b2*(-2/7)*(b1-1)*bOps["b2bs"] +\
                ((-2/7)*(b1-1))**2*bOps["bsbs"] +\
                b1*b3*bOps["b1b3"] +\
                alpha*bOps["alpha"]
    
    else:
        model = b1*bOps["b1b1"] +\
                b2*bOps["b1b2"] +\
                (-2/7)*(b1-1)*bOps["b1bs"] +\
                b3*bOps["b1b3"] +\
                alpha*bOps["alpha"]
    
    sigma2 = yerr**2

    return -0.5*np.sum((y-model)**2/sigma2)

def logUniPrior(pars, bounds):
    
    inBounds = True
    for par, bound in zip(pars, bounds):
        inBounds &= (bound[0]<par<bound[1])
            
    if inBounds:
        return 0
    
    return -np.inf

def logPosterior(pars, bounds, y, yerr, bOps, isCross):
    
    lp = logUniPrior(pars, bounds)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp+logNormLikelihood(pars, bounds, y, yerr, bOps, isCross)