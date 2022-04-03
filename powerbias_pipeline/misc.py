import numpy as np
import sys
import time
import warnings

def printflush(string, rank="ROOT"):

    caller = sys._getframe().f_back.f_code.co_filename
    if "/" in caller:
        caller = caller.split("/")[-1]
        
    t = time.time()
    print("[{:.0f}] cpu{}//{}: {}".format(t, rank, caller, string))
    sys.stdout.flush()
    
def _warning(message,
             category = UserWarning,
             filename = "",
             lineno = -1,
             file=None,
             line=None):
    printflush("WARNING: {}".format(message))

nintp = 2
nintph = 1

def assignDoubleGrid(r, L, Ng):

    # particle position is adjusted for periodicity and normalized
    r = (r+L)%L
    r = r/L if np.all(r/L<=1) else np.ones(3)

    # position of the particle scaled to the number of points in the grid
    Nv = np.zeros(3, dtype=int)+Ng
    vr = Nv*r

    if nintp==2:
        # first grid
        iv = assignIndex(vr.astype(int), Nv)
        w1 = assignWeights2(vr)

    W_real, idx = [], []
    # Assign to the 8 cells from the 6 indexes found with assign_index
    for x in range(nintp):
        for y in range(nintp):
            for z in range(nintp):
                W_real.append(w1[0, x]*w1[1, y]*w1[2, z])
                idx.append((iv[0, x], iv[1, y], iv[2, z]))

    return W_real, idx

# For CIC it finds the indices of the 2 cells that delimit the cube of 8 cells
# along the 3 directions
def assignIndex(v, Nv):

    idx = np.zeros((3, 4), dtype=int)

    for j in range(nintp):
        idx[:, j] = (v+j+Nv)%Nv

    return idx

# Computes the weights
def assignWeights2(v):

    w = np.zeros((3, 4))

    for i in range(3):
        h = v[i]-int(v[i])
        w[i, 0] = 1-h
        w[i, 1] = h

    return w

def computeDeconvolve(index, H, kf, kNyq):

    i, j, l = index
    
    ki = kf*i
    kj = kf*j
    kl = kf*l
    k = np.sqrt(ki**2+kj**2+kl**2)

    if 0<k<=kNyq:
        Wkx = np.sin(ki*H/2)/(ki*H/2) if ki!=0 else 1
        Wky = np.sin(kj*H/2)/(kj*H/2) if kj!=0 else 1
        Wkz = np.sin(kl*H/2)/(kl*H/2) if kl!=0 else 1

        return k, (Wkx*Wky*Wkz)**2

    else:
        return k, np.inf

def computeSinglePk(k, knorms, deltak1, deltak2, L, Ng, kf):

    Pks, ks = [], []
    for i in range(Ng):
        for j in range(Ng):
            for l in range(Ng):
                if k-kf/2<knorms[i, j, l]<=k+kf/2:
                    Pks.append((deltak1[i, j, l]*deltak2[i, j, l].conjugate()).real/L**3)
                    ks.append(knorms[i, j, l])

    meanPk = sum(Pks)/len(Pks) if len(Pks)!=0 else 0
    singlek = sum(ks)/len(ks) if len(ks)!=0 else 0

    sigmaPk = 0
    for i in Pks:
        sigmaPk += (i-meanPk)**2

    sigmaPk = np.sqrt(sigmaPk)/(len(Pks)-1) if len(Pks)>1 else 0

    return singlek, meanPk, sigmaPk

def logNormLikelihood(pars, y, yerr, model, mask):
    
    b1, b2 = pars
    model = model(b1, b2)
    sigma2 = yerr**2
    return -0.5*np.sum(((y-model)**2/sigma2+np.log(2*np.pi*sigma2))[mask])
    
def logUniPrior(pars):
    
    b1, b2 = pars
    if 0<b1<100 and -10<b2<5000:
        return 0
    return -np.inf
    
def logPosterior(pars, y, yerr, model, mask):
    
    lp = logUniPrior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp+logNormLikelihood(pars, y, yerr, model, mask)