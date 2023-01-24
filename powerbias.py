import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import warnings
import os
import sys
import functions


abspath = "/".join(__file__.split("/")[:-1])+"/"

def _warning(message,
             category=UserWarning,
             filename="",
             lineno=-1,
             file=None,
             line=None):
    
    functions.printflush("WARNING: {}".format(message))
    
warnings.showwarning = _warning

class Timer:    
    def __enter__(self):
    
        self.laps = 0
        self.start = time.perf_counter()
    
        return self

    def __exit__(self, *args):
    
        self.end = time.perf_counter()
        self.CPUTime = self.end-self.start
        
    def getTimeFromStart(self):
        
        """
        Gives the time from the initialization of the object.
        
        Returns
        -------
        float
            Seconds from initialization.
        """
    
        return time.perf_counter()-self.start
        
    def getLapTime(self):
        
        """
        Gives the time between consecutive calls of the function.
        
        Returns
        -------
        float
            Seconds from last call.
        """
        
        self.lap = self.getTimeFromStart()-self.laps
        self.laps += self.lap
            
        return self.lap

class PowerSpectrum:

    """
    An object that represents all the properties of a power spectrum computed from the spatial
    position of particles within the cubic box of a cosmological simulation.
    
    Attributes
    ----------
    ignoreWarnings : bool, optional, default False
        flag to enable or disable warnings.
    verbose : bool, optional, default True
        flag to enable or disable prints.
    useMpi : bool, optional, default False
        flag to enable or disable openMP parallelization.
    nCPU : int, optional, default 0
        number of cores to be used in parallel computation. If the number is less than 1 or exceeds
        the number of cores in the machine, it defaults to the number of cores in the machine.
    filename : string, optional
        string that refers to the file where the collection of points is stored in 
        (e.g. /path/to/simulation_file).
    L : float, optional
        size of the box in units of [:math:`h{^-1}\,`Mpc].
    Ng : int, optional
        number of equally-spaced grid points in which the box is subdivided to perform interpolation.
        Should be a power of 2.
    H : float, optional
        size of the interval between consecutive grid points in units of [:math:`h{^-1}\,`Mpc].
    Np : int, optional
        number of particles.
    kf : float, optional
        fundamental Fourier mode of the box in units of [:math:`h\,`Mpc:math:`^{-1}`].
    ks : float, optional
        sampling Fourier mode of the grid in units of [:math:`h\,`Mpc:math:`^{-1}`].
    kNyq : float, optional
        Nyquist Fourier mode of the grid in units of [:math:`h\,`Mpc:math:`^{-1}`].
    isIntrl : bool, optional, default False
        flag to specify and/or enable interlacing of the power spectrum.
    isCross : bool, optional, default False
        flag that specifies the power spectrum is the cross-correlation between different sets of
        particles.
    isAvrg : bool, optional, default False
        flag that specifies the power spectrum is the average of multiple power spectra.
    MASOrder : int, optional, default 2
        order of the B-spline used for the interpolation scheme.
    kNorms : ndarray, optional
        array of shape (`Ng`, `Ng`, `Ng`) containing the vector magnitude of the Fourier modes. The 
        vectors associated to each element is just the index of the element multiplied by `kf`.
    deltak : complex ndarray, optional
        array of shape (`Ng`, `Ng`, `Ng`) containing the values of the Fourier-transformed overdensity 
        field of the particles.
    k : ndarray, optional
        array containing values of the average magnitude of the Fourier modes within certain bins in
        Fourier space.
    Pk : ndarray, optional
        array containing values of the power spectrum estimator within certain bins in Fourier space.
    sigmaPk : ndarray, optional
        array containing values of the Gaussian standard deviation of the power spectrum within 
        certain bins in Fourier space.
    shotNoise : float or ndarray, optional
        shot noise to be subtracted to `Pk`.
        
    Methods
    -------
    _setParameters(**kwargs)
        called at initialization, sets the values of the class Attributes.
    _necessaryImports()
        called at initialization, imports packages necessary for the class but not necessary for the
        whole module to function.
    _conditionalImports()
        called at initialization, imports packages based on the class Attributes.
    computeChunkSize(size, factor=4)
        gives an estimate of the optimal number of jobs-per-CPU to be used by the 
        `multiprocessing.Pool.imap` method, called when `useMpi` is set to False.
    showParameters()
        prints the class Attributes to screen.
    save(filename, saveDeltak=True, savePlot=False)
        saves the current class instance to disk under `filename`. Also saves a plot of the power
        spectrum if `savePlot` is set to True. Not all class Attributes are saved (`ignoreWarnings`,
        `verbose`, `useMpi` and `nCPU` are set to None in the saved file), setting `saveDeltak` to 
        False will set `kNorms` and `deltak` to None, as saving them can lead to large file size.
    load(filename)
        loads a class instance previously saved to disk under `filename` onto the current class 
        instance.
    pkPlot(filename)
        saves a plot of the power spectrum to disk under `filename`.
    computeKbins()
        gives an array containing equally-spaced bin edges in Fourier space. The bins have a width of
        `kf` and go up to `kNyq`.
    computeMAS(catalog)
        gives the number of particles interpolated onto a regular grid of shape (`Ng`, `Ng`, `Ng`)
        according to weights given by a B-spline of order `MASOrder`.
    computeDeltar(N)
        gives the overdensity field given the interpolated number of particles `N`. 
    computeFFT(deltar)
        gives the unnormalized Fourier transform of the overdensity field `deltar`. To normalize use
        the 'normalizeDeltak' method.
    deconvolve(deltak, chunkSize=256)
        gives the Fourier transform of the overdensity field, deconvolved to remove the effects of the
        interpolation scheme.
    normalizeDeltak(deltak)
        gives the normalized Fourier transform of the overdensity field
    computePk(self, kNorms, deltak1, deltak2, kbins=None, returnMask=False):
        gives the power spectrum values, the square root of their gaussian variance and the respective
        Fourier modes by averaging `kNorms` and the product of `deltak1` and `deltak2` inside each 
        `kbin`. `computeKbins` is called if `kbins` is None.
    computeShotNoise()
        gives the shot noise to be subtracted to `Pk`.
    storeResults(kNorms=None, deltak=None, k=None, Pk=None, sigmaPk=None, shotNoise=None)
        stores the relevant power spectrum quantities inside the class instance.
    computeAutoPk(catalog)
        wraps all the operations needed to compute an auto-power spectrum directly from a `catalog`
        of sources.
    computeCrossPk(obj1, obj2)
        wraps all the operations needed to compute a cross-power spectrum between two class instances.
        this method should be called on a third instance that exists independendently of the two.
    computeAvrgPk(sigmaType="var", **PowerSpectrumObjects)
        wraps all the operations needed to compute an average power spectrum between multiple class 
        instances. This method should be called on a third instance that exists independently of the
        others.
    """

    def __init__(self, **kwargs):

        self.ignoreWarnings = False
        self.verbose = True
        self.useMpi = False
        self.nCPU = 0
        self.filename = None

        self.L = None
        self.Ng = None

        # derived parameters
        self.H = None
        self.Np = None
        self.kf = None
        self.ks = None
        self.kNyq = None

        self.isIntrl = False
        self.isCross = False
        self.isAvrg = False
        # da implementare per valori > 2
        self.MASOrder = 2

        # output
        self.kNorms = None
        self.deltak = None
        self.k = None
        self.Pk = None
        self.sigmaPk = None
        self.shotNoise = None
        
        self._setParameters(**kwargs)
        self._necessaryImports()
        self._conditionalImports()
        
    def _setParameters(self, **kwargs):
    
        if kwargs!={}:
            try:
                if kwargs["ignoreWarnings"]:
                    warnings.filterwarnings("ignore", category=UserWarning)
            
            except:
                pass
            
            for k, v in kwargs.items():
                if k in self.__dict__:
                    setattr(self, k, v)
                    
                else:
                    warnings.warn("Invalid input parameter: {}. It is being ignored, however this may throw an exception later.".format(k))

            assert not np.log2(self.Ng)%1, "'Ng' should be a power of two"
            self.H = self.L/self.Ng
            self.kf = 2*np.pi/self.L
            self.ks = 2*np.pi/self.H
            self.kNyq = np.pi/self.H

    def _necessaryImports(self):
    
        global fft
        
        from scipy import fft
    
    def _conditionalImports(self):
        
        minCPU = 1
        maxCPU = os.cpu_count()
        
        if self.nCPU>maxCPU:
            warnings.warn("nCPU value ({}) exceeds the number of cpu cores present in the machine ({}), setting nCPU to total number of cores".format(self.nCPU, maxCPU))
            self.nCPU = maxCPU

        elif self.nCPU<minCPU:
            warnings.warn("nCPU value ({}) is less than one, setting nCPU to total number of cores ({})".format(self.nCPU, maxCPU))
            self.nCPU = maxCPU

        if self.useMpi:
            global MPI
            from mpi4py import MPI
        
        else:
            global mp, partial, tqdm
            import multiprocessing as mp
            from functools import partial
            from tqdm import tqdm
            
        global pflush
        if self.verbose:
            from functions import printflush as pflush
        
        else:
            from functions import fakePrintflush as pflush
        
    def computeChunkSize(self, size, factor=4):
        
        # add code that accounts for memory usage
        
        chunkSize = size//(factor*self.nCPU)
        chunkSize += 1 if size%(factor*self.nCPU) else 0
        
        return chunkSize

    def showParameters(self):

        print("Parameters:\n")
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if value.shape[0]>3:
                    string = "  {:<15} = ".format(key)+"{}...".format(value[:3])

            else:
                string = "  {:<15} = ".format(key)+"{}".format(value)

            print(string)

    def save(self, filename, saveDeltak=True, savePlot=False):

        notToBeSaved = ["ignoreWarnings", "verbose", "useMpi", "nCPU"]

        if not saveDeltak:
            notToBeSaved.extend(["kNorms", "deltak"])
            
        if savePlot:
            self.pkPlot(filename)
            
        if filename[-4:]!=".pkl":
            filename += ".pkl"

        with open("{}".format(filename), "wb") as f:
            saveData = {k: v for k, v in self.__dict__.items() if k not in notToBeSaved}
            saveData.update({k: None for k in notToBeSaved})
            pickle.dump(saveData, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):

        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
    
    def pkPlot(self, filename):
        
        if any(i in filename for i in [".png", ".jpg", ".gif", ".pdf"]):
            pass
        
        else:
            filename += "_plot.png"
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.errorbar(self.k, self.Pk-self.shotNoise, yerr=self.sigmaPk, marker="o", ms=3, ls="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}\,$Mpc$^3$]")
        
        ax.tick_params(axis="both", which='both', top=True, bottom=True, left=True, right=True, direction="in")
        ax.tick_params(axis="both", which="major", length=10)
        ax.tick_params(axis="both", which="minor", length=4)

        fig.tight_layout()
        
        fig.savefig(filename, dpi=200)
        plt.close()
    
    def computeKbins(self):

        return np.linspace(self.kf, self.kNyq, int(self.kNyq//self.kf))

    # to be done: implement different weighting functions from CIC
    def computeMAS(self, catalog):

        Np = len(catalog)

        if self.useMpi:
            pflush("starting MAS in MPI mode")
            comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=[abspath+"MAS_worker.py"],
                                       maxprocs=self.nCPU)
            parameters = {k: v for k, v in self.__dict__.items() if k in ["L",
                                                                          "Ng",
                                                                          "H",
                                                                          "isIntrl",
                                                                          "MASOrder",
                                                                          "verbose"]}
            parameters["Np"] = Np
            comm.bcast(parameters, root=MPI.ROOT)

            n = Np//(self.nCPU)
            r = Np%(self.nCPU)

            for i in range(self.nCPU):
                if i<r:
                    start = i*(n+1)
                    stop = start+n+1

                else:
                    start = i*n+r
                    stop = start+n

                localPos = catalog[start:stop].astype(float)
                comm.Send(localPos, dest=i, tag=42)

            N = np.zeros((self.Ng, self.Ng, self.Ng))
            for i in range(self.nCPU):
                tmp = np.empty((self.Ng, self.Ng, self.Ng))
                comm.Recv(tmp, source=i, tag=35)

                N += tmp

                pflush("root has received MAS results from {}".format(i))

            comm.Disconnect()
            
        else:
            positions, counts = np.unique(catalog, return_counts=True, axis=0)
            pool = mp.Pool(self.nCPU)
            chunkSize = self.computeChunkSize(counts.size)
            
            N = np.zeros((self.Ng, self.Ng, self.Ng))
            pflush("starting MAS in multiprocessing mode")
            for position, count in tqdm(zip(pool.imap(partial(functions.assignDoubleGrid,
                                                              L=self.L,
                                                              Ng=self.Ng,
                                                              isIntrl=self.isIntrl),
                                                      positions,
                                                      chunksize=chunkSize),
                                            counts),
                                        total=counts.size):
                weights, indexes = position
                
                for weight, index in zip(weights, indexes):
                    N[index] += count*weight
                    
            pool.terminate()
            
        N /= 2 if self.isIntrl else 1

        return N

    def computeDeltar(self, N):
    
        return N/self.H**3/self.Np*self.L**3-1

    def computeFFT(self, deltar):

        pflush("starting FFT in multiprocessing mode")
        deltak = fft.fftn(deltar, workers=self.nCPU)

        return deltak

    def deconvolve(self, deltak, chunkSize=256):
    
        def f(arr, mod):
            
            mask = arr<mod
            
            arr[mask] = arr[mask]%mod
            arr[~mask] = arr[~mask]%mod-mod
        
            return arr
        
        default = np.geterr()
        np.seterr(invalid="ignore")
        
        chunkSize = self.Ng if self.Ng<chunkSize else chunkSize
        chunks = self.Ng//chunkSize if self.Ng>chunkSize else 1
        
        kNorms, Wk = np.empty(0), np.empty(0)
        for i in range(chunks**3):
            binArray = np.arange(i*chunkSize**3, (i+1)*chunkSize**3)
            kVectors = np.c_[f(binArray//self.Ng**2, self.Ng//2),
                             f((binArray%self.Ng**2)//self.Ng, self.Ng//2),
                             f(binArray%self.Ng, self.Ng//2)]*self.kf
            WkVectors = np.sin(kVectors*self.H/2)/kVectors/self.H*2
            np.nan_to_num(WkVectors, nan=1, copy=False)
        
            kNorms = np.append(kNorms, np.sqrt(np.sum(kVectors**2, axis=-1)))
            Wk = np.append(Wk, (np.prod(WkVectors, axis=-1)**2))
            
        kNorms = kNorms.reshape(self.Ng, self.Ng, self.Ng)
        Wk = Wk.reshape(self.Ng, self.Ng, self.Ng)
        
        deltak /= Wk
        
        np.seterr(**default)
        
        return kNorms, deltak
        
    def normalizeDeltak(self, deltak):
    
        return deltak/self.L**(3/2)*self.H**3

    def computePk(self, kNorms, deltak1, deltak2, kbins=None, returnMask=False):
    
        kbins = self.computeKbins() if kbins is None else kbins
            
        allPks = (deltak1*deltak2.conjugate()).real
        
        pflush("starting P(k) computation in single core mode")
        
        k, Pk, sigmaPk = np.empty(0), np.empty(0), np.empty(0)
        for i in range(0, kbins.size-1):
            mask = np.logical_and(kNorms>=kbins[i], kNorms<kbins[i+1])
            ks = kNorms[mask]
            Pks = allPks[mask]

            meank = np.mean(ks) if len(ks)!=0 else 0
            meanPk = np.mean(Pks) if len(Pks)!=0 else 0
            sqrtGVar = np.sqrt(2/len(Pks))*meanPk if len(Pks)!=0 else 0

            k = np.append(k, meank)
            Pk = np.append(Pk, meanPk)
            sigmaPk = np.append(sigmaPk, sqrtGVar)
            
        if returnMask:
            return mask, k, Pk, sigmaPk
            
        return k, Pk, sigmaPk
        
    def computeShotNoise(self):
    
        return self.L**3/self.Np
        
    def storeResults(self, 
                     kNorms=None,
                     deltak=None,
                     k=None,
                     Pk=None,
                     sigmaPk=None,
                     shotNoise=None):
                     
        self.kNorms = kNorms if kNorms is not None else self.kNorms
        self.deltak = deltak if deltak is not None else self.deltak
        self.k = k if k is not None else self.k
        self.Pk = Pk if Pk is not None else self.Pk
        self.sigmaPk = sigmaPk if sigmaPk is not None else self.sigmaPk
        self.shotNoise = shotNoise if shotNoise is not None else self.shotNoise
        
    def computeAutoPk(self, catalog):
    
        if self.L is None or self.Ng is None:
            raise ValueError("Auto-P(k) cannot be computed if parameters 'L' or 'Ng' aren't set.")
        
        pflush("Computing auto power spectrum...")
        
        with Timer() as t:
            N = self.computeMAS(catalog)
            del catalog
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            deltar = self.computeDeltar(N)
            shotNoise = self.computeShotNoise()
            del N
            
            deltak = self.computeFFT(deltar)
            del deltar
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            kNorms, deltak = self.deconvolve(deltak)
            deltak = self.normalizeDeltak(deltak)
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            k, Pk, sigmaPk = self.computePk(kNorms=kNorms, deltak1=deltak, deltak2=deltak)    
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
        
        self.storeResults(kNorms=kNorms,
                          deltak=deltak,
                          k=k,
                          Pk=Pk,
                          sigmaPk=sigmaPk,
                          shotNoise=shotNoise)
        pflush("Done. Total run time {:.0f}m{:.0f}s".format(t.CPUTime//60, t.CPUTime%60))
        
    def computeCrossPk(self, obj1, obj2):
    
        self.isCross = True
    
        if not isinstance(obj1, type(self)):
            raise TypeError("obj1 ({}) is not an instance of PowerSpectrum".format(obj1))
        
        if not isinstance(obj2, type(self)):
            raise TypeError("obj2 ({}) is not an instance of PowerSpectrum".format(obj2))
    
        try:
            if obj1.deltak.shape!=obj2.deltak.shape:
                raise ValueError("Incompatible shapes:\n"
                                 "parameter deltak in obj1 has shape {}\n"
                                 "parameter deltak in obj2 has shape {}".format(obj1.deltak.shape, obj2.deltak.shape))
        
        except AttributeError:
            raise TypeError("Parameter deltk in either obj1 or obj2 (or both) is not an array. Make sure to use instances that contain deltak.")
                            
        try:
            if obj1.kNorms.shape!=obj2.kNorms.shape:
                raise ValueError("Incompatible shapes:\n"
                                 "parameter kNorms in obj1 has shape {}\n"
                                 "parameter kNorms in obj2 has shape {}".format(obj1.kNorms.shape, obj2.kNorms.shape))
        
        except AttributeError:
            raise TypeError("Parameter 'kNorms' in either obj1 or obj2 (or both) is not an array. Make sure to use instances that that contain kNorms.")
            
        if not np.all(np.isclose(obj1.kNorms, obj1.kNorms, rtol=1e-3)):
            raise ValueError("The Fourier modes of the two power spectra differ by more than 0.1%. Make sure that the power spectra have been computed on boxes of the same size.")
        
        pflush("Computing cross power spectrum...")
        
        with Timer() as t:
            k, Pk, sigmaPk = self.computePk(obj1.kNorms, obj1.deltak, obj2.deltak)
        
        self.storeResults(k=k,
                          Pk=Pk,
                          sigmaPk=sigmaPk,
                          shotNoise=0)
        print("Done. Total run time {:.0f}m{:.0f}s".format(t.CPUTime//60, t.CPUTime%60))
              
    def computeAvrgPk(self, sigmaType="var", **PowerSpectrumObjects):
    
        self.isAvrg = True
        
        pflush("Computing average power spectrum of {}...".format(len(PowerSpectrumObjects)))
        
        with Timer() as t:
            n = 0 
            for key, obj in PowerSpectrumObjects.items():
                if not isinstance(obj, type(self)):
                    raise TypeError("{} is not an instance of PowerSpectrum".format(key))
                                    
                if n==0:
                    k = obj.k
                    Pk = obj.Pk-obj.shotNoise
                    oldKey = key
                    n += 1
                    
                else:
                    try:
                        if obj.k.shape!=k.shape:
                            raise ValueError("Incompatible shapes:\n"
                                             "parameter k in {} has shape {}\n"
                                             "parameter k in {} has shape {}".format(key, obj.k.shape, oldKey, k.shape))
                        
                        if not np.all(np.isclose(obj.k, k, rtol=1e-3)):
                            raise ValueError("The Fourier modes of {} and {} differ by more than 0.1%. Make sure that the power spectra have been computed on boxes of the same size.".format(key, oldKey))
                        
                        if obj.Pk.shape!=Pk.shape:
                            raise ValueError("Incompatible shapes:\n"
                                             "parameter Pk in {} has shape {}\n"
                                             "parameter Pk in {} has shape {}".format(key, obj.Pk.shape, oldKey, Pk.shape))
                                                       
                        k = obj.k
                        Pk += obj.Pk-obj.shotNoise
                        oldKey = key
                        n += 1
                        
                    except AttributeError:
                        raise TypeError("Parameters k or Pk (or both) in either {} or {} (or both) are not an array. Make sure to use instances that that contain k and Pk.".format(key, oldKey))
                        
            Pk /= n
            
            if sigmaType=="var":
                sigmaPk = np.zeros(Pk.size)
                for key, obj in PowerSpectrumObjects.items():
                    sigmaPk += (obj.Pk-obj.shotNoise-Pk)**2
                
                sigmaPk = np.sqrt(sigmaPk/(n-1))
                
            elif sigmaType=="cov":
                sigmaPk = np.zeros((Pk.size, Pk.size))
                for row in range(Pk.size):
                    for col in range(row+1):
                        for key, obj in PowerSpectrumObjects.items():
                            sigmaPk[row, col] += (obj.Pk[row]-obj.shotNoise-Pk[row])*(obj.Pk[col]-obj.shotNoise-Pk[col])/(n-1)
                            
                    sigmaPk[col, row] = sigmaPk[row, col]
                    
        self.storeResults(k=k,
                          Pk=Pk,
                          sigmaPk=sigmaPk,
                          shotNoise=0)
        print("Done. Total run time {:.0f}m{:.0f}s".format(t.CPUTime//60, t.CPUTime%60))    


class FKPowerSpectrum(PowerSpectrum):

    """
    An object that represents all the properties of a power spectrum computed from the spatial
    position of particles within the cubic box of a cosmological simulation. This object implements
    the Feldman-Kaiser-Peacock (FKP) estimator for the power spectrum, and inherits most of the 
    attributes and methods from the base `PowerSpectrum` class.
    
    Attributes
    ----------
    alpha : float, optional
        number that represents the fraction of the density of the real catalog with respect to the
        density of the synthetic catalog required by the FKP estimator
    wPk : float, optional
        value of the power spectrum to be used in the computation of the FKP's optimal weights.
    isOptimized : bool, optional, default False
        flag to specify and/or enable iterative optimization of the FKP's optimal weights.
        
    Methods
    -------
    estimateMinimumParticlesPerBin(alpha=0.1)
        gives the minimum number of particles per grid-cell needed to compute the synthetic catalog.
    generateSyntheticCatalog(n, chunkSize=256)
        gives the synthetic catalog based on the number `n` of particles per grid-cell. `alpha` is
        determined by calling this method.
    computeOptimalWeights(Ns):
        gives the optimal weights based on the interpolated syntetic catalog `Ns`.
    computeFr(Nr, Ns, optWeights)
        gives the FKP estimator in real space.
    normalizeDeltak(deltak, Ns, optWeights)
        gives the normalized Fourier transform of the FKP estimator.
    computeShotNoise(Ns, optWeights)
        gives the shot noise to be subtracted to `Pk`.
    computeAutoPk(realCatalog, n=None, optimizeWeights=False)
        wraps all the operations needed to compute an auto-power spectrum directly from a `realCatalog`
        of sources and a synthetic catalog made with `n` particles per grid-cell (if None, `n` is set
        automatically so that `alpha` is at most 0.1).
    optimizeWeights(Nr, Ns, kbins=None, t=None, maxit=10, threshold=0.01)
        wraps all the operations needed to optimize the weights of the estimator. The optimization is
        performed iteratively inside each bin, for a maximum number of iterations per bin given by
        `maxit` and up until consecutive iterations return values of the power spectrum that are less
        than a percentage `threshold` apart.
    """

    def __init__(self, **kwargs):
        
        self.alpha = None
        self.wPk = None
        self.isOptimized = False
                
        super().__init__(**kwargs)
                
    def estimateMinimumParticlesPerBin(self, alpha=0.1):
    
        return int(1+alpha**(-1)*self.Np/self.Ng**3)
    
    def generateSyntheticCatalog(self, n, chunkSize=256):
    
        self.alpha = self.Np/self.Ng**3/n
    
        if self.alpha>0.1:
            warnings.warn("Number of particles per bin is low, resulting in a value of alpha greater than 0.1 ({:.2f}), this may lead to an inaccurate estimate of P(k)".format(self.alpha))
            
        pflush("Generating synthetic catalog with {} particles per bin".format(n))
        
        chunkSize = self.Ng if self.Ng<chunkSize else chunkSize
        chunks = self.Ng//chunkSize if self.Ng>chunkSize else 1
        
        synthPos = np.empty((0, 3))
        for i in range(chunks**3):
            binArray = np.arange(i*chunkSize**3, (i+1)*chunkSize**3)
            centers = np.c_[binArray//256**2, binArray%(256**2)//256, binArray%256]*self.H
            
            for i in range(n):
                shifts = np.random.uniform(-1, 1, size=(chunkSize**3, 3))*self.H/2
                synthPos = np.append(synthPos, centers+shifts, axis=0)
        
        return np.array(synthPos)
        
    def computeOptimalWeights(self, Ns):
        
        return self.H**3/(self.H**3+self.alpha*Ns*self.wPk)
        
    def computeFr(self, Nr, Ns, optWeights):
        
        return optWeights*(Nr-self.alpha*Ns)
        
    def normalizeDeltak(self, deltak, Ns, optWeights):
    
        W2 = np.sum((self.alpha*Ns*optWeights)**2)/self.H**3
    
        return deltak/np.sqrt(W2)
        
    def computeShotNoise(self, Ns, optWeights):
    
        shotNoise = self.H**3*(1+self.alpha)/self.alpha*np.sum(Ns*optWeights**2/self.H**6) /\
                    np.sum(Ns**2*optWeights**2/self.H**6)
    
        return shotNoise
        
    def computeAutoPk(self,
                      realCatalog,
                      n=None,
                      optimizeWeights=False):
    
        if self.L is None or self.Ng is None or self.Np is None or self.wPk is None:
            raise ValueError("Auto-P(k) cannot be computed if parameters 'L', 'Ng', 'Np' or 'wPk' aren't set.")
        
        pflush("Computing auto power spectrum...")
        
        with Timer() as t:
            Nr = self.computeMAS(realCatalog)
            del realCatalog
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            n = self.estimateMinimumParticlesPerBin() if n is None else n
            synthCatalog = self.generateSyntheticCatalog(n)
            Ns = self.computeMAS(synthCatalog)
            del synthCatalog
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            optWeights = self.computeOptimalWeights(Ns)
            Fr = self.computeFr(Nr, Ns, optWeights)
            shotNoise = self.computeShotNoise(Ns, optWeights)
            if not optimizeWeights:
                del Nr
            
            deltak = self.computeFFT(Fr)
            del Fr
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            kNorms, deltak = self.deconvolve(deltak)
            deltak = self.normalizeDeltak(deltak, Ns, optWeights)
            del optWeights
            if not optimizeWeights:
                del Ns
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
            
            k, Pk, sigmaPk = self.computePk(kNorms=kNorms, deltak1=deltak, deltak2=deltak)    
            lap = t.getLapTime()
            pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
                
            self.storeResults(kNorms=kNorms,
                              deltak=deltak,
                              k=k,
                              Pk=Pk,
                              sigmaPk=sigmaPk,
                              shotNoise=shotNoise)
                          
            if optimizeWeights:
                self.optimizeWeights(Nr, Ns, t=t)
        
        pflush("Done. Total run time {:.0f}m{:.0f}s".format(t.CPUTime//60, t.CPUTime%60))
        
    def optimizeWeights(self, Nr, Ns, kbins=None, t=None, maxit=10, threshold=0.01):
    
        self.isOptimized = True
    
        kbins = self.computeKbins() if kbins is None else kbins
        
        pflush("Optimizing auto power spectrum...")
        
        shotNoiseOpt = np.empty(0)
        for i in range(kbins.size-1):
            pflush("Optimizing bin #{} (out of {})...".format(i+1, kbins.size-1))
            
            it = 1
            while np.abs(self.wPk-self.Pk[i])/self.Pk[i]>=threshold and it<=maxit:
                pflush("Iteration {}".format(it))
                self.wPk = self.Pk[i]
                
                optWeights = self.computeOptimalWeights(Ns)
                Fr = self.computeFr(Nr, Ns, optWeights)
                shotNoise = self.computeShotNoise(Ns, optWeights)
                
                deltak = self.computeFFT(Fr)
                del Fr
                lap = t.getLapTime()
                pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
                
                kNorms, deltak = self.deconvolve(deltak)
                deltak = self.normalizeDeltak(deltak, Ns, optWeights)
                del optWeights
                lap = t.getLapTime()
                pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
                
                mask, k, Pk, sigmaPk = self.computePk(kNorms=kNorms,
                                                      deltak1=deltak,
                                                      deltak2=deltak,
                                                      kbins=kbins[i:i+2],
                                                      returnMask=True)
                del kNorms
                lap = t.getLapTime()
                pflush("completed in {:.0f}m{:.0f}s".format(lap//60, lap%60))
                
                self.deltak[mask] = deltak[mask]
                self.Pk[i] = Pk[0]
                it += 1
                
            self.sigmaPk[i] = sigmaPk[0]
            shotNoiseOpt = np.append(shotNoiseOpt, shotNoise)
            
        self.storeResults(shotNoise=shotNoiseOpt)


class BiasSampler:
    
    def __init__(self, **kwargs):
        
        self.ignoreWarnings = False
        self.verbose = True
        self.forceNSteps = False
        
        self.powerSpectrumObj = None
        self.kcut = None         
        
        self.z = None
        self.Mnu = None
        self.Omm = None
        self.Omb = None
        self.h = None
        self.ns = None
        self.sigma8 = None
        
        self.seed = None
        self.Ndim = None
        self.Nwalkers = None
        self.Nsteps = None
        self.initialGuess = None
        
        # output
        self.tau = None
        self.burnIn = None
        self.thin = None
        self.samples = None
        self.logProbSamples = None
        self.best = None
        self.percs = None
        
        self._setParameters(**kwargs)
        self._necessaryImports()
        self._conditionalImports()
        
    def _setParameters(self, **kwargs):
    
        if kwargs!={}:
            try:
                if kwargs["ignoreWarnings"]:
                    warnings.filterwarnings("ignore", category=UserWarning)
            
            except:
                pass
            
            for k, v in kwargs.items():
                if k in self.__dict__:
                    setattr(self, k, v)
                    
                else:
                    warnings.warn("Invalid input parameter: {}. It is being ignored, however this may throw an exception later.".format(k))
                    
    def _necessaryImports(self):
    
        global minimize, CubicSpline, MomentExpansion, camb, emcee, corner
        
        from scipy.optimize import minimize
        from scipy.interpolate import CubicSpline
        from velocileptors.EPT.moment_expansion_fftw import MomentExpansion
        import camb
        import emcee
        import corner
        
    def _conditionalImports(self):
        
        global pflush
        if self.verbose:
            from functions import printflush as pflush
        
        else:
            from functions import fakePrintflush as pflush
        
    def showParameters(self):

        print("Parameters:\n")
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if value.shape[0]>3:
                    string = "  {:<15} = ".format(key)+"{}...".format(value[:3])

            else:
                string = "  {:<15} = ".format(key)+"{}".format(value)

            print(string)
            
    def save(self, filename, saveCornerPlot=False):

        notToBeSaved = ["ignoreWarnings", "verbose", "powerSpectrumObj"]
        
        if saveCornerPlot:
            self.cornerPlot(filename)
        
        if filename[-4:]!=".pkl":
            filename += ".pkl"

        with open("{}".format(filename), "wb") as f:
            saveData = {k: v for k, v in self.__dict__.items() if k not in notToBeSaved}
            saveData.update({k: None for k in notToBeSaved})
            pickle.dump(saveData, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, filename):

        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
        
    def cornerPlot(self, filename, labels=[r"$b_1$", r"$b_2$", r"$b_3$", r"$\alpha$"]):
        
        if any(i in filename for i in [".png", ".jpg", ".gif", ".pdf"]):
            pass
        
        else:
            filename += "_plot.png"
        
        fig = corner.corner(self.samples, labels=labels)
        
        axes = np.array(fig.axes).reshape((self.Ndim, self.Ndim))
        for i in range(self.Ndim):
            for j in range(i+1):
                ax = axes[i, j]
                
                if i==j:
                    ax.axvline(self.best[i], color="r")

                else:
                    ax.axvline(self.best[j], color="r")
                    ax.axhline(self.best[i], color="r")
                    ax.plot(self.initialGuess[j],
                            self.initialGuess[i],
                            marker="d",
                            ls="none",
                            color="r",
                            markerfacecolor="w")


                ax.tick_params(axis="both", which='both', top=True, bottom=True, left=True, right=True, direction="in")
                ax.tick_params(axis="both", which="major", length=10)
                ax.tick_params(axis="both", which="minor", length=4)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        
        plt.savefig(filename, dpi=200)
        plt.close()
    
    def computeLinearMatterPower(self, kMin=5e-3, kMax=10, Nk=5000):
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.h*100,
                           ombh2=self.Omb*self.h**2,
                           omch2=(self.Omm-self.Omb)*self.h**2,
                           mnu=self.Mnu)
        pars.InitPower.set_params(ns=self.ns)
        pars.WantTransfer = True
        pars.set_matter_power(redshifts=[self.z])
        pars.NonLinear = camb.model.NonLinear_none
        results = camb.get_results(pars)
        k, z, PkL = results.get_matter_power_spectrum(var1=2,
                                                      var2=2,
                                                      minkh=kMin,
                                                      maxkh=kMax,
                                                      npoints=Nk)
        
        return k, PkL[0]
    
    def computeBiasOperators(self, k, PkL, kMin, kMax, Nk=300, mask=None, **kwargs):
        
        mask = np.ones(self.powerSpectrumObj.k.shape, dtype=bool) if mask is None else mask
        
        moments = MomentExpansion(k, PkL,
                                  pnw=None,
                                  kmin=kMin, 
                                  kmax=kMax,
                                  nk=Nk,
                                  **kwargs)
        
        bOps = {}
        keys = ["k", "b1b1", "b1b2", "b1bs", "b2b2", "b2bs", "bsbs", "b1b3", "alpha"]
        for i in range(1, len(keys)):
            bOps[keys[i]] = CubicSpline(moments.kv, moments.pktable[:, i]).__call__(self.powerSpectrumObj.k)[mask]
        
        return bOps
    
    def samplePosterior(self,
                        bounds=[(0, 10), (-1000, 1000), (-1000, 1000), (-1000, 1000)],
                        likelihood=None,
                        posterior=None,
                        posteriorArgs=None,
                        optimizeGuess=True,
                        checkConvergenceEvery=100,
                        returnFullChains=False):
            
        if returnFullChains:
            warnings.warn("Saving the entire chains may require significant amounts of disk space")
        
        if (likelihood is None and posterior is not None) or (likelihood is not None and posterior is None):
            raise ValueError("Arguments 'likelihood' and 'posterior' must be both assigned together")
            
        if posterior is not None and posteriorArgs is None:
            raise ValueError("A custom posterior function has been assigned, but no arguments for it have been provided through 'posteriorArgs' argument. Please specify a tuple for it containing the power spectrum data to be computed and any other argument")
        
        initialGuess = np.asarray(self.initialGuess)
        
        if optimizeGuess:
            inverseLikelihood = lambda *args: -likelihood(*args)
            soln = minimize(inverseLikelihood,
                            initialGuess+0.1*np.random.randn(*initialGuess.shape),
                            args=(posteriorArgs),
                            bounds=bounds)
            pflush("{} -- optimized guess -> {}".format(initialGuess, soln.x))
            initialGuess = soln.x
        
        pflush("starting sampler with {} walkers for {} steps maximum".format(self.Nwalkers, self.Nsteps))
        
        sampler = emcee.EnsembleSampler(self.Nwalkers,
                                        self.Ndim,
                                        posterior,
                                        args=posteriorArgs)
        index = 0
        oldTau = np.inf
        for sample in sampler.sample(initialGuess+1e-3*np.random.randn(self.Nwalkers, self.Ndim),
                                     iterations=self.Nsteps,
                                     progress=True,
                                     tune=True):
            
            if self.forceNSteps:
                continue
            
            if sampler.iteration%checkConvergenceEvery:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            
            converged = np.all(tau*100<sampler.iteration)
            converged &= np.all(np.abs(oldTau-tau)/tau<0.01)
            if converged:
                break
            
            oldTau = tau
        
        tau = sampler.get_autocorr_time()
        pflush("chains are at least {:.0f} times the autocorrelation time ({:.0f}). If this value is more than 100, then it's a good indication that the posterior distribution has been extensively sampled. Further tests should be ran to ensure proper convergence".format(sampler.iteration/np.mean(tau), np.mean(tau)))
        pflush("MCMC sampling complete")
        
        burnIn = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        
        if returnFullChains:
            samples = sampler.get_chain(flat=True)
            logProbSamples = sampler.get_log_prob(flat=True)
        
        else:
            samples = sampler.get_chain(flat=True, discard=burnIn, thin=thin)
            logProbSamples = sampler.get_log_prob(flat=True, discard=burnIn, thin=thin)
                                                  
        return tau, burnIn, thin, samples, logProbSamples
        
    @staticmethod
    def thinChains(samples, logProbSamples, burnIn, thin):
    
        samples = samples[burnIn::thin]
        logProbSamples = logProbSamples[burnIn::thin]
        
        return samples, logProbSamples
    
    @staticmethod
    def computeMAPEstimateAndErrors(samples, logProbSamples, percentiles=[16, 84]):
        
        best = samples[np.argmax(logProbSamples)]
        percs = np.percentile(samples, percentiles, axis=0)
        
        return best, percs
        
    def storeResults(self, 
                     tau=None,
                     burnIn=None,
                     thin=None,
                     samples=None,
                     logProbSamples=None,
                     best=None,
                     percs=None):
                     
        self.tau = tau if tau is not None else self.tau
        self.burnIn = burnIn if burnIn is not None else self.burnIn
        self.thin = thin if thin is not None else self.thin
        self.samples = samples if samples is not None else self.samples
        self.logProbSamples = logProbSamples if logProbSamples is not None else self.logProbSamples
        self.best = best if best is not None else self.best
        self.percs = percs if percs is not None else self.percs
        
    def fit(self):
        
        with Timer() as t:
            mask = self.powerSpectrumObj.k<=self.kcut
            
            kL, PkL = self.computeLinearMatterPower()
            bOps = self.computeBiasOperators(kL, PkL,
                                             kMin=self.powerSpectrumObj.kf,
                                             kMax=self.powerSpectrumObj.kNyq,
                                             Nk=300,
                                             mask=mask)
            
            likelihood = functions.logNormLikelihood
            posterior = functions.logPosterior
            posteriorArgs = (bounds,
                             self.powerSpectrumObj.Pk[mask]-self.powerSpectrumObj.shotNoise,
                             self.powerSpectrumObj.sigmaPk[mask],
                             bOps,
                             self.powerSpectrumObj.isCross)
            
            pflush("Sampling Posterior function...")
            tau, burnIn, thin, samples, logProbSamples = self.samplePosterior(likelihood=likelihood,
                                                                              posterior=posterior,
                                                                              posteriorArgs=posteriorArgs)
                                                                            
            best, percs = self.computeMAPEstimateAndErrors(samples, logProbSamples)
        
        self.storeResults(tau=tau,
                          burnIn=burnIn,
                          thin=thin,
                          samples=samples,
                          logProbSamples=logProbSamples,
                          best=best,
                          percs=percs)
        pflush("Done. Total run time {:.0f}m{:.0f}s".format(t.CPUTime//60, t.CPUTime%60))
        
