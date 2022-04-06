from scipy import fft
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import pickle
import warnings
import os
import sys
import misc

def _warning(message,
             category = UserWarning,
             filename = "",
             lineno = -1,
             file=None,
             line=None):
    misc.printflush("WARNING: {}".format(message))

warnings.showwarning = _warning

class PowerSpectrum:

    def __init__(self, **kwargs):

        self.use_mpi = False
        self.nproc = -1
        self.filename = None
        self.read_sim_func = None
        self.read_sim_args = None
        self.preprocess = True

        self.L = None
        self.Ng = None

        # derived parameters
        self.H = None
        self.Np = None
        self.kf = None
        self.ks = None
        self.kNyq = None

        # da implementare
        self.MAS_order = 2
        self.is_intrl = False
        self.is_cross = False
        self.is_avrg = False

        # output
        self.knorms = None
        self.deltak = None
        self.k = None
        self.Pk = None
        self.sigmaPk = None
        self.shotnoise = None
        
        if kwargs!={}:
            for k, v in kwargs.items():
                setattr(self, k, v)

            self.H = self.L/self.Ng
            self.kf = 2*np.pi/self.L
            self.ks = 2*np.pi/self.H
            self.kNyq = np.pi/self.H
        
            try:
                if self.preprocess:
                    self.preprocessCatalog()

                else:
                    self.Np = len(self.read_sim_func(self.filename, **self.read_sim_args))
            except:
                warnings.warn("Could not load catalog because arguments read_sim_func, read_sim_args or filename were not specifies. This may result in an error if computing an auto P(k)")

        self._multiprocImports()

    def _multiprocImports(self):
        
        if self.nproc==-1:
            self.nproc = os.cpu_count()

        elif self.nproc>os.cpu_count():
            warnings.warn("nproc value exceeds the number of cpu cores present in the machine, setting nproc to total number of cores")
            self.nproc = os.cpu_count()

        if self.use_mpi:
            global MPI
            from mpi4py import MPI

        else:
            global mp, partial
            import multiprocessing as mp
            from functools import partial
            
    def _computeChunkSize(self, size, single_output_memory="to be added", bandwidth="to be added", factor=4):
        
        # add code that accounts for memory usage
        
        chunksize = size//(factor*self.nproc)
        chunksize += 1 if size%(factor*self.nproc) else 0
        
        return chunksize

    def showParameters(self):

        print("Parameters:\n")
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if value.shape[0]>3:
                    string = "  {:<15} = ".format(key)+"{}...".format(value[:3])

            else:
                string = "  {:<15} = ".format(key)+"{}".format(value)

            print(string)

    def save(self, filename, save_deltak=True, save_plot=False):

        not_to_be_saved = ["use_mpi", "nproc", "read_sim_func", "read_sim_args"]

        if not save_deltak:
            not_to_be_saved.extend(["knorms", "deltak"])
            
        if save_plot:
            self.pkPlot(filename)
            
        if filename[-4:]!=".pkl":
            filename += ".pkl"

        with open("{}".format(filename), "wb") as f:
            save_data = {k: v for k, v in self.__dict__.items() if k not in not_to_be_saved}
            save_data.update({k: None for k in not_to_be_saved})
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):

        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)

    def preprocessCatalog(self):
        
        positions = self.read_sim_func(self.filename, **self.read_sim_args)
        np.save("pos_tmp.npy", positions)
        
        self.filename = "pos_tmp.npy"
        self.read_sim_func = lambda file: np.load(file)
        self.read_sim_args = {}
        self.Np = len(positions)
    
    def pkPlot(self, filename):
        
        if any(i in filename for i in [".png", ".jpg", ".gif", ".pdf"]):
            pass
        
        else:
            filename += "_plot.png"
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.errorbar(self.k, self.Pk-self.shotnoise, yerr=self.sigmaPk, marker="o", ms=3, ls="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}\,$Mpc$^3$]")
        
        ax.tick_params(axis="both", which='both', top=True, bottom=True, left=True, right=True, direction="in")
        ax.tick_params(axis="both", which="major", length=10)
        ax.tick_params(axis="both", which="minor", length=4)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        
        fig.savefig(filename, dpi=200)
    
    def computeKbins(self):

        return np.arange(1.5*self.kf, self.kNyq//self.kf*self.kf, self.kf)

    # to be done: implement different weighting functions from CIC
    def computeMAS(self):

        if self.use_mpi:
            misc.printflush("starting MAS in MPI mode")
            comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=["MAS_worker.py"],
                                       maxprocs=self.nproc)
            parameters = {k: v for k, v in self.__dict__.items() if k in ["L", "Ng", "Np", "H"]}
            comm.bcast(parameters, root=MPI.ROOT)

            n = self.Np//(self.nproc)
            r = self.Np%(self.nproc)

            for i in range(self.nproc):
                if i<r:
                    start = i*(n+1)
                    stop = start+n+1

                else:
                    start = i*n+r
                    stop = start+n

                local_pos = self.read_sim_func(self.filename, **self.read_sim_args)[start:stop].astype(float)
                comm.Send(local_pos, dest=i, tag=42)

            rho = np.zeros((self.Ng, self.Ng, self.Ng))
            for i in range(self.nproc):
                tmp = np.empty((self.Ng, self.Ng, self.Ng))
                comm.Recv(tmp, source=i, tag=35)

                rho += tmp

                misc.printflush("root has received MAS results from {}".format(i))

            deltar = rho/self.Np*self.L**3-1

            comm.Disconnect()
            
        else:
            positions = self.read_sim_func(self.filename, **self.read_sim_args)
            positions, counts = np.unique(positions, return_counts=True, axis=0)
            pool = mp.Pool(self.nproc)
            chunksize = self._computeChunkSize(counts.size)
            
            rho = np.zeros((self.Ng, self.Ng, self.Ng))
            misc.printflush("starting MAS in multiprocessing mode")
            for position, count in tqdm(zip(pool.imap(partial(misc.assignDoubleGrid,
                                                              L=self.L,
                                                              Ng=self.Ng),
                                                      positions,
                                                      chunksize=chunksize),
                                            counts),
                                        total=counts.size):
                weights, indexes = position
                
                for weight, index in zip(weights, indexes):
                    rho[index] += count*weight/self.H**3
                    
            deltar = rho/self.Np*self.L**3-1
            
            pool.terminate()
            
        if self.preprocess:
            sp.run(["rm", "pos_tmp.npy"])

        return deltar

    def computeFFT(self, deltar):

        misc.printflush("starting FFT in multiprocessing mode")
        deltak = fft.fftn(deltar, workers=self.nproc)*self.H**3

        return deltak

    def deconvolve(self, deltak):

        if self.use_mpi:
            misc.printflush("starting deconvolution in MPI mode")
            comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=["deconvolve_worker.py"],
                                       maxprocs=self.nproc)
            parameters = {k: v for k, v in self.__dict__.items() if k in ["Ng", "H", "kf", "kNyq"]}
            comm.bcast(parameters, root=MPI.ROOT)

            knorms = np.zeros((self.Ng, self.Ng, self.Ng))
            Wk = np.zeros((self.Ng, self.Ng, self.Ng))
            for i in range(self.nproc):
                tmp = np.empty((self.Ng, self.Ng, self.Ng))
                comm.Recv(tmp, source=i, tag=63)

                knorms += tmp

                tmp = np.empty((self.Ng, self.Ng, self.Ng))
                comm.Recv(tmp, source=i, tag=24)

                Wk += tmp

                misc.printflush("root has received knorm and Wk results from {}".format(i))

            deltak = deltak/Wk

            comm.Disconnect()
            
        else:            
            knorms = np.zeros((self.Ng, self.Ng, self.Ng))
            Wk = np.zeros((self.Ng, self.Ng, self.Ng))
            misc.printflush("starting deconvolution in single core mode")
            for m in tqdm(range(self.Ng**3)):
                index = (m//(self.Ng**2), m%(self.Ng**2)//self.Ng, m%self.Ng)
                knorms[index], Wk[index] = misc.computeDeconvolve(index, self.H, self.kf, self.kNyq)
                
            deltak = deltak/Wk

        self.knorms = knorms
        self.deltak = deltak

        return knorms, deltak

    def computePk(self, knorms=None, deltak1=None, deltak2=None):
        
        if knorms is None:
            knorms = np.copy(self.knorms)
        
        if deltak1 is None:
            deltak1 = np.copy(self.deltak)
            
        if deltak2 is None:
            deltak2 = np.copy(self.deltak)
            
        if knorms is None or deltak1 is None or deltak2 is None:
            raise ValueError("computePk method requires knorms and deltak arguments")
        
        kbins = self.computeKbins()

        if self.use_mpi:
            misc.printflush("starting P(k) computation in MPI mode")
            comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=["Pk_worker.py"],
                                       maxprocs=self.nproc)
            parameters = {k: v for k, v in self.__dict__.items() if k in ["L", "Ng", "kf"]}
            parameters["knorms"], parameters["deltak1"], parameters["deltak2"] = knorms, deltak1, deltak2
            comm.bcast(parameters, root=MPI.ROOT)

            n = kbins.size//self.nproc
            r = kbins.size%self.nproc

            nbins = []
            for i in range(self.nproc):
                if i<r:
                    start = i*(n+1)
                    stop = start+n+1

                else:
                    start = i*n+r
                    stop = start+n

                nbins.append(stop-start)
                comm.Send(np.array(stop-start, dtype=int), dest=i, tag=73)
                comm.Send(kbins[start:stop], dest=i, tag=99)

            ks, Pks, sigmaPks = [], [], []
            for i, nbin in zip(range(self.nproc), nbins):
                tmp = np.zeros(nbin)

                comm.Recv(tmp, source=i, tag=89)
                ks.extend(list(tmp))

                comm.Recv(tmp, source=i, tag=12)
                Pks.extend(list(tmp))

                comm.Recv(tmp, source=i, tag=90)
                sigmaPks.extend(list(tmp))

            misc.printflush("P(k) computed")

            comm.Disconnect()
            
        else:
            pool = mp.Pool(self.nproc)
            chunksize = self._computeChunkSize(kbins.size)
            
            ks, Pks, sigmaPks = [], [], []
            misc.printflush("starting P(k) computation in multiprocessing mode")
            for pk in tqdm(pool.imap(partial(misc.computeSinglePk,
                                             knorms=knorms,
                                             deltak1=deltak1,
                                             deltak2=deltak2,
                                             L=self.L,
                                             Ng=self.Ng,
                                             kf=self.kf),
                                     kbins,
                                     chunksize=chunksize),
                           total=kbins.size):
                ks.append(pk[0])
                Pks.append(pk[1])
                sigmaPks.append(pk[2])
                
            pool.terminate()

        self.k = np.array(ks)
        self.Pk = np.array(Pks)
        self.sigmaPk = np.array(sigmaPks)
        self.shotnoise = self.L**3/self.Np
        
    def computeAutoPk(self):
        
        misc.printflush("Computing auto power spectrum...")
        T = time()
        
        t = time()
        deltar = self.computeMAS()
        t = time()-t
        misc.printflush("completed in {:.0f}m{:.0f}s".format(t//60, t%60))
        
        t = time()
        deltak = self.computeFFT(deltar)
        t = time()-t
        misc.printflush("completed in {:.0f}m{:.0f}s".format(t//60, t%60))
        
        t = time()
        self.deconvolve(deltak)
        t = time()-t
        misc.printflush("completed in {:.0f}m{:.0f}s".format(t//60, t%60))
        
        t = time()
        self.computePk()    
        t = time()-t
        misc.printflush("completed in {:.0f}m{:.0f}s".format(t//60, t%60))
        
        T = time()-T
        misc.printflush("Done. Total run time {:.0f}m{:.0f}s".format(T//60, T%60))
        
    def computeCrossPk(self, obj1, obj2):
    
        self.is_cross = True
    
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
            if obj1.knorms.shape!=obj2.knorms.shape:
                raise ValueError("Incompatible shapes:\n"
                                 "parameter knorms in obj1 has shape {}\n"
                                 "parameter knorms in obj2 has shape {}".format(obj1.knorms.shape, obj2.knorms.shape))
        
        except AttributeError:
            raise TypeError("Parameter 'knorms' in either obj1 or obj2 (or both) is not an array. Make sure to use instances that that contain knorms.")
            
        if not np.all(np.isclose(obj1.knorms, obj1.knorms, rtol=1e-3)):
            raise ValueError("The Fourier modes of the two power spectra differ by more than 0.1%. Make sure that the power spectra have been computed on boxes of the same size.")
        
        misc.printflush("Computing cross power spectrum...")
        T = time()
        
        self.Np = np.inf # needed to give 0 shotnoise
        self.computePk(obj1.knorms, obj1.deltak, obj2.deltak)
            
        T = time()-T
        print("Done. Total run time {:.0f}m{:.0f}s".format(T//60, T%60))
              
    def computeAvrgPk(self, sigma_type="var", **PowerSpectrumObjects):
    
        self.is_avrg = True
        
        misc.printflush("Computing average power spectrum of {}...".format(len(PowerSpectrumObjects)))
        T = time()
        
        n = 0 
        for key, obj in PowerSpectrumObjects.items():
            if not isinstance(obj, type(self)):
                raise TypeError("{} is not an instance of PowerSpectrum".format(key))
                                
            if n==0:
                self.k = obj.k
                self.Pk = obj.Pk-obj.shotnoise
                oldkey = key
                n += 1
                
            else:
                try:
                    if obj.k.shape!=self.k.shape:
                        raise ValueError("Incompatible shapes:\n"
                                         "parameter k in {} has shape {}\n"
                                         "parameter k in {} has shape {}".format(key, obj.k.shape, oldkey, self.k.shape))
                    
                    if not np.all(np.isclose(obj.k, self.k, rtol=1e-3)):
                        raise ValueError("The Fourier modes of {} and {} differ by more than 0.1%. Make sure that the power spectra have been computed on boxes of the same size.".format(key, oldkey))
                    
                    if obj.Pk.shape!=self.Pk.shape:
                        raise ValueError("Incompatible shapes:\n"
                                         "parameter Pk in {} has shape {}\n"
                                         "parameter Pk in {} has shape {}".format(key, obj.Pk.shape, oldkey, self.Pk.shape))
                                                   
                    self.k += obj.k
                    self.Pk += obj.Pk-obj.shotnoise
                    oldkey = key
                    n += 1
                    
                except AttributeError:
                    raise TypeError("Parameters k or Pk (or both) in either {} or {} (or both) are not an array. Make sure to use instances that that contain k and Pk.".format(key, oldkey))
                    
        self.k /= n
        self.Pk /= n
        
        if sigma_type=="var":
            self.sigmaPk = np.zeros(self.Pk.size)
            for key, obj in PowerSpectrumObjects.items():
                self.sigmaPk += (obj.Pk-obj.shotnoise-self.Pk)**2
            
            self.sigmaPk = np.sqrt(self.sigmaPk)/(n-1)
            
        elif sigma_type=="cov":
            self.sigmaPk = np.zeros((self.Pk.size, self.Pk.size))
            for row in range(self.Pk.size):
                for col in range(row+1):
                    for key in kwargs:
                        self.sigmaPk[row, col] += (obj.Pk[row]-obj.shotnoise-self.Pk[row])*(obj.Pk[col]-obj.shotnoise-self.Pk[col])/(n-1)
                        
                self.pars["sigmaPk"][col, row] = self.pars["sigmaPk"][row, col]
            
        T = time()-T
        print("Done. Total run time {:.0f}m{:.0f}s".format(T//60, T%60))    
