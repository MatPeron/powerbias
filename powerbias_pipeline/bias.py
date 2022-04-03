import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import fastpt as fpt
import camb
import emcee
import misc
import warnings
import pickle

warnings.showwarning = misc._warning

class BiasSampler:
    
    def __init__(self, **kwargs):
        
        self.powerspectrum_obj = None
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
        self.initial_guess = None
        
        # output
        self.tau = None
        self.burnin = None
        self.thin = None
        self.samples = None
        self.log_prob_samples = None
        self.best = None
        self.percs = None
        
        if kwargs!={}:
            for k, v in kwargs.items():
                setattr(self, k, v)
        
    def showParameters(self):

        print("Parameters:\n")
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if value.shape[0]>3:
                    string = "  {:<15} = ".format(key)+"{}...".format(value[:3])

            else:
                string = "  {:<15} = ".format(key)+"{}".format(value)

            print(string)
            
    def save(self, filename):

        not_to_be_saved = ["powerspectrum_obj"]
            
        if filename[-4:]!=".pkl":
            filename += ".pkl"

        with open("{}".format(filename), "wb") as f:
            save_data = {k: v for k, v in self.__dict__.items() if k not in not_to_be_saved}
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
        
    def compute1loopModel(self):
        
        # linear matter power spectrum with camb
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
        ks, z, PkL = results.get_matter_power_spectrum(var1=2,
                                                       var2=2,
                                                       minkh=self.powerspectrum_obj.kf,
                                                       maxkh=self.powerspectrum_obj.kNyq,
                                                       npoints = 1000)
                                             
        # 1-loop tracer power spectrum with fastPT
        to_do = ['one_loop_dd', "dd_bias"]
        pad_factor = 1
        n_pad = pad_factor*len(ks)
        low_extrap = -5
        high_extrap = 3
        P_window = None
        C_window = .75

        fpt_obj = fpt.FASTPT(ks, to_do=to_do, low_extrap=low_extrap, high_extrap=high_extrap, n_pad=n_pad)
        P1loop = fpt_obj.one_loop_dd_bias(PkL[0], C_window=C_window, P_window=P_window)
        
        # interpolate 1-loop tracer power spectrum with cubic splines onto the measured modes
        keys = ["PNLO", "PLO", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2"]
        b_ops = {"is_cross": self.powerspectrum_obj.is_cross}
        for i in range(len(keys)):
            CSintrp = CubicSpline(ks, P1loop[i])
            b_ops[keys[i]] = CSintrp(self.powerspectrum_obj.k)
        
        if self.powerspectrum_obj.is_cross:
            model = lambda b1, b2: b1*(b_ops["PLO"]+b_ops["PNLO"])+b2*b_ops["Pd1d2"]/2-1/7*(b1-1)*b_ops["Pd1s2"]
            
        else:
            model = lambda b1, b2: b1**2*(b_ops["PLO"]+b_ops["PNLO"]) +\
                                   b1*b2*b_ops["Pd1d2"] +\
                                   b2**2*b_ops["Pd2d2"]/4 +\
                                   b1*(-2/7)*(b1-1)*b_ops["Pd1s2"] +\
                                   b2*(-2/7)*(b1-1)*b_ops["Pd2s2"]/2 +\
                                   ((-2/7)*(b1-1))**2*b_ops["Ps2s2"]/4
            
        return model
    
    def fit(self, likelihood="auto", posterior="auto", posterior_args=None):
        
        if (likelihood=="auto" and posterior!="auto") or (likelihood!="auto" and posterior=="auto"):
            raise ValueError("Arguments likelihood and posterior must be both assigned together or left both to the default \"auto\" value")
            
        if posterior!="auto" and posterior_args is None:
            raise ValueError("A custom posterior function has been assigned, but no arguments for it have been provided through posterior_args argument. Please specify a tuple for it containing the powerspectrum data to be computed and any other positional argument")
        
        np.random.seed(self.seed)
        
        mask = self.powerspectrum_obj.k<=self.kcut
        
        if posterior=="auto":
            posterior = misc.logPosterior
            posterior_args = (self.powerspectrum_obj.Pk-self.powerspectrum_obj.shotnoise,
                              self.powerspectrum_obj.sigmaPk,
                              self.compute1loopModel(),
                              mask)
        
        sampler = emcee.EnsembleSampler(self.Nwalkers,
                                        self.Ndim,
                                        posterior,
                                        args=posterior_args)
        
        
        initial_guess = np.asarray(self.initial_guess)
        
        if likelihood=="auto":
            likelihood = misc.logNormLikelihood
        
        inverse_likelihood = lambda *args: -likelihood(*args)
        
        # to be implemented: a way to adjust the bounds on the parameters
        soln = minimize(inverse_likelihood,
                        initial_guess+0.1*np.random.randn(*initial_guess.shape),
                        args=(posterior_args),
                        bounds=[(0, 100), (-10, 5000)])
        misc.printflush("{} -- optimized guess -> {}".format(initial_guess, soln.x))
        initial_guess = soln.x

        misc.printflush("starting sampler with {} walkers for {} steps".format(self.Nwalkers, self.Nsteps))
        sampler.run_mcmc(initial_guess+1e-3*np.random.randn(self.Nwalkers, self.Ndim),
                         self.Nsteps,
                         progress=True,
                         tune=True)
        
        self.tau = sampler.get_autocorr_time()
        self.burnin = int(2 * np.max(self.tau))
        self.thin = int(0.5 * np.min(self.tau))
        self.samples = sampler.get_chain(discard=self.burnin,
                                         flat=True,
                                         thin=self.thin)
        self.log_prob_samples = sampler.get_log_prob(discard=self.burnin,
                                                     flat=True,
                                                     thin=self.thin)
        
        self.best = self.samples[np.argmax(self.log_prob_samples)]
        self.percs = np.percentile(self.samples, [16, 84], axis=0)
        
        misc.printflush("MCMC sampling complete")
        