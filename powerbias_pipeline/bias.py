import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import fastpt as fpt
import camb
import emcee
import misc
import warnings
import pickle
import corner

def _warning(message,
             category = UserWarning,
             filename = "",
             lineno = -1,
             file=None,
             line=None):
    misc.printflush("WARNING: {}".format(message))

warnings.showwarning = _warning

class BiasSampler:
    
    def __init__(self, **kwargs):
        
        self.force_Nsteps = False
        
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
            
    def save(self, filename, save_plot=False):

        not_to_be_saved = ["powerspectrum_obj"]
            
        if filename[-4:]!=".pkl":
            filename += ".pkl"
            
        if save_plot:
            self.cornerPlot(filename)
            self.pkPlot(filename)

        with open("{}".format(filename), "wb") as f:
            save_data = {k: v for k, v in self.__dict__.items() if k not in not_to_be_saved}
            save_data.update({k: None for k in not_to_be_saved})
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, filename):

        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
        
    def cornerPlot(self, filename):
        
        if any(i in filename for i in [".png", ".jpg", ".gif", ".pdf"]):
            pass
        
        else:
            filename += "_corner.png"
        
        fig = corner.corner(self.samples, labels=[r"$b_1$", r"$b_2$"])
        
        axes = np.array(fig.axes).reshape((self.Ndim, self.Ndim))
        for i in range(self.Ndim):
            ax = axes[i, i]
            ax.axvline(self.best[i], color="r")
            ax.tick_params(axis="both", which='both', top=True, bottom=True, left=True, right=True, direction="in")
            ax.tick_params(axis="both", which="major", length=10)
            ax.tick_params(axis="both", which="minor", length=4)
    
        ax = axes[1, 0]
        ax.axvline(self.best[0], color="r")
        ax.axhline(self.best[1], color="r")
        ax.plot(self.initial_guess[0], self.initial_guess[1], marker="d", ls="none", color="r", markerfacecolor="w")
        
        ax.tick_params(axis="both", which='both', top=True, bottom=True, left=True, right=True, direction="in")
        ax.tick_params(axis="both", which="major", length=10)
        ax.tick_params(axis="both", which="minor", length=4)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(filename, dpi=200)
        
    def pkPlot(self, filename):
        
        if any(i in filename for i in [".png", ".jpg", ".gif", ".pdf"]):
            pass
        
        else:
            filename += "_plot.png"
        

        mask = self.powerspectrum_obj.k<=self.kcut

        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.errorbar(self.powerspectrum_obj.k[mask],
                    self.powerspectrum_obj.Pk[mask]-self.powerspectrum_obj.shotnoise,
                    yerr=self.powerspectrum_obj.sigmaPk[mask], marker="o", ms=3, ls="none")
        
        model = self.compute1loopModel()
        ax.plot(self.powerspectrum_obj.k[mask], model(*self.best)[mask], color="k")
        b1s, b2s = self.samples[:, 0], self.samples[:, 1]
        b1s, b2s = b1s[np.logical_and(b1s>self.percs[0, 0], b1s<=self.percs[1, 0])], \
                   b2s[np.logical_and(b2s>self.percs[0, 1], b2s<=self.percs[1, 1])]
        for i in range(100):
            b1 = np.random.choice(b1s)
            b2 = np.random.choice(b2s)
            ax.plot(self.powerspectrum_obj.k[mask], model(b1, b2)[mask], color="0.7", lw=0.3, alpha=0.5)
            
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
        
        sampler = emcee.EnsembleSampler(self.Nwalkers,
                                        self.Ndim,
                                        posterior,
                                        args=posterior_args)

        misc.printflush("starting sampler with {} walkers for {} steps maximum".format(self.Nwalkers, self.Nsteps))
        
        ###
        
        index = 0
        autocorr = np.empty(self.Nsteps)
        old_tau = np.inf

        for sample in sampler.sample(initial_guess+1e-3*np.random.randn(self.Nwalkers, self.Ndim),
                                     iterations=self.Nsteps,
                                     progress=True,
                                     tune=True):
            if self.force_Nsteps:
                continue
                
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            
            old_tau = tau
        
        ###
        
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
        
