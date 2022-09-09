# powerbias
Pure Python code that computes power spectrum and/or bias from a simulated catalog of cosmological sources.
Features:
 - fast interpolation of the density field in multiprocessing mode through the built-in python `multiprocessing` package or through openMP (suggested for large collections of simulated particles);
 - can be easily expanded to higher order interpolation schemes (currently supports order 2);
 - fast Fourier transform and deconvolution thanks to `numpy`;
 - defines a `PowerSpectrum` object to standardize operations;
 - save and load `PowerSpectrum` objects to disk in a standard binary format thanks to `pickle`;
 - can compute the cross-correlation power spectrum between two `PowerSpectrum` instances;
 - can compute the average power spectrum between multiple `PowerSpectrum` instances, together with their covariance matrix;
 - can compute the Feldman-Kaiser-Peacock estimator and iteratively optimize its weights for high accuracy;
 - can perform interlacing to mitigate aliasing effects on high frequency modes;
 - can sample a user-defined posterior distribution on a generic power spectrum model for Bayesian inference.

Documentation is currently being built.

Requires the following packages:
 - `numpy`
 - `matplotlib`
 - `scipy`
 - `mpi4py`
 - `tqdm`
 - `camb`
 - `emcee`
 - `corner`
 - `velocileptors`
