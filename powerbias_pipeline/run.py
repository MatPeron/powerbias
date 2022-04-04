import argparse as ap
import subprocess as sp
from time import strftime
import os
import ps
import misc
import yaml
import bias

def _warning(message,
             category = UserWarning,
             filename = "",
             lineno = -1,
             file=None,
             line=None):
    misc.printflush("WARNING: {}".format(message))

warnings.showwarning = _warning

parser = ap.ArgumentParser(description="Computes the Power Spectrum P(k) of a simulated catalog of sources and measures the bias parameters up to 1-loop. For large datasets it's strongly advised to run this script using openMPI (add mpirun before issuing python).\n"
                                       "See \033[31mexample/example.yaml\033[0m for an a guide on how to set up an init file.\n"
                                       "\n"
                                       "\033[7m*** USAGE ***\033[0m\n"
                                       "To compute auto P(k):\n"
                                       "    python3 run.py /path/to/init_file autopk\n"
                                       "\n"
                                       "To compute cross P(k):\n"
                                       "    python3 run.py /path/to/init_file crosspk /path/to/pk_file1 /path/to/pk_file2"
                                       "To compute P(k) and bias:\n"
                                       "    python3 run.py /path/to/init_file (autopk/crosspk [args])  --do-bias\n"
                                       "\n"
                                       "To compute bias from P(k) file:\n"
                                       "    python3 run.py /path/to/init_file load /path/to/pk_file\n"
                                       "*************\n",
                           formatter_class=ap.RawDescriptionHelpFormatter)

# initialization file containing the params
parser.add_argument("init_file",
                    metavar="/path/to/init_file",
                    help="path to initialization file")

subparsers = parser.add_subparsers(help="sub-commands", dest="command")

# sub-command to compute auto P(k) and (optionally) the bias
apk_subparser = subparsers.add_parser("autopk", help = "specify to compute auto P(k)")
apk_subparser.add_argument("--do-bias",
                           action="store_true",
                           help = "specify to compute bias")

# sub-command to compute cross P(k) and (optionally) the bias
cpk_subparser = subparsers.add_parser("autopk", help = "specify to compute cross P(k)")
cpk_subparser.add_argument("pk_file1",
                           metavar="/path/to/pk_file1",
                           help="path to first auto P(k) file")
cpk_subparser.add_argument("pk_file2",
                           metavar="/path/to/pk_file2",
                           help="path to second auto P(k) file")
cpk_subparser.add_argument("--do-bias",
                           action="store_true",
                           help = "specify to compute bias")

# sub-command to compute the bias from an existing P(k)
bias_subparser = subparsers.add_parser("load", help = "specify to load P(k) and compute bias, requires P(k) file")
bias_subparser.add_argument("pk_file",
                             metavar="/path/to/pk_file",
                             help="path to P(k) file")

args = parser.parse_args()

if "mpirun" in os.environ["_"] or any("MPI" in i for i in os.environ.keys()):
    misc.printflush("Running program through MPI parallelization")
    use_mpi = True

else:
    misc.printflush("Running program with multiprocessing package")
    use_mpi = False

# import init file
init_file = args.init_file
with open(init_file, "r") as f:
    init = yaml.safe_load(f)

# save output in init_file path with same name as init file
out_path = init_file[:-5]

if args.command=="autopk":
    do_bias = args.do_bias
        
    # create temporary file in cwd to avoid import problems, and import readSim function
    sp.run(["cp", init["Pk_parameters"]["read_sim_func"]["file_path"], "./tmp.py"])
    from tmp import readSim
    
    # prepare parameters for P(k) computation
    pk_params = {"L": init["Pk_parameters"]["L"],
                 "Ng": init["Pk_parameters"]["Ng"],
                 "read_sim_func": readSim,
                 "read_sim_args": init["Pk_parameters"]["read_sim_func"]["args"],
                 "nproc": init["System_parameters"]["nproc"],
                 "use_mpi": use_mpi,
                 "filename": init["Pk_parameters"]["simfile"]}

    # create power spectrum object and initialize the parameters
    pk = ps.PowerSpectrum(**pk_params)

    # perform computation
    pk.computeAutoPk()
    
    # delete temporary files
    sp.run(["rm", "tmp.py"])
    
    if do_bias:
        # prepare parameters for bias computation
        bias_params = {"powerspectrum_obj": pk,
                       "kcut": init["MCMC_parameters"]["kcut"],
                       "seed": init["MCMC_parameters"]["seed"],
                       "Ndim": init["MCMC_parameters"]["Ndim"],
                       "Nwalkers": init["MCMC_parameters"]["Nwalkers"],
                       "Nsteps": init["MCMC_parameters"]["Nsteps"],
                       "initial_guess": init["MCMC_parameters"]["initial_guess"]}
        bias_params.update(init["MCMC_parameters"]["cosmo_pars"])
        
        bs = bias.BiasSampler(**bias_params)
        bs.fit()
    
    out_path += "_POWERBIAS_autopk_{}_{}/".format("bias" if do_bias else "nobias", strftime("%Y%m%d%H%M"))
    sp.run(["mkdir", out_path])
    
    # save copy of init_file in out_path
    sp.run(["cp", init_file, out_path])
    # save power spectrum in out_path
    pk.save(out_path+"PowerSpectrumObj", save_plot=True)
    # save bias posterior if --do-bias flag has been called
    bs.save(out_path+"BiasObj", save_plot=True) if do_bias else None
    
    misc.printflush("Files saved in {}".format(out_path))
    
elif args.command=="crosspk":
    do_bias = args.do_bias
    pk_file1 = args.pk_file1
    pk_file2 = args.pk_file2
    
    # prepare parameters for P(k) computation
    pk_params = {"L": init["Pk_parameters"]["L"],
                 "Ng": init["Pk_parameters"]["Ng"],
                 "nproc": init["System_parameters"]["nproc"],
                 "use_mpi": use_mpi}

    # create power spectrum objects
    pk1 = ps.PowerSpectrum()
    pk1.load(pk_file1)
    pk2 = ps.PowerSpectrum()
    pk2.load(pk_file2)
    
    crosspk = ps.PowerSpectrum(**pk_params)

    # perform computation
    crosspk.computeCrossPk(pk1, pk2)
    
    if do_bias:
        # prepare parameters for bias computation
        bias_params = {"powerspectrum_obj": crosspk,
                       "kcut": init["MCMC_parameters"]["kcut"],
                       "seed": init["MCMC_parameters"]["seed"],
                       "Ndim": init["MCMC_parameters"]["Ndim"],
                       "Nwalkers": init["MCMC_parameters"]["Nwalkers"],
                       "Nsteps": init["MCMC_parameters"]["Nsteps"],
                       "initial_guess": init["MCMC_parameters"]["initial_guess"]}
        bias_params.update(init["MCMC_parameters"]["cosmo_pars"])
        
        bs = bias.BiasSampler(**bias_params)
        bs.fit()
    
    out_path += "_POWERBIAS_crosspk_{}_{}/".format("bias" if do_bias else "nobias", strftime("%Y%m%d%H%M"))
    sp.run(["mkdir", out_path])
    
    # save copy of init_file in out_path
    sp.run(["cp", init_file, out_path])
    # save power spectrum in out_path
    pk.save(out_path+"PowerSpectrumObj", save_plot=True)
    # save bias posterior if --do-bias flag has been called
    bs.save(out_path+"BiasObj", save_plot=True) if do_bias else None
    
    misc.printflush("Files saved in {}".format(out_path))

elif args.command=="load":
    pk_file = args.pk_file
    
    pk = ps.PowerSpectrum()
    pk.load(pk_file)
        
    bias_params = {"powerspectrum_obj": pk,
                   "kcut": init["MCMC_parameters"]["kcut"],
                   "seed": init["MCMC_parameters"]["seed"],
                   "Ndim": init["MCMC_parameters"]["Ndim"],
                   "Nwalkers": init["MCMC_parameters"]["Nwalkers"],
                   "Nsteps": init["MCMC_parameters"]["Nsteps"],
                   "initial_guess": init["MCMC_parameters"]["initial_guess"]}
    bias_params.update(init["MCMC_parameters"]["cosmo_pars"])

    bs = bias.BiasSampler(**bias_params)
    bs.fit()
    
    out_path = "/" if pk_file[0]=="/" else ""
    out_path += "/".join(pk_file.split("/")[:-1])+"/"
    
    # save copy of init_file in out_path
    sp.run(["cp", init_file, out_path])
    # save bias posterior
    bs.save(out_path+"BiasObj", save_plot=True)
    
    misc.printflush("Files saved in {}".format(out_path))
