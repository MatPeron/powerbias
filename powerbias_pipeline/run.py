import argparse as ap
import subprocess as sp
from time import strftime
import os
import ps
import warnings
import misc
import yaml

warnings.showwarning = misc._warning

parser = ap.ArgumentParser(description="Computes the Power Spectrum P(k) of a simulated catalog of sources and measures the bias parameters up to 1-loop. For large datasets it's strongly advised to run this script using openMPI (add mpirun before issuing python).\n"
                                       "See \033[31mexample/example.yaml\033[0m for an a guide on how to set up an init file.\n"
                                       "\n"
                                       "\033[7m*** USAGE ***\033[0m\n"
                                       "To compute P(k) (but no bias):\n"
                                       "    python3 run.py /path/to/init_file do_pk /path/to/sim_file\n"
                                       "\n"
                                       "To compute P(k) and bias:\n"
                                       "    python3 run.py /path/to/init_file do_pk --do-bias /path/to/sim_file\n"
                                       "\n"
                                       "To compute bias from P(k) file:\n"
                                       "    python3 run.py /path/to/init_file load /path/to/pk_file\n"
                                       "*************\n",
                           formatter_class=ap.RawDescriptionHelpFormatter)

# initialization file containing the params
parser.add_argument("init_file",
                    metavar="/path/to/init_file",
                    help="path to initialization file")

subparsers = parser.add_subparsers(help="sub-commands")

# sub-command to compute P(k) and (optionally) the bias
pk_subparser = subparsers.add_parser("do_pk", help = "specify to compute P(k), requires simulation file")
pk_subparser.add_argument("sim_file",
                          metavar="/path/to/sim_file",
                          help="path to simulation file")
pk_subparser.add_argument("--do-bias",
                          action="store_true",
                          help = "specify to compute bias")

# sub-command to compute the bias from an existing P(k)
bias_subparser = subparsers.add_parser("load", help = "specify to load P(k) and compute bias, requires P(k) file")
bias_subparser.add_argument("pk_file",
                             metavar="/path/to/pk_file",
                             help="path to P(k) file")

args = parser.parse_args()

if "mpirun" in os.environ["_"]:
    misc.printflush("Running program through MPI parallelization")
    use_mpi = True

else:
    misc.printflush("Running program with multiprocessing package")
    use_mpi = False

init_file = args.init_file
if hasattr(args, "sim_file"):
    sim_file = args.sim_file
    do_bias = args.do_bias
    
    if do_bias:
        print("BIAS COMPUTATION TO BE IMPLEMENTED");quit()

    # import init file
    with open(init_file, "r") as f:
        init = yaml.safe_load(f)
        
    # create temporary file in cwd to avoid import problems and import readSim function
    sp.run(["cp", init["Pk_parameters"]["read_sim_func"]["file_path"], "./tmp.py"])
    from tmp import readSim
    
    # prepare parameters for P(k) computation
    pk_params = {"L": init["Pk_parameters"]["L"],
                 "Ng": init["Pk_parameters"]["Ng"],
                 "read_sim_func": readSim,
                 "read_sim_args": init["Pk_parameters"]["read_sim_func"]["args"],
                 "nproc": init["System_parameters"]["nproc"],
                 "use_mpi": use_mpi,
                 "filename": sim_file}

    # create power spectrum object and initialize the parameters
    pk = ps.PowerSpectrum(**pk_params)

    # perform computation
    pk.computeAutoPk()
    
    # create save folder in init_file path
    if "/" in init_file:
        out_path = "/".join(init_file.split("/")[:-1])
        
        if init_file[0]=="/":
            out_path = "/"+out_path
            
        if init_file[-1]!="/":
            out_path += "/"
            
    else:
        out_path = "./"
        
    out_path += "pk_{}_output{}/".format("bias" if do_bias else "nobias", strftime("%Y%m%d%H%M"))
    sp.run(["mkdir", out_path])
    
    # save power spectrum and copy of init_file in out_path
    pk.save(out_path+"test")
    sp.run(["cp", init_file, out_path])
    
    misc.printflush("Files saved in {}".format(out_path))

elif hasattr(args, "pk_file"):
    pk_file = args.pk_file
    print("BIAS COMPUTATION TO BE IMPLEMENTED");quit()

# delete temporary files
sp.run(["rm", "tmp.py"])
