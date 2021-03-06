import os
import logging
import argparse
import subprocess

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    description='iterate over simulation folders and create a plot for each simulation run')

parser.add_argument('--base-dir', metavar='dir', type=str,
                    help='path to the parent-folder of simulation results',
                    default=os.path.join('..', 'CTRNN_Simulation_Results', 'data'))

parser.add_argument('--plot-file-name', metavar='filename', type=str,
                    help='the filename under which to store the plots',
                    default='plot.png')

parser.add_argument('--handle-existing', type=str,
                    help='what to do when the plot-file already exists? options: skip, replace, raise',
                    default='skip')

args = parser.parse_args()

simulation_folders = [f.path for f in os.scandir(args.base_dir) if f.is_dir()]

for sim in simulation_folders:
    file = os.path.join(sim, args.plot_file_name)
    if os.path.exists(file):
        if args.handle_existing == 'skip':
            logging.info("skipping because file already exist" + str(file))
            continue
        elif args.handle_existing == 'raise':
            raise RuntimeError("File already exists: " + str(file))
        elif args.handle_existing == 'replace':
            logging.info("replacing existing file: " + str(file))

    # instead of calling the other file, it would probably better to move the generation of the plot into a
    # separate module which is used both by plot_experiment.py and this script
    subprocess.run(["python", "neuro_evolution_ctrnn/plot_experiment.py",
                    "--dir", sim,
                    "--no_show",
                    "--save_png", file])
