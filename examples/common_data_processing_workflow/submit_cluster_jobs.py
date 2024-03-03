#!/users/conda/envs/2023-3.3-py310/bin/python
"""
This script is used for submitting a list of cluster jobs (using the SLURM scheduler)
to reconstruct a list of tomographic datasets in embarrassingly parallel fashion.
"""
import os
import sys
import time
import signal
import subprocess

# Specify the folder for cluster output-file and error-file.
cluster_base = "/facl/data/beamline/proposals/2024/pass-123456/tomography/raw_data//cluster_msg/"

list_proj_scan = ["scan_00005", "scan_00007", "scan_00009"]
list_dark_flat = ["scan_00006", "scan_00008", "scan_00010"]

start_slice = 0
stop_slice = -1
crop_left = 0
crop_right = 0

center = 0.0  # For auto-determination
ratio = 0.0
output_format = "hdf"
ring = "all"
zing = 0
method = "fbp"
ncore = 0  # For auto-selection
num_iteration = 1  # For iterative reconstrution method

if method != "gridrec":
    use_gpu = True
else:
    use_gpu = False


print("\n=============================================")
print("          Submit jobs to cluster             ")
print("          Time: {}".format(time.ctime(time.time())))
print("=============================================\n")

python_script = "full_reconstruction.py"  #  Make sure it's executable (chmod a+x <script.py>)

sbatch_script_cpu = """#!/bin/bash

#SBATCH --job-name=tomo_recon
#SBATCH --ntasks 1
#SBATCH --output={0}/output_%j.out
#SBATCH --cpus-per-task 32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --time=45:00

srun -o {0}/output_%j.out -e {0}/error_%j.err ./{1} -p {2} -d {3} -c {4} -r {5} -f {6} --start {7} \\
--stop {8} --left {9} --right {10} --ring {11} --zing {12} --method {13} --ncore {14} --iter {15}
"""

sbatch_script_gpu = """#!/bin/bash

#SBATCH --job-name=tomo_recon
#SBATCH --ntasks 1
#SBATCH --output={0}/output_%j.out
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --time=45:00

srun -o {0}/output_%j.out -e {0}/error_%j.err ./{1} -p {2} -d {3} -c {4} -r {5} -f {6} --start {7} \\
--stop {8} --left {9} --right {10} --ring {11} --zing {12} --method {13} --ncore {14} --iter {15}
"""

def make_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Failed to create directory: '{path}'. Error: {e}")

def submit_job(sbatch_script):
    process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=sbatch_script.encode())
    return stdout.decode().strip(), stderr.decode().strip()

def check_job_status():
    process = subprocess.Popen(['squeue', '--me'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode().strip(), stderr.decode().strip()

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def signal_handler(sig, frame):
    print("\n*********************************************")
    print("           !!!!!! Exit !!!!!!                ")
    print("*********************************************")
    sys.exit(0)

# =============================================================================
# Main
# =============================================================================

if len(list_proj_scan) != len(list_dark_flat):
    raise ValueError("Number of scans vs number of dark-flar scans is not the same!!!")

for i, scan in enumerate(list_proj_scan):
    cluster_output = cluster_base + "/" + scan + "/"
    print("Submit to process the scan : {}...".format(scan))
    make_folder(cluster_output)
    if use_gpu is True:
        sbatch_script = sbatch_script_gpu.format(cluster_output, python_script,
                list_proj_scan[i], list_dark_flat[i], center, ratio, output_format,
                start_slice, stop_slice, crop_left, crop_right,
                ring, zing, method, ncore, num_iteration)
    else:
        sbatch_script = sbatch_script_cpu.format(cluster_output, python_script,
                list_proj_scan[i], list_dark_flat[i], center, ratio, output_format,
                start_slice, stop_slice, crop_left, crop_right,
                ring, zing, method, ncore, num_iteration)
    stdout, stderr = submit_job(sbatch_script)
    print(stdout)
    print(stderr)

time.sleep(5)
signal.signal(signal.SIGINT, signal_handler)
time_gap = 10
# Live update job status
while True:
    clear_screen()
    msg, _ = check_job_status()
    print(msg)
    time.sleep(time_gap)