#!/facility/conda/envs/python_lib/bin/python
import os
import sys
import glob
import time
import signal
import subprocess

proposal_id = "commissioning/pass-123456"

# Specify the folder for cluster output-file and error-file.
cluster_base = "/nsls2/data/hex/proposals/" + proposal_id+ "/tomography/processed/cluster_msg/"

list_proj_scan = [21, 22]
list_dark_flat = [20, 22]

start_slice = 0
stop_slice = 3200
crop_left = 0
crop_right = 0

center = 0.0 # For auto-determination
ratio = 0.0 # For denoising
output_format = "hdf"
ring = "all"
zing = 1
method = "fbp"
ncore = 0  # For auto-selection
num_iteration = 1  # For iterative reconstrution method

if method == "gridrec":
    use_gpu = False
else:
    use_gpu = True


print("\n=============================================")
print("          Submit jobs to cluster             ")
print("          Time: {}".format(time.ctime(time.time())))
print("=============================================\n")

python_script = "full_reconstruction_360_cli.py"

sbatch_script_cpu = """#!/bin/bash

#SBATCH --job-name=tomo_recon
#SBATCH --ntasks 1
#SBATCH --output={0}/output_%j.out
#SBATCH --cpus-per-task 32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --time=120:00

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
#SBATCH --time=120:00

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

# =====================================================================================
# Main

if len(list_proj_scan) != len(list_dark_flat):
    raise ValueError("Number of scans vs number of dark-flar scans is not the same!!!")

for i, scan in enumerate(list_proj_scan):
    proj_scan_name = "scan_" + ("0000" + str(scan))[-5:]
    cluster_output = cluster_base + "/" + proj_scan_name + "/"
    print("Submit to process the scan : {}...".format(proj_scan_name))
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
# Show job status
while True:
    clear_screen()
    msg, _ = check_job_status()
    print(msg)
    time.sleep(time_gap)