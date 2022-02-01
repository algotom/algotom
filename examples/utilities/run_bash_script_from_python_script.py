"""
The following example shows how to write a python script to call a bash
script. This is useful for submitting a list of jobs to reconstruct
different scans on a cluster.

"""

import subprocess

python_script = "run_script_CLI.py"
cluster_output = "/home/user_id/processing/cluster_output_ms/"

scan_list = ["scan_00001", "scan_00004", "scan_00005"]
flat_name = "scan_00002"
dark_name = "scan_00003"
for scan_name in scan_list:
    bash_script = """
    module load cluster;
    mydir=`pwd`
    mkdir -p {0}; 
    cmd="module load python;python -u $mydir/{1} -s {2} -f {3} -d {4}"
    #echo $cmd | qsub -e {0} -o {0} -l gpu=1
    echo $cmd | qsub -e {0} -o {0} -P i12
    """.format(cluster_output + "/" + scan_name + "/", python_script, scan_name, flat_name, dark_name)
    subprocess.call(bash_script, shell=True)
print("******************")
print("!!!!!! Done !!!!!!")
print("******************")
