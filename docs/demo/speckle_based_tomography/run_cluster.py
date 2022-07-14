"""
To run python script on the DLS cluster
"""
import subprocess


python_script = "reconstruct_single_slice_from_raw_data_DLS.py"
# python_script = "retrieve_phase_all_projections_DLS.py"
# python_script = "reconstruct_from_phase_projections.py"

cluster_output = "/dls/tmp/speckle_phase/cluster_msg/"

gpu = True
if gpu:
    bash_script = """
    module load hamilton;
    mydir=`pwd`
    mkdir -p {0}; 
    cmd="module load python/recast3d;python -u $mydir/{1}"
    echo $cmd | qsub -e {0} -o {0} -l gpu=1 -P i12
    """.format(cluster_output, python_script)
    subprocess.call(bash_script, shell=True)
else:
    bash_script = """
    module load hamilton;
    mydir=`pwd`
    mkdir -p {0}; 
    cmd="module load python/recast3d;python -u $mydir/{1}"
    echo $cmd | qsub -e {0} -o {0} -P i12
    """.format(cluster_output, python_script)
    subprocess.call(bash_script, shell=True)

print("*****************************")
print("!!!!!!!! Job submitted !!!!!!")
print("*****************************")

