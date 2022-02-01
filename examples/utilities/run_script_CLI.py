#!/home/apps/python/anaconda/1.7.0/64/bin/python

"""
The following example shows how to write a command line interface (CLI)
script to run from a terminal (Linux). Users need to add the body of
the script to suit their need.

Remember to replace the link to a python interpreter above and make
the script executable using the command "chmod a+x the_script.py"

"""

import argparse
import algotom.io.loadersaver as losa


parser = argparse.ArgumentParser(description="This script is used to reconstruct a few slices of tomographic data")
parser.add_argument("-s", dest="scan_name", help="Name of tomographic data", type=str, required=True)
parser.add_argument("-f", dest="flat_name", help="Name of flat-field data", type=str, required=True)
parser.add_argument("-d", dest="dark_name", help="Name of dark-field data", type=str, required=True)
parser.add_argument("-c", dest="center", help="Center of rotation", type=float, required=False, default=0.0)

parser.add_argument("--cleft", "--cleft", dest="cleft", help="Crop left", default=0, type=int, required=False)
parser.add_argument("--cright", "--cright", dest="cright", help="Crop right", default=0, type=int, required=False)
parser.add_argument("--startslice", "--startslice", dest="startslice", help="Start position of reconstructed slices",
                    default=0, type=int, required=False)
parser.add_argument("--stopslice", "--stopslice", dest="stopslice", help="End position of reconstructed slices",
                    default=-1, type=int, required=False)
parser.add_argument("--stepslice", "--stepslice", dest="stepslice", help="Gap between reconstructed slices",
                    default=1, type=int,required=False)
args = parser.parse_args()

scan_name = args.scan_name
flat_name = args.flat_name
dark_name = args.dark_name
center = args.center

crop_left = args.cleft
crop_right = args.cright

start_slice = args.startslice
stop_slice = args.stopslice
step_slice = args.stepslice

input_base = "/home/user_id/raw_data/"
output_base = "/home/user_id/processing/reconstruction/"

proj_path = losa.find_file(input_base + "/*" + scan_name + "*.hdf")[0]
flat_path = losa.find_file(input_base + "/*" + flat_name + "*.hdf")[0]
dark_path = losa.find_file(input_base + "/*" + dark_name + "*.hdf")[0]
hdf_key = "/entry/data/data"

output_name = losa.make_folder_name(output_base, "Recon_few_slices")

print("******************")
print("Run the script....")
print("******************")

# Add body here
# ...
# ...

print("******************")
print("!!!!!! Done !!!!!!")
print("******************")

