import os
import shutil

output_directory = './runfiles/output/'
if os.path.isdir(output_directory):
    shutil.rmtree(output_directory, ignore_errors=True)
os.path.mkdir(output_directory)

