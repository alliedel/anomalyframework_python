# break on failures
set -e

# cd to the root directory
user_dir=pwd
SCRIPT=$(readlink -f "$0")
echo $SCRIPT
root_dir=$(dirname "$SCRIPT")


# Install third-party packages
cd ${root_dir}/src/external/attrdict-2.0.0/
python setup.py install --user

# Compile C++ source
cd ${root_dir}/build
cmake ../
make
