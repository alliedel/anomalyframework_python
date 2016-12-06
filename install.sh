# break on failures
set -e

# cd to the root directory
USER_DIR=$(pwd)
SCRIPT=$(readlink -f "$0")
echo $SCRIPT
ROOT_DIR=$(dirname "$SCRIPT")


# Install third-party packages
cd ${ROOT_DIR}/src/external/attrdict-2.0.0/
python setup.py install --user

# Compile C++ source
mkdir -p ${ROOT_DIR}/build 
cd ${ROOT_DIR}/build
cmake ../
make

# Run tests
cd ${ROOT_DIR}
pip install nose
nosetests

# Return to directory
cd ${USER_DIR}
