# Python implementation of the Anomaly Detection Framework
A cleaner implementation of the Del Giorno, Bagnell, Hebert ECCV 2016 paper 'A Discriminative Framework for Anomaly 
Detection in Large Videos'

## Building (C++ code)
Rather than cluttering the source directory with build files, run the following:
cmake -Bbuild -H.; cd build/; make; cd ../
# or with debug flags: 
cmake -DCMAKE_BUILD_TYPE=Debug -Bbuild -H.; cd build/; make; cd ../

This will create a build in the build/ folder.

### Using python virtualenv
Given that you'll need a set of Python packages to run, the best method to run this software is to install virtualenv, to mimic the same python system I developed in.  Here are the steps:

Install virtualenv
`pip install virtualenv`

Create the 'vanilla' virtual env
`virtualenv anomalyframework_virtualenv`

Activate the virtual environment
`bash anomalyframework_virtualenv/bin/activate`

Install the packages from requirements.txt
`pip install -r requirements.txt`

When running code from this repository, you can then either activate the virtual environment each time (and use `python`, `ipython` as normal), or execute `anomalyframework_virtualenv/bin/python` to circumvent the activation, and call the executables directly.

## Testing
run:
`python unit_tests/example_runscript.py`
