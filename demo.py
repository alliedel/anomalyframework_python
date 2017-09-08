import os
import matplotlib.pyplot as plt
import subprocess

import local_pyutils
from anomalyframework import liblinear_utils
from anomalyframework import run

# Open logging file: stdout
local_pyutils.open_stdout_logger()

# Build project files
subprocess.check_call('cmake -Bbuild -H.', shell=True)
os.chdir('build')
try:
    subprocess.check_output('make')
    os.chdir('../')
except:
    os.chdir('../')
    raise

# Mark this directory as root
os.environ['ANOMALYROOT'] = os.path.abspath(os.path.curdir)
print(os.environ['ANOMALYROOT'])

infile_features = 'data/input/features/Avenue/03_feaPCA_new.train'

infile_features = os.path.abspath(os.path.expanduser(infile_features))
if not os.path.isfile(infile_features):
    raise ValueError('{} does not exist.'.format(infile_features))

# Run anomaly detection
a, pars = run.main(infile_features=infile_features, n_shuffles=10)

import pickle
pickle.dump(dict(a=a, pars=pars), open(infile_features.replace('.train', '_res.pickle'),'wb'))

# Display
if(0):
    X, y = liblinear_utils.read(pars.paths.files.infile_features, False)
    plt.figure(1)
    plt.cla()
    plt.plot(a/(1.0-a))
    plt.figure(2)
    plt.cla()
    plt.plot(a)
    plt.figure(3)
    plt.cla()
    X = X.toarray()
    plt.imshow(X.T)
    plt.title('X')
    plt.show(block=True)
