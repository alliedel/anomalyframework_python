import os
from src.local_pyutils import AttrDict


class ParameterStructure(AttrDict):

    def __str__(self):  # for use with the print function: print x now works if x is this type.
        my_str = '{\n'
        for k in self.__class__.__dict__.keys():
            if not k.startswith("__"):
                my_str += k + ': ' + getattr(self, k).__str__() + '\n'
        my_str += '}'
        return my_str


class Files(ParameterStructure):
    shufflenames_libsvm = ''
    shuffle_idxs = ''
    runinfo_fnames = ''
    done_files = ''
    verbose_fnames = ''


class Folders(ParameterStructure):
    path_to_tmp = ''
    path_to_results = ''
    predict_directories = ''


class AlgorithmPars(ParameterStructure):
    n_shuffles = 10
    window_size = 100
    window_stride = 50
    average_over_splits = 'mean'
    shuffle_size = 1
    lambd = 0.1
    alpha = 1e-30


class Paths(ParameterStructure):
    files = Files()
    folders = Folders()


class SystemPars(ParameterStructure):
    anomalyframework_root = os.path.abspath('./')
    path_to_trainpredict_relative = 'build/src/cpp/score_shuffle'


class Pars(ParameterStructure):
    paths = Paths()
    algorithm = AlgorithmPars()
    system = SystemPars()


class Tags(ParameterStructure):
    datestring = ''
    timestring = ''
    processId = ''
    results_name = ''

