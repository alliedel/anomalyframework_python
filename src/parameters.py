from src.local_pyutils import dotdictify

default_pars = dotdictify(dict(
    paths=dict(
        files=dict(
            infile_features='',
            shufflenames_libsvm='',
            shuffle_idxs='',
            runinfo_fnames='',
            done_files='',
            verbose_fnames=''
        ),
        folders=dict(
            path_to_tmp='',
            path_to_results='',
            predict_directories=''
        )
    ),
    algorithm=dict(
        permutations=dict(
            n_shuffles=10,
            window_size=100,
            window_stride=50,
            shuffle_size=1
        ),
        discriminability=dict(
            lambd=0.2,
            alpha=1e-30,
            solver_num=0
            ),
        aggregation=dict(
            average_over_splits='mean'
        )
    ),
    system=dict(
        anomalyframework_root='./',
        path_to_trainpredict_relative='build/src/cpp/score_shuffle',
        num_threads=8
    ),
    tags=dict(
        datestring='',
        timestring='',
        processId='',
        results_name=''
    )
))


class Pars(dotdictify):
    def __init__(self, infile_features=default_pars.paths.files.infile_features):
        for key in default_pars:
            self.__setitem__(key, default_pars[key])
        self.paths.files.infile_features = infile_features

