from src import filenames
import local_pyutils

# TODO(allie): generate a static class from this file (to enable autocomplete)

default_pars = local_pyutils.dotdictify(dict(
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
            shuffle_size=1
        ),
        discriminability=dict(
            lambd=0.1,
            alpha=1e-30,
            solver_num=0,
            window_size=100,
            window_stride_multiplier=0.5,
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


class Pars(local_pyutils.dotdictify):
    def __init__(self, **kwargs):
        for key in default_pars:
            self.__setitem__(key, default_pars[key])
        self.set_values(**kwargs)
        filenames.fill_tags_and_paths(self)

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            local_pyutils.replace_in_nested_dictionary(self, key, value)
