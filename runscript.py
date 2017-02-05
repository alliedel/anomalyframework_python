import os
from src import run


if __name__ == '__main__':

    for videonum in [i + 1 for i in range(21)]:
        for lambd in [0.01, 0.1, 1]:
            infile_features = os.path.abspath(os.path.expanduser(
                '/home/allie/projects/focus/data/cache/Avenue/{:0>2}.avi_raw'
                '._feature_patch_sz__10x10x5_.image_dtype__float32_.search_patch_sz__10x10x5_.'
                'frame_sz_in_blocks___.image_resz__180x320_.mt__5_.'
                'search_patch_stride__10x10x5_.pca_dim__100_.pts_per_vol__3000.train'
                ''.format(videonum)))

            assert os.path.isfile(infile_features)

            # Run anomaly detection
            a, pars = run.main(infile_features=infile_features, n_shuffles=10, lambd=lambd)
