import numpy as np
import os

from src import liblinear_utils
from src import run


if __name__ == '__main__':

    for videonum in [i + 1 for i in range(21)]:
        for lambd in [0.001, 0.01, 1]:
            whiten_string = '_self_whiten'
            infile_features = os.path.abspath(os.path.expanduser(
                '/home/allie/projects/focus/data/cache/Avenue/{:0>2}_raw'
                '._feature_patch_sz__10x10x5_.image_dtype__float32_.search_patch_sz__10x10x1_.frame'
                '_sz_in_blocks___.bin_sz__10x10x1_.image_resz__120x160_.mt__0.01_.pca'
                '_dim__100_.pts_per_vol__3000.npy'
                ''.format(videonum)))
            if whiten_string == '_self_whiten':
                print('Whitening...')
                X = np.load(infile_features)
                infile_features = infile_features.replace('.npy', '{}.npy'.format(whiten_string))
                import sklearn.decomposition
                pca = sklearn.decomposition.PCA(whiten=True, tol=1e-4)
                pca.fit(X)
                Xw = pca.transform(X)
                # Scale so eigenvalues are 1
                Xw = Xw.dot(np.diag(1/np.sqrt(np.diagonal(Xw.T.dot(Xw)))))
                Xw.T.dot(Xw)
                np.save(infile_features, Xw)

            assert os.path.isfile(infile_features), ValueError(infile_features +
                                                                   ' doesn\'t exist')
            infile_features_libsvm = infile_features.replace('.npy', '.train')
            if not os.path.isfile(infile_features_libsvm):
                print('Creating the .train file for {}'.format(infile_features))
                X = np.load(infile_features)
                liblinear_utils.write(X, y=None, outfile=infile_features_libsvm, zero_based=True)

            # Run anomaly detection
            a, pars = run.main(infile_features=infile_features_libsvm, n_shuffles=10, lambd=lambd)
