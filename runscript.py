import glob
import numpy as np

import logging
import os
import pickle
from src import run
from python import generalized_eigenvector, parameters
import local_pyutils
from src import liblinear_utils
import sklearn.decomposition


if __name__ == '__main__':
    local_pyutils.open_stdout_logger()
    k_vec = [100]
    lambd = 0.1
    n_shuffles = 10
    videonums = [1]
    for videonum in videonums:  # Avenue-specific
        for k in k_vec:
            for transformation_type in ['pca']:
                # Load parameters
                focus_params = generalized_eigenvector.FocusPars(
                    k=k, transformation_type=transformation_type)
                focus_hash = generalized_eigenvector.get_focus_params_hash_string(focus_params)
                dataset_name = 'fall'
                transformation_cache_dir = '../focus/data/cache/{}/transformations/'.format(
                    dataset_name)
                feature_pars = parameters.Pars(bin_size=[40, 40, 1], feature_patch_size=[40, 40, 5])
                feature_params_hash = parameters.get_raw_features_hash_string(feature_pars)
                training_files_glob = '../focus/data/cache/{}/' \
                                      '*_raw._{}.npy'.format(dataset_name, feature_params_hash)
                test_file_raw = '../focus/data/cache/{}/{:0>2}_raw._{}.npy'.format(
                    dataset_name, videonum, feature_params_hash)

                # Learn transformation
                all_feature_files_list = sorted(glob.glob(training_files_glob))
                my_feature_file = np.array([file.find('/{:0>2}'.format(videonum)) != -1 for file in
                                            all_feature_files_list])
                assert np.sum(my_feature_file) == 1, SystemError(
                    'My video number not found. Looking for:\n {}'.format(
                        training_files_glob))
                # # training_files_list = [all_feature_files_list[i] for i in
                # #                        np.where(~my_feature_file)[0]]
                # training_files_list = all_feature_files_list
                # transformation = generalized_eigenvector.learn_transformation(
                #     training_files_list, focus_params)
                #
                # # Save transformation, pars, and list of input features
                # transformation_outfile, params_outfile, infiles_list_outfile = [
                #     os.path.join(transformation_cache_dir, basename) for basename in
                #     generalized_eigenvector.generate_transformation_and_params_outnames(
                #         focus_params=focus_params, infiles_hash='all_avenue_test_normal_but_me')]
                # local_pyutils.mkdir_if_needed(transformation_cache_dir)
                # pickle.dump(transformation, open(transformation_outfile, 'w'))
                # pickle.dump(focus_params, open(params_outfile, 'w'))
                # np.save(infiles_list_outfile, training_files_list)
                # logging.info('transformation saved in {}'.format(transformation_outfile))
                #
                # # Apply transformation to test set
                # testing_outfile_dir = os.path.dirname(test_file_raw) + '/{}'.format(
                #     focus_hash)
                # local_pyutils.mkdir_if_needed(testing_outfile_dir)
                # testing_outfile = testing_outfile_dir + '/' + os.path.basename(test_file_raw)
                # X = np.load(test_file_raw)
                # X_transformed = transformation.transform(X)
                # np.save(testing_outfile, X_transformed)

                pca = sklearn.decomposition.PCA(n_components=50)
                infile_features = test_file_raw
                X = np.load(test_file_raw)
                X_transformed = pca.fit_transform(X)
                coeff, score, latent = generalized_eigenvector.princomp(X_transformed)
                infile_features_libsvm = infile_features.replace('.npy', '.train')
                print('Creating the .train file for {}'.format(infile_features))
                liblinear_utils.write(X_transformed, y=None, outfile=infile_features_libsvm,
                                      zero_based=True)

                # Run anomaly detection
                print('Running anomaly detection')

                a, pars = run.main(infile_features=infile_features_libsvm, n_shuffles=n_shuffles,
                                   lambd=lambd)


# features_dir = '/home/allie/projects/focus/data/cache/Avenue/{}'.format(focus_hash)
# infile_features_list = sorted(glob.glob('{}/*.npy'.format(features_dir)))
# if len(infile_features_list) == 0:
#     raise Warning('No feature files in {}'.format(features_dir))
# if videonums:
#     infile_features_list = infile_features_list[np.array(videonums) - 1]
# for infile_features in infile_features_list:
#     assert os.path.isfile(infile_features), ValueError(infile_features +
#                                                        ' doesn\'t exist')
#
