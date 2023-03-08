import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from mne.stats import ttest_1samp_no_p
from mne.time_frequency import read_tfrs
import scipy
from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
from mne.report import Report
from mne.preprocessing import ICA, create_eog_epochs
import matplotlib.pyplot as plt
from functools import partial
from oct2py import octave
from tqdm import tqdm

# Get all points marked for rejection
from mne.annotations import _annotations_starts_stops

# %matplotlib inline

from mne.channels import read_custom_montage
from skimage.measure import block_reduce

####################################################
# Helper function (from Bashivan et al.) https://github.com/pbashivan/EEGLearn
####################################################

from sklearn.preprocessing import scale
from scipy.interpolate import griddata

def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])

    n_samples = features.shape[0]

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

###############################
# Parameters
###############################

# Load Acticap location file. Note that
montage = read_custom_montage('CACS-64_NO_REF.bvef')



inpath = "/media/mp/MP5TB/2020_embcp/deep_derivatives/visual_clean"
outpath_3d = "/media/mp/MP5TB/2020_embcp/bashivan_deep/test1/3dspacespacefreq"
if not os.path.exists(outpath_3d):
    os.makedirs(outpath_3d)
outpath_4d = "/media/mp/MP5TB/2020_embcp/bashivan_deep/test1/4dspacespacefreqtime"
if not os.path.exists(outpath_4d):
    os.makedirs(outpath_4d)


# Get files
files = sorted([s for s in os.listdir(inpath) if '-epo.fif' in s])



freqs = {'alpha1': [7, 10],
         'alpha2': [11, 14],
         'beta1': [15, 20],
         'beta2': [21, 30],
         'gamma1': [30, 60],
         'gamma2': [61, 100]}

# Loop participants
metadata = pd.DataFrame()
for f in tqdm(files):
    # Read data
    epochs = mne.read_epochs(opj(inpath, f))

    # Drop EOG
    epochs = epochs.drop_channels(['HEOG', 'VEOG'])

    # Average reref
    epochs = epochs.set_eeg_reference('average')

    # Interpolate bads
    epochs = epochs.interpolate_bads()
    # Set montage
    epochs = epochs.set_montage(montage)

    # Get electrodes 2d coordinates
    xypos = np.array(list(epochs.get_montage()._get_ch_pos().values()))[:, 0:2]


    # TODO Replace this with hilbert to get freq bands
    # Alpha1, alpha2, beta1, beta2, gamma1, gamma2
    # Get the TF representation

    # Loop the frequencies
    for freq in freqs:
        epochs_tf = mne.time_frequency.tfr_morlet(epochs,
                                                freqs=freqs,
                                                n_cycles = freqs / 2,
                                                n_jobs=-2,
                                                return_itc=False,
                                                average=False,
                                                decim=16)


    # Reduce time dim by averaging over X points
    arr_reduced = block_reduce(dat, block_size=(20,1), func=np.mean)

    # SAve individual epochs for easy handling in torch
    for idx, ep in enumerate(epochs_tf):


        ###############################
        # Save as 3d - spacexspacexfreq
        ###############################
        fname = f.replace('-epo.fif',  '-trial_' + str(idx+1) + '_3dssf.npy')
        fname = fname.replace('_task-painaudio', '').replace('_visualclean', '')

        # Average time to get chan x freq
        dat = np.average(np.squeeze(epochs_tf[idx].data), 2)
        # Need features to be (n_features, n_channels)
        dat = np.swapaxes(dat, 0, 1)
        # Make it an image
        trial_img = np.squeeze(gen_images(locs=xypos, features=dat,
                                          n_gridpoints=50, edgeless=False))
        # Make it space x space x freq
        trial_img = np.swapaxes(trial_img, 0, 2)
        outpd = epochs[idx].metadata.copy()
        outpd['filename3d'] = fname
        outpd['filename4d'] = fname.replace('3dssf', '4dssft')

        np.save(opj(outpath_3d, fname), trial_img)

        metadata = metadata.append(outpd)

        ###############################
        # Save as 4d - spacexspacexfreqxtime
        ###############################
        # Todo get meaningful band
        # Loop time
        # epoch_dat = []
        # for t in range(epochs_tf[idx].data.shape[3]):
        #     # Get freq data for this time point
        #     dat = np.squeeze(epochs_tf[idx].data)[:, :, t]
        #     # Need features to be (n_features, n_channels)
        #     dat = np.swapaxes(dat, 0, 1)
        #     # Make it an image
        #     trial_img = np.squeeze(gen_images(locs=xypos, features=dat,
        #                                       n_gridpoints=32, edgeless=False))
        #     # Make it space x space x freq
        #     trial_img = np.swapaxes(trial_img, 0, 2)
        #     # Append to list
        #     epoch_dat.append(trial_img)
        # # Get a space x space x freq x time array
        # epoch_dat_test = np.moveaxis(np.stack(epoch_dat), 0, -1)
        # np.save(opj(outpath_4d, fname.replace('3dssf', '4dssft')),
        #         epoch_dat_test)

    metadata.reset_index(drop=True).to_csv(opj(outpath_3d, 'metadata.csv'),
                                           index=False)

    metadata.reset_index(drop=True).to_csv(opj(outpath_4d, 'metadata.csv'),
                                           index=False)
