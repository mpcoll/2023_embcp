import numpy as np
import pandas as pd
import mne
from os.path import join as opj
import os
from tqdm import tqdm
from bids import BIDSLayout
from mne.preprocessing import ICA
from oct2py import octave
from skimage.measure import block_reduce
from mne.channels import read_custom_montage

# %matplotlib inline
# %matplotlib qt5

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
layout = BIDSLayout("/media/mp/lxhdd/2020_embcp/source")
part = sorted(["sub-" + s for s in layout.get_subject()])
bidsout = "/media/mp/lxhdd/2020_embcp/source"
# Raw cleaned files
inpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives/visual_clean"

# Output folder
outpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives/epoched"
if not os.path.exists(outpath):
    os.mkdir(outpath)

outpath_strials = "/media/mp/lxhdd/2020_embcp/bashivan_deep/test4d_larger"
if not os.path.exists(outpath_strials):
    os.mkdir(outpath_strials)



part = list(set([s.split('_')[0] for s in os.listdir(inpath) if 'sub-' in s]))


epochs_duration_s = 3.5

freqs = {
    'theta': [4, 7],
    'alpha': [8, 14],
    'beta1': [15, 30],
    'gamma1': [31, 60],
    'gamma2': [61, 90]}

# Loop participants
for p in tqdm(part):


    # _________________________________________________________________
    # Epoch according to condition to remove breaks and pre/post

    events = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_events.tsv"), sep="\t")

    ratings = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_beh.tsv"), sep="\t")

    for cond in [
        "rest_start",
        "thermal_start",
        "auditory_start",
        "thermalrate_start",
        "auditoryrate_start",]:
        # Keep only start of trial
        events_cond = events[events.condition.isin([cond])]

        file = p + '_task-painaudio_' + cond + '_visualclean-raw.fif'
        # Load raw cleaned file
        raw_cond = mne.io.read_raw_fif(opj(inpath, file))

        for band, cutoffs in freqs.items():
            # Filter hilbert to keep amplitude only in bands of interest
            raw_cond_tf = raw_cond.copy().load_data()

            # bandpass filter
            raw_cond_tf.filter(cutoffs[0], cutoffs[1],  # use more jobs to speed up.
                        l_trans_bandwidth="auto",  # make sure filter params are the same
                        h_trans_bandwidth="auto")  # in each band and skip "auto" option.

            # apply hilbert
            raw_cond_tf.apply_hilbert(envelope=True, n_jobs=-2)


            if cond != "rest_start":
                epochs_cond = []
                for idx, intensity in enumerate([50, 100, 150]):

                    # Find where specific intensity starts
                    stim_start = np.where(
                        (ratings["stim_intensity"] == intensity)
                        & (np.diff(ratings["stim_intensity"], prepend=[0]) != 0)
                        & (np.ediff1d(ratings["stim_intensity"], to_end=[0]) == 0)
                    )[0]

                    # Remove two first seconds to make sure it is stabilised
                    stim_start = stim_start + raw_cond_tf.info["sfreq"] * 2

                    # Find where specific intensity ends
                    stim_end = np.where(
                        (ratings["stim_intensity"] == intensity)
                        & (np.ediff1d(ratings["stim_intensity"], to_end=[0]) != 0)
                        & (np.diff(ratings["stim_intensity"], prepend=[0]) == 0)
                    )[0]
                    # Make sure all ok
                    assert len(stim_start) == len(stim_end)
                    for s_start, s_end in zip(stim_start, stim_end):
                        # Loop start stop with indexes
                        raw_epo = raw_cond_tf.copy().crop(
                            tmin=s_start / raw_cond_tf.info["sfreq"], tmax=s_end / raw_cond_tf.info["sfreq"]
                        )

                        if cond == "thermalrate_start":
                            pain_rating = np.mean(
                                np.asarray(ratings["pain_rating"])[
                                    int(s_start) : int(s_end)
                                ]
                            )
                        else:
                            pain_rating = 999

                        if cond == "auditoryrate_start":
                            audio_rating = np.mean(
                                np.asarray(ratings["audio_rating"])[
                                    int(s_start) : int(s_end)
                                ]
                            )
                        else:
                            audio_rating = 999

                        epochs = mne.make_fixed_length_epochs(
                            raw_epo, duration=epochs_duration_s,
                        ).drop_bad()
                        stophere
                        epochs.metadata = pd.DataFrame(
                            dict(
                                subject_id=[p] * len(epochs),
                                condition=[cond] * len(epochs),
                                stim_intensity=[intensity] * len(epochs),
                                stim_start_s=[s_start / raw_cond_tf.info["sfreq"]] * len(epochs),
                                stim_end_s=[s_end / raw_cond_tf.info["sfreq"]] * len(epochs),
                                pain_rating=[pain_rating] * len(epochs),
                                audio_rating=[audio_rating] * len(epochs),
                                order_in_stim=range(len(epochs)),
                            )
                        )
                        if len(epochs) != 0:
                            epochs_cond.append(epochs)

                epochs_cond = mne.concatenate_epochs(epochs_cond)
                epochs_cond.save(opj(outpath, file.replace('-raw', '_' + band + '-epo')), overwrite=True)

            else:
                epochs = mne.make_fixed_length_epochs(raw_cond_tf, duration=epochs_duration_s).drop_bad()
                epochs.metadata = pd.DataFrame(
                    dict(
                        subject_id=[p] * len(epochs),
                        condition=[cond] * len(epochs),
                        stim_intensity=[0] * len(epochs),
                        stim_start_s=["NA"] * len(epochs),
                        stim_end_s=[["NA"]] * len(epochs),
                        pain_rating=[["NA"]] * len(epochs),
                        audio_rating=[["NA"]] * len(epochs),
                        order_in_stim=range(len(epochs)),
                    )
                )
                epochs.save(opj(outpath, file.replace('-raw', '_' + band + '-epo')), overwrite=True)



montage = read_custom_montage('/home/mp/gdrive/projects/2020_embcp/code/bashivan_deep/CACS-64_NO_REF.bvef')


# Create images for single trials
metadata = pd.DataFrame()
for p in tqdm(part):
    # Loop conditions
    for cond in ["rest_start", "thermal_start", "auditory_start",
                 "thermalrate_start", "auditoryrate_start",]:

        # Load all bands for this condition
        cond_data = []
        for band, _ in freqs.items():
            fname = p + '_task-painaudio_'  + cond + '_visualclean_' + band + '-epo.fif'
            epochs = mne.read_epochs(opj(outpath, fname  ))

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

            cond_data.append(epochs.get_data())

        #Get a band x epochs x channel x time array
        cond_data = np.stack(cond_data)

        for epo in range(cond_data.shape[1]):
            outpd = epochs[epo].metadata.copy()
            stfname = fname.replace('-epo.fif',  'trial_' + str(epo+1) + '_4dfsst.npy')
            stfname = stfname.replace('_task-painaudio', '').replace('_visualclean', '').replace(band, '')

            arr = cond_data[:, epo, ::]
            # Average across time to get 7
            arr_reduced = block_reduce(arr, block_size=(1,1,25), func=np.mean)
            # Generate a n_band x 2d space image for each timebin
            epo_imgs = []
            for timebin in range(arr_reduced.shape[2]):
                dat = arr_reduced[:, :, timebin]

                img = gen_images(locs=xypos, features=dat,
                                            n_gridpoints=50, edgeless=False)
                epo_imgs.append(np.squeeze(img))

            # Bands x space x space x time epoch data
            epo_dat = np.stack(epo_imgs, axis=0)

            outpd['filename'] = stfname

            np.save(opj(outpath_strials, stfname), epo_dat)

            metadata = metadata.append(outpd)

metadata.to_csv(opj(outpath_strials, 'metadata.csv'))
# Load all files concatenate data (bands x channels x epoch x time)


# Loop epochs



# Create images

# Save as numpy
