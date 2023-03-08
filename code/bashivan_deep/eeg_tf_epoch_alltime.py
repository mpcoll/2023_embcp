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
import matplotlib.pyplot as plt
from mne.channels import read_custom_montage

# %matplotlib inline
# %matplotlib qt5

# Goal here:
    # TF the whole epoch

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

outpath_strials = "/media/mp/lxhdd/2020_embcp/bashivan_deep/3dssf_larger_cleaner"
if not os.path.exists(outpath_strials):
    os.mkdir(outpath_strials)

part = list(set([s.split('_')[0] for s in os.listdir(inpath) if 'sub-' in s]))

montage = read_custom_montage('/home/mp/gdrive/projects/2020_embcp/code/bashivan_deep/CACS-64_NO_REF.bvef')


# Loop participants
metadata = pd.DataFrame()
for p in tqdm(part):

    events = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_events.tsv"), sep="\t")


    for cond in [
        "thermal_start",
        "rest_start",
        "auditory_start",
        "thermalrate_start",
        "auditoryrate_start",]:
        ratings = pd.read_csv(
            opj(bidsout, p, "eeg", p + "_task-painaudio_beh.tsv"), sep="\t")

        # Keep only start of trial
        events_cond = events[events.condition.isin([cond])]

        file = p + '_task-painaudio_' + cond + '_visualclean-raw.fif'
        # Load raw cleaned file
        raw_cond = mne.io.read_raw_fif(opj(inpath, file)).load_data()

        raw_cond = raw_cond.drop_channels(['HEOG', 'VEOG'])

        # Average reref
        raw_cond = raw_cond.set_eeg_reference('average')
        # Interpolate bads
        raw_cond = raw_cond.interpolate_bads()

        raw_cond = raw_cond.set_montage(montage)
        xypos = np.array(list(raw_cond.get_montage()._get_ch_pos().values()))[:, 0:2]

        # Add ratings as channels
        # Adjust slight diff
        ratings = ratings[:len(raw_cond)]

        info = mne.create_info(['prate', 'arate', 'intensity'],
                               raw_cond.info['sfreq'], 'misc')
        rate_dat = np.asarray(ratings[['pain_rating', 'audio_rating',
                                       'stim_intensity']]).transpose()
        rate_raw = mne.io.RawArray(rate_dat, info)
        raw_cond.add_channels([rate_raw], force_update_info=True)

        # add to array to resample in the same way
        # raw_cond = raw_cond.resample(250)
        freqs = np.arange(4, 101, 1)

        markers = mne.make_fixed_length_events(raw_cond, duration=4,
                                             overlap=2, id=99999)
        epoch = mne.Epochs(raw_cond, markers, event_id=99999,
                           tmin=-2, tmax=2,
                           reject_by_annotation=True,
                           reject=dict(eeg=200e-6),
                           baseline=None,
                           # Do not look in overlap
                           reject_tmin=-1,
                           reject_tmax=1)

        epoch.drop_bad()

        # Get average ratubg an intensity in each epoch
        # Crop the overlap
        ecrop = epoch.copy().load_data().crop(tmin=-1, tmax=1)
        prating, arating, intensity = [] , [], []
        for idx in range(len(ecrop)):
            prating.append(np.mean(ecrop.get_data(picks='prate')[idx, ::]))
            arating.append(np.mean(ecrop.get_data(picks='arate')[idx, ::]))
            intensity.append(np.mean(ecrop.get_data(picks='intensity')[idx, ::]))


        epochs_tf = mne.time_frequency.tfr_morlet(epoch,
                                                freqs=freqs,
                                                n_cycles = 7,
                                                n_jobs=-2,
                                                use_fft=True,
                                                return_itc=False,
                                                average=False,
                                                decim=8)
        # Crop the overlap
        epochs_tf = epochs_tf.crop(tmin=-1, tmax=1)

        # Create the metadata
        if cond == 'rest_start':
            prating, arating, intensity = 'nan', 'nan', 'nan'
        if cond == 'thermalrate_start':
            arating = 'nan'
        if cond == 'thermal_start':
            arating, prating = 'nan', 'nan'
        if cond == 'auditory_start':
            arating, prating = 'nan', 'nan'
        if cond == 'auditoryrate_start':
            prating = 'nan'

        epochs_tf.metadata = pd.DataFrame(data={'subject_id': p,
                                                'condition': cond,
                                                'pain_rating': prating,
                                                'audio_rating': arating,
                                                'stim_intensity': intensity,
                                                'epoch_num': np.arange(1,
                                                                       len(epochs_tf)+1)})

        cond_data = epochs_tf.data
        epo_imgs = []
        for epo in range(cond_data.shape[0]):
            outpd = epochs_tf[epo].metadata.copy()
            stfname = file.replace('-raw.fif',  '_trial_' + str(epo+1) + '_3dssf.npy')
            stfname = stfname.replace('_task-painaudio', '').replace('_visualclean', '')

            arr = cond_data[epo, ::]

            # Average across time
            arr = np.mean(arr, axis=2)
            # Generate a n_band x 2d space image for each timebin

            img = gen_images(locs=xypos, features=np.swapaxes(arr, 1, 0),
                                        n_gridpoints=50, edgeless=False)
            outpd['filename'] = stfname

            np.save(opj(outpath_strials, stfname), img)

            metadata = metadata.append(outpd)

metadata.to_csv(opj(outpath_strials, 'metadata.csv'))
# Load all files concatenate data (bands x channels x epoch x time)


# Loop epochs



# Create images

# Save as numpy
