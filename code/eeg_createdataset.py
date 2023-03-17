import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import scipy
from mne.report import Report
from mne.preprocessing import ICA, create_eog_epochs
import matplotlib.pyplot as plt
from autoreject import Ransac
from scipy import signal, stats
from scipy.signal import savgol_filter
from autoreject import AutoReject
import multiprocessing
from tqdm import tqdm
import coffeine
from joblib import Parallel, delayed

njobs = multiprocessing.cpu_count()
print(njobs)

###############################
# Parameters
###############################
ccanada = 1
if ccanada:
    bidsout = '/lustre04/scratch/mpcoll/2023_embcp'
else:
    bidsout = '/Users/mp/data/2023_embcp'
derivpath = opj(bidsout, 'derivatives')


part = [s for s in os.listdir(derivpath) if 'sub' in s]

all_epochs = []
for task in tqdm(['auditory', 'auditoryrate', 'thermal', 'thermalrate', 'rest']):
    # Loop bands
    for p in part:

        raw = mne.io.read_raw_fif(opj(derivpath, p,  'eeg', p
                                      + "_task-painaudio_" + task + "_start"
                                        + "_cleanedeeg-raw.fif"), preload=True)

        epochs = mne.make_fixed_length_epochs(
            raw, duration=4, overlap=0).load_data()

        # Drop bad epochs
        ar = AutoReject(n_interpolate=[0], n_jobs=njobs, random_state=42)
        epochs_clean = ar.fit_transform(epochs)

        # Get the average rating for each epoch
        intensity = [np.mean(e[-1, :]) for e in epochs_clean]
        rating = [np.mean(e[-2, :]) for e in epochs_clean]

        # Get the average rating difference in each epoch
        diff_rate = [np.max(e[-2, :])-np.min(e[-2, :]) for e in epochs_clean]
        diff_stim = [np.max(e[-1, :])-np.min(e[-1, :]) for e in epochs_clean]

        # Drop non eeg channels
        epochs_clean.pick_types(eeg=True, misc=False)

        # Add metadata
        meta_data = pd.DataFrame({'participant_id': p,
                                  'task': task,
                                  'rating': rating,
                                  'epoch_num': np.arange(len(epochs_clean)),
                                  'intensity': intensity,
                                  'diff_rate': diff_rate,
                                  'diff_intensity': diff_stim,
                                  'reject_prop': 1-(len(epochs_clean)/len(epochs))})
        epochs_clean.metadata = meta_data

        # Add to list
        all_epochs.append(epochs_clean)


# Save all epochs
all_epochs = mne.concatenate_epochs(all_epochs)

# Calculate covariances
frequency_bands = {
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 30.0),
    "beta_high": (30.0, 45.0),
    "gamma_low": (45.0, 65.0),
    "gamma_high": (65.0, 100.0)}


def compute_covs_epoch(epoch, frequency_bands, baseline=(None, None)):
    if baseline:
        epoch = epoch.apply_baseline(baseline)
    mne.set_log_level('WARNING')

    epoch_cov = coffeine.power_features._compute_covs_epochs(
        epoch.apply_baseline((None, None)), frequency_bands)
    return np.expand_dims(epoch_cov, 0)


covs = Parallel(n_jobs=njobs,
                verbose=0)(delayed(compute_covs_epoch)(epoch=all_epochs[i],
                                                       frequency_bands=frequency_bands)
                           for i in tqdm(range(len(all_epochs))))

covs = np.vstack(covs)
np.save(opj(derivpath, 'all_epochs-covs.npy'), covs)

# Downsample and save epochs
all_epochs.resample(250)
all_epochs.save(opj(derivpath, 'all_epochs-epo.fif'), overwrite=True)
