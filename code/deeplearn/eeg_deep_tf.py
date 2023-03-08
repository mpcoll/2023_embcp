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

###############################
# Parameters
###############################


inpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives/visual_clean"
outpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives/timefreq"
if not os.path.exists(outpath):
    os.mkdir(outpath)


outpath_strials = "/media/mp/lxhdd/2020_embcp/deep_derivatives/timefreq/strials"
if not os.path.exists(outpath_strials):
    os.mkdir(outpath_strials)

all_epochs = []
files = sorted([s for s in os.listdir(inpath) if '-epo.fif' in s])

ch_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz',
            'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2']



# Loop participants
metadata = pd.DataFrame()
for f in tqdm(files):
    epochs = mne.read_epochs(opj(inpath, f))

    epochs = epochs.set_eeg_reference('average')

    epochs = epochs.interpolate_bads()
    freqs = np.arange(4, 101, 1)
    epochs_tf = mne.time_frequency.tfr_morlet(epochs,
                                              freqs=np.arange(4, 101, 1),
                                              n_cycles = freqs / 2,
                                              n_jobs=-2,
                                              return_itc=False,
                                              average=False,
                                              decim=4)


    # SAve individual epochs for easy handling in torch
    for idx, ep in enumerate(epochs_tf):
        fname = f.replace('-epo.fif',  '-trial_' + str(idx+1) + '-tfr.npy')
        fname = fname.replace('_task-painaudio', '').replace('_visualclean', '')
        outpd = epochs_tf[idx].metadata.copy()
        epochs_tf.reorder_channels()
        outpd['filename'] = fname




        np.save(opj(outpath_strials, fname), epochs_tf[idx].data)

        metadata = metadata.append(outpd)


    metadata.reset_index(drop=True).to_csv(opj(outpath_strials, 'metadata.csv'),
                                           index=False)

