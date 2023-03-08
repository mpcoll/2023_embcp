from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mne.time_frequency import tfr_morlet
from bids import BIDSLayout
from mne.datasets import fetch_fsaverage
import bioread
from scipy.stats import linregress, zscore, ttest_1samp
from mne.stats import fdr_correction
from sklearn.linear_model import HuberRegressor, RANSACRegressor,  TheilSenRegressor
from tqdm import tqdm
from autoreject import AutoReject
from autoreject.autoreject import _apply_interp
from scipy.signal import hilbert

###############################
# Parameters
###############################

source_path = '/media/mp/lxhdd/2020_embcp/source'
layout = BIDSLayout(source_path)
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives2'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/source_analysis_v4'
# use robust regression
use_robust = True
huber = HuberRegressor()
ransac = RANSACRegressor(max_trials=10000000, random_state=7)
theil = TheilSenRegressor()

if not os.path.exists(outpath):
    os.mkdir(outpath)

part = ['sub-' + s for s in layout.get_subject()]
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

# Load data
bands = {'alpha': [8, 13, 2],
         'beta': [14, 29, 4],
         'gamma': [30, 100, 7],
         'gamma2': [45, 100, 7],
         'theta': [4, 7, 2]}


epo_types = ['rest_start',
            'thermal_start',
            'auditory_start',
            'thermalrate_start',
            'auditoryrate_start']


###################################################################
# Make forward solution (same for all participants)
###################################################################
# Mni template
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# src = opj(fs_dir, 'bem', 'fsaverage-ico-4-src.fif')
bem = opj(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
os.listdir(opj(fs_dir, 'bem'))
# src = opj(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')

# # Volume source space with 1 cm spacing
# src = mne.setup_volume_source_space('fsaverage', 10, bem=bem,
#                                     subjects_dir=subjects_dir, verbose=True)
# _________________________________________________________________
# Build forward solution
# # Surface source space
if not os.path.exists(opj(outpath, 'forward_solution-fwd.fif')):
    # 10 mm grid on surface source space
    # src = mne.setup_source_space('fsaverage', 'oct5',
    #                             subjects_dir=subjects_dir, verbose=True)

    # 10 mm grid volume source space
    src = mne.setup_volume_source_space('fsaverage', 10.0,
                                        bem=bem,
                                        mri='T1.mgz',
                                        subjects_dir=subjects_dir,
                                        verbose=True)

    os.listdir('/home/mp/mne_data/MNE-fsaverage-data/fsaverage/mri/')
    # src.save(opj(outpath, 'source_space-src.fif'))
    # Load data from first part
    p = 'sub-003'
    pdir = opj(derivpath, p, 'eeg')
    raw = mne.io.read_raw_fif(opj(pdir, p
                                + '_task-painaudio_rest_start_cleanedeeg_raw.fif'),
                            preload=True)

    # Plot the alignement
    fig = mne.viz.plot_alignment(
            raw.info, src=src, eeg=['original', 'projected'], trans=trans,
            show_axes=True, mri_fiducials=True, dig='fiducials')

    # Compute the foward solution
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                    bem=bem, eeg=True, n_jobs=1)
    mne.write_forward_solution(opj(outpath, 'forward_solution-fwd.fif'), fwd,
                               overwrite=True)
else:
    fwd = mne.read_forward_solution(opj(outpath, 'forward_solution-fwd.fif'))



# Helper smoothing function
def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
###################################################################
# Filter and save different bands of interest
###################################################################
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027', 'sub-008', 'sub-031']]


epo_types = [
            'thermal_start',
            'auditory_start',
            'thermalrate_start',
            'auditoryrate_start']




def run_all(bname, bedge, epo_type):
    betas_rating_all = []
    betas_stim_all = []
    betas_rating_all_rob = []
    betas_stim_all_rob = []
    betas_stim_all_rob2 = []
    betas_rating_all_rob2 = []
    if 'thermal' in epo_type:
        rate_col = 'pain_rate'
    elif 'auditory' in epo_type:
        rate_col = 'audio_rate'
    reject_dict = pd.DataFrame(columns=[epo_type],
                               index=part)
    for p in part:

        print(p)
        pdir = opj(derivpath, p, 'eeg')

        # _________________________________________________________________
        # Load and filter data

        # Use rest condition as baseline matrix
        # base_raw = mne.io.read_raw_fif(opj(pdir, p
        #                 + '_task-painaudio_'
        #                 + 'rest_start' + '_cleanedeeg_raw.fif'),
        #             preload=True)


        # Load cond file
        raw = mne.io.read_raw_fif(opj(pdir, p
                                        + '_task-painaudio_'
                                        + epo_type + '_cleanedeeg_raw.fif'),
                                    preload=True)

        # Use AutoReject on epoched
        # ar = AutoReject(n_interpolate=np.array([1, 2, 3, 4]),
        #          verbose=False)
        epo = mne.make_fixed_length_epochs(raw, 0.2, preload=True)
        n_orig = len(epo)
        # ar.fit(epo)
        # epo_test = epo.copy()
        # _apply_interp(rej_log, epo, ar.threshes_, ar.picks_, None, ar.verbose)
        epo.drop_bad(reject=dict(eeg=100e-6))
        # epo.drop_bad(reject=dict(eeg=200e-6))
        bad_epochs = epo.drop_log
        reject_dict.loc[p, epo_type] = (n_orig - len(epo))/n_orig
        # reject_log = ar.get_reject_log(epo)

        # 1. bandpass filter the  full data
        # base_raw.filter(bedge[0], bedge[1])


        raw.filter(bedge[0], bedge[1],
                l_trans_bandwidth=bedge[2],
                h_trans_bandwidth=bedge[2],
               fir_design='firwin')

        # Apply autoreject bad segments
        epo = mne.make_fixed_length_epochs(raw, 0.2, preload=True)
        # nall = len(epo)
        # _apply_interp(rej_log, epo, ar.threshes_, ar.picks_, None, ar.verbose)

        epo.drop(np.where(bad_epochs)[0])
        # # epo.drop_bad(reject=dict(egg=200e-6))

        raw = mne.io.RawArray(np.concatenate(epo.get_data(), axis=1),
                           raw.info)


        # # # 2. compute the covariance matrices
        # baseline_cov = mne.compute_raw_covariance(base_raw, tmin=0,
        #                                           tstep=0.2,
        #                                           tmax=None,
        #                                           reject=dict(eeg=100e-6))


        active_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None,
                                                tstep=0.2,
                                                reject=dict(eeg=100e-6))


        filters = mne.beamformer.make_lcmv(raw.info, fwd,
                                            data_cov=active_cov,
                                            reduce_rank=True,
                                            noise_cov=None,
                                            pick_ori='max-power',
                                            weight_norm='unit-noise-gain',
                                            reg=0.05)

        # 5. use spatial filter on hilbert data
        stcs = mne.beamformer.apply_lcmv_raw(raw, filters)


        # Apply hilbert
        hdat = np.zeros(stcs.data.shape)
        for ii in range(stcs.data.shape[0]):
            hdat[ii, :] = np.abs(hilbert(stcs.data[ii, :]))

        vertices = stcs.vertices
        sub = stcs.subject
        del stcs
        # Smooth and downsample
        n = int(raw.info['sfreq'])
        window = (1.0 / n) * np.ones(n,)
        jump = int(n/10)
        dat = np.zeros((hdat.shape[0], int(hdat.shape[1]/jump)))
        for ii in range(dat.shape[0]):
            # Smooth with 1 s window moving by 0.1
            dat[ii, :] = np.convolve(hdat[ii, ], window, mode='same')[::jump]

        del hdat


        # _________________________________________________________________
        # header2
        # Load ratingsm donwsample and smooth
        ratings = np.squeeze(raw.copy().pick_channels([rate_col]).get_data())
        stim_int = np.squeeze(raw.copy().pick_channels(['stim_int']).get_data())


        smht_ratings = zscore(np.convolve(ratings, window, mode='same')[::jump])
        smht_stim = zscore(np.convolve(stim_int, window, mode='same')[::jump])


        # Initialize betas arrays


        # Calculate regression between behav and eeg
        # print(bname)
        # print(epo_type)
        # print(p)
        betas_rating = np.zeros(dat.shape[0])
        betas_stim = np.zeros(dat.shape[0])
        # Add dim for sklearn

        betas_rating_rob = np.zeros(dat.shape[0])
        betas_stim_rob = np.zeros(dat.shape[0])
        betas_rating_rob2 = np.zeros(dat.shape[0])
        betas_stim_rob2 = np.zeros(dat.shape[0])
        for vertex in tqdm(range(dat.data.shape[0])):
            eeg_vertex = zscore(dat[vertex, :])
            # eeg_vertex = np.expand_dims(eeg_vertex, 1)
            # Get beta value using regression
            # if not use_robust:
            betas_rating[vertex] = linregress(np.squeeze(eeg_vertex), smht_ratings)[0]
            betas_stim[vertex] = linregress(np.squeeze(eeg_vertex), smht_stim)[0]

            # betas_rating_rob[vertex] = ransac.fit(np.expand_dims(smht_ratings, 1), eeg_vertex).estimator_.coef_
            # betas_stim_rob[vertex] = ransac.fit(np.expand_dims(smht_stim, 1), eeg_vertex).estimator_.coef_
            try:
                betas_rating_rob2[vertex] = huber.fit(np.expand_dims(smht_ratings, 1), eeg_vertex).coef_
                betas_stim_rob2[vertex] = huber.fit(np.expand_dims(smht_stim, 1), eeg_vertex).coef_
            except:
                pass

        betas_rating_all.append(betas_rating)
        betas_stim_all.append(betas_stim)
        # betas_rating_all_rob.append(betas_rating_rob)
        # betas_stim_all_rob.append(betas_stim_rob)
        betas_rating_all_rob2.append(betas_rating_rob2)
        betas_stim_all_rob2.append(betas_stim_rob2)
    # Save as a array and source estimate with subjects insteadd of time points
    betas_stim_all_np = np.stack(betas_stim_all)
    betas_rating_all_np = np.stack(betas_rating_all)
    # betas_stim_all_robnp = np.stack(betas_stim_all_rob)
    # betas_rating_all_robnp = np.stack(betas_rating_all_rob)
    betas_stim_all_rob2np = np.stack(betas_stim_all_rob2)
    betas_rating_all_rob2np = np.stack(betas_rating_all_rob2)
    np.save(opj(outpath, bname + '_' + epo_type + '_betas_stim.npy'),
            betas_stim_all_np)
    np.save(opj(outpath, bname + '_' + epo_type + '_betas_rate.npy'),
            betas_rating_all_np)
    # np.save(opj(outpath, bname + '_' + epo_type + '_betas_stim_rob.npy'),
    #         betas_stim_all_robnp)
    # np.save(opj(outpath, bname + '_' + epo_type + '_betas_rate_rob.npy'),
    #         betas_rating_all_robnp)
    np.save(opj(outpath, bname + '_' + epo_type + '_betas_stim_rob2.npy'),
            betas_stim_all_rob2np)
    np.save(opj(outpath, bname + '_' + epo_type + '_betas_rate_rob2.npy'),
            betas_rating_all_rob2np)
    betas_stim_all_src = mne.VolSourceEstimate(data=np.swapaxes(betas_stim_all_np, 1, 0),
                                            vertices=vertices, subject=sub,
                                            tmin=0, tstep=1)
    betas_stim_all_src.save(opj(outpath, bname + '_' + epo_type + '_betas_stim-src'))

    betas_rating_all_src = mne.VolSourceEstimate(data=np.swapaxes(betas_rating_all_np, 1, 0),
                                            vertices=vertices, subject=sub,
                                            tmin=0, tstep=1)
    betas_rating_all_src.save(opj(outpath, bname + '_' + epo_type + '_betas_rate-src'))

    reject_dict.to_csv(opj(outpath, epo_type + '_rejectdict.csv'))




# Create inputs list
params = []
from joblib import Parallel, delayed
for epo_type in epo_types:
    for bname, bedge in bands.items():
        # Loop parts
        params.append((bname, bedge, epo_type))


# Run it all (need 16 cores, otherwise reduce n_jobs)
Parallel(n_jobs=4)(delayed(run_all)(params[i][0], params[i][1], params[i][2])
                   for i  in tqdm(range(len(params))))

