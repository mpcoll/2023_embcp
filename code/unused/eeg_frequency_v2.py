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
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm
###############################
# Parameters
###############################

source_path = '/media/mp/lxhdd/2020_embcp/source'
layout = BIDSLayout(source_path)
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives2'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/source_analysis_v2'
# use robust regression
use_robust = True
robust_reg = HuberRegressor()

if not os.path.exists(outpath):
    os.mkdir(outpath)

part = ['sub-' + s for s in layout.get_subject()]
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

# Load data
bands = {'alpha': [8, 13],
         'beta': [14, 29],
         'gamma': [30, 100],
         'theta': [4, 7],}


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
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]


epo_types = [
            'thermal_start',
            'auditory_start',
            'thermalrate_start',
            'auditoryrate_start']




# Loop bands
for epo_type in epo_types:
    if 'thermal' in epo_type:
        rate_col = 'pain_rating'
    elif 'auditory' in epo_type:
        rate_col = 'audio_rating'
    else:
        raise BaseException('Check rating column.')
    for bname, bedge in bands.items():
        # Loop parts
        betas_rating_all = []
        betas_stim_all = []
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

            # Get good epochs
            epo = mne.make_fixed_length_epochs(raw, 0.2, preload=True)
            to_keep = np.ones(len(epo)).astype(bool)
            epo.drop_bad(reject=dict(eeg=200e-6))
            to_keep[np.where(epo.drop_log)[0]] = False

            # 1. bandpass filter the data
            # base_raw.filter(bedge[0], bedge[1])
            raw.filter(bedge[0], bedge[1])

            # Remove the bad epochs

            # # 2. compute the covariance matrices
            # baseline_cov = mne.compute_raw_covariance(base_raw, tmin=0,
            #                                           tstep=1,
            #                                           tmax=None,
            #                                           reject=dict(eeg=100e-6))


            active_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None,
                                                    tstep=1,
                                                    reject=dict(eeg=100e-6))


            filters = mne.beamformer.make_lcmv(raw.info, fwd,
                                               data_cov=active_cov,
                                               reduce_rank=True,
                                               noise_cov=None,
                                               pick_ori='max-power',
                                               weight_norm='unit-noise-gain', reg=0.05)

            # 4. compute hilbert transform on raw data
            raw = raw.apply_hilbert(n_jobs=10, envelope=True)

            # 5. use spatial filter on hilbert data
            stcs = mne.beamformer.apply_lcmv_raw(raw, filters,
                                                  max_ori_out= 'signed')

            # Downsample
            stcs.resample(100)

            # Smooth and zscore source data
            dat = np.zeros(stcs.data.shape)
            for ii in range(stcs.data.shape[0]):
                dat[ii, :] = smooth(np.abs(stcs.data[ii, :]), 101)

            # _________________________________________________________________
            # header2
            # Load ratingsm donwsample and smooth
            behav = pd.read_csv(opj(source_path, p, 'eeg',
                                    p + '_task-painaudio_beh.tsv'), sep='\t')
            behav = behav.reset_index()[behav.reset_index()['index']%5==0].reset_index(drop=True)
            smht_ratings = zscore(smooth(np.asarray(behav[rate_col]), 101))
            smht_stim = zscore(smooth(np.asarray(behav['stim_intensity']), 101))




            # Initialize betas arrays


            # Calculate regression between behav and eeg
            # print(bname)
            # print(epo_type)
            # print(p)
            betas_rating = np.zeros(dat.shape[0])
            betas_stim = np.zeros(dat.shape[0])
            # Add dim for sklearn


            for vertex in tqdm(range(dat.data.shape[0])):
                eeg_vertex = dat[vertex, :]
                # eeg_vertex = np.expand_dims(eeg_vertex, 1)
                # Get beta value using regression
                # if not use_robust:
                betas_rating[vertex] = linregress(np.squeeze(eeg_vertex), smht_ratings)[0]
                betas_stim[vertex] = linregress(np.squeeze(eeg_vertex), smht_stim)[0]

                # betas_rating[vertex] = robust_reg.fit(eeg_vertex, smht_ratings).coef_
                # betas_stim[vertex] = robust_reg.fit(eeg_vertex, smht_stim).coef_

            betas_rating_all.append(betas_rating)
            betas_stim_all.append(betas_stim)

        # Save as a array and source estimate with subjects insteadd of time points
        betas_stim_all_np = np.stack(betas_stim_all)
        betas_rating_all_np = np.stack(betas_rating_all)
        np.save(opj(outpath, bname + '_' + epo_type + '_betas_stim.npy'),
                betas_stim_all_np)
        np.save(opj(outpath, bname + '_' + epo_type + '_betas_rate.npy'),
                betas_rating_all_np)

        betas_stim_all_src = mne.VolSourceEstimate(data=np.swapaxes(betas_stim_all_np, 1, 0),
                                                vertices=stcs.vertices, subject=stcs.subject,
                                                tmin=0, tstep=1)
        betas_stim_all_src.save(opj(outpath, bname + '_' + epo_type + '_betas_stim-src'))

        betas_rating_all_src = mne.VolSourceEstimate(data=np.swapaxes(betas_rating_all_np, 1, 0),
                                                vertices=stcs.vertices, subject=stcs.subject,
                                                tmin=0, tstep=1)
        betas_rating_all_src.save(opj(outpath, bname + '_' + epo_type + '_betas_rate-src'))



        # betas_rating_all_np = np.load(opj(outpath, bname + '_' + epo_type + '_betas_rate.npy'))
        # betas_stim_all_np = np.load(opj(outpath, bname + '_' + epo_type + '_betas_stim.npy'))
        # # T-tests on betas
        # from mne.stats import permutation_t_test
        # t_stim, p_stim = ttest_1samp(betas_stim_all_np, 0)
        # p_stim_fdr = fdr_correction(p_stim, alpha=0.05)[1]
        # t_stim_fdr = np.where(p_stim_fdr < 0.05, t_stim, 0)
        # t, p ,h = permutation_t_test(betas_stim_all_np, n_permutations=10000,
        #                           tail=0, n_jobs=5)


        # t_rate, p_rate = ttest_1samp(betas_rating_all_np, 0)
        # p_rate_fdr = fdr_correction(p_rate, alpha=0.05)[1]
        # t_rate_fdr = np.where(p_rate_fdr < 0.05, t_rate, 0)

