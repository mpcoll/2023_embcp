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
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/source_analysis'
# use robust regression
use_robust = True
robust_reg = HuberRegressor()

if not os.path.exists(outpath):
    os.mkdir(outpath)

part = ['sub-' + s for s in layout.get_subject()]
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

# Load data
bands = {'theta': [4, 7],
         'alpha': [8, 13],
         'beta': [14, 29],
         'gamma': [30, 100]}
bands = {
         'beta': [14, 29],
         'gamma': [30, 100]}


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

src = opj(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')

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
    # src = mne.setup_volume_source_space('fsaverage', 6,
    #                             subjects_dir=subjects_dir, verbose=True)

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
                                    bem=bem, eeg=True, mindist=7.5, n_jobs=1)
    mne.write_forward_solution(opj(outpath, 'forward_solution-fwd.fif'), fwd,
                               overwrite=True)
else:
    fwd = mne.read_forward_solution(opj(outpath, 'forward_solution-fwd.fif'))




# Use fwd to compute the sensitivity map for illustration purposes
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
# brain = eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
#                      clim=dict(lims=[5, 50, 100]))

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
            base_raw = mne.io.read_raw_fif(opj(pdir, p
                            + '_task-painaudio_'
                            + 'rest_start' + '_cleanedeeg_raw.fif'),
                        preload=True)


            # Load cond file
            raw = mne.io.read_raw_fif(opj(pdir, p
                                            + '_task-painaudio_'
                                            + epo_type + '_cleanedeeg_raw.fif'),
                                        preload=True)

            # Bandpass filter
            raw = raw.filter(bedge[0], bedge[1], n_jobs=10)

            # Hilbert transform
            raw = raw.apply_hilbert(envelope=True, n_jobs=10)


            baseline_cov = mne.compute_raw_covariance(base_raw, tmin=0,
                                                      tstep=1,
                                                      tmax=None, method='shrunk',
                                                  rank=None, reject=dict(eeg=100e-6))





            active_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None,
                                                    tstep=1,
                                            method='shrunk', rank=None,
                                            reject=dict(eeg=100e-6))


            common_cov = baseline_cov + active_cov

            # generate lcmv source estimate

            filters = mne.beamformer.make_lcmv(raw.info,
                                                fwd, common_cov, reg=0.05,
                                noise_cov=None, pick_ori='max-power')
            stc_base = mne.beamformer.apply_lcmv_cov(baseline_cov, filters)
            stc_act = mne.beamformer.apply_lcmv_cov(active_cov, filters)
            stc_act /= stc_base


            # Load ratings, downsample and smooth
            behav = pd.read_csv(opj(source_path, p, 'eeg',
                                    p + '_task-painaudio_beh.tsv'), sep='\t')
            behav = behav.reset_index()[behav.reset_index()['index']%5==0].reset_index(drop=True)
            # Smooth ratings with (1 s window)
            smht_ratings = zscore(smooth(np.asarray(behav[rate_col]), 101))
            smht_stim = zscore(smooth(np.asarray(behav['stim_intensity']), 101))




            # Smooth 1 s window
            smth_raw = raw.get_data().copy()
            for c in range(raw.get_data().shape[0]):
                smth_raw[c, :] = smooth(raw.get_data()[c, :], int(raw.info['sfreq']+1))

            # Reput in array
            raw = mne.io.RawArray(data=smth_raw, info=raw.info)

            # Resmaple both to 100 Hz
            raw = raw.resample(100)
            behav = behav.reset_index()[behav.reset_index()['index']%5==0].reset_index(drop=True)
            # Smooth ratings with (1 s window)
            smht_ratings = zscore(smooth(np.asarray(behav[rate_col]), 101))
            smht_stim = zscore(smooth(np.asarray(behav['stim_intensity']), 101))



            # Apply filters
            raw = raw.set_eeg_reference(projection=True)
            raw_source = mne.beamformer.apply_lcmv_raw(raw, filters=lcmv,
                                                    verbose=True)

            # Zscore eeg according to time
            dat = zscore(raw_source.data, axis=1)
            # Add dim for sklearn
            dat = np.expand_dims(raw_source.data, 2)

            # Initialize betas arrays
            betas_rating = np.zeros(raw_source.shape[0])
            betas_stim = np.zeros(raw_source.shape[0])

            # Calculate regression between behav and eeg
            print(bname)
            print(epo_type)
            print(p)

            for vertex in tqdm(range(raw_source.data.shape[0])):
                eeg_vertex = dat[vertex]

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

        betas_stim_all_src = mne.SourceEstimate(data=np.swapaxes(betas_stim_all_np, 1, 0),
                                                vertices=raw_source.vertices,
                                                tmin=0, tstep=1)
        betas_stim_all_src.save(opj(outpath, bname + '_' + epo_type + '_betas_stim-src'))

        betas_rating_all_src = mne.SourceEstimate(data=np.swapaxes(betas_rating_all_np, 1, 0),
                                                vertices=raw_source.vertices,
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

