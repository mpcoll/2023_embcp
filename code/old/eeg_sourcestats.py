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
from mne.stats import permutation_t_test, permutation_cluster_1samp_test
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image

###############################
# Parameters
###############################

source_path = '/media/mp/lxhdd/2020_embcp/source'
layout = BIDSLayout(source_path)
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/source_analysis_v4'

mni = datasets.load_mni152_template()
if not os.path.exists(outpath):
    os.mkdir(outpath)

part = ['sub-' + s for s in layout.get_subject()]
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

# Load data
bands = {'alpha': [8, 13],
         'gamma': [30, 100],
        'gamma2': [45, 100],
         'theta': [4, 7],
         'beta': [14, 29],}



epo_types = ['rest_start',
            'thermal_start',
            'auditory_start',
            'thermalrate_start',
            'auditoryrate_start']


subject_dir = 'fsaverage'
fwd = mne.read_forward_solution(opj(outpath, 'forward_solution-fwd.fif'))
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# src = opj(fs_dir, 'bem', 'fsaverage-ico-4-src.fif')
bem = opj(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

src = mne.read_source_spaces(opj(outpath, 'forward_solution-fwd.fif'))


src_est = mne.read_source_estimate(opj(outpath, 'alpha_thermal_start_betas_stim-src-vl.stc'))

part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

adjacency = mne.spatial_src_adjacency(fwd['src'])

epo_types = ['thermal_start',
             'auditory_start',
             'thermalrate_start',
             'auditoryrate_start']


# Loop bands
for epo_type in epo_types:
    for bname, bedge in bands.items():
        rate_betas = np.load(opj(outpath, bname + '_'
                                 + epo_type + '_betas_rate.npy'))

        stim_betas = np.load(opj(outpath, bname + '_'
                            + epo_type + '_betas_stim.npy'))


        # T-tests on betas
        t_stim, p_stim = ttest_1samp(stim_betas, 0)
        p_stim_fdr = fdr_correction(p_stim, alpha=0.05)[1]
        t_stim_fdr = np.where(p_stim_fdr < 0.05, t_stim, 0)
        # t, p ,h = permutation_t_test(betas_stim_all_np, n_permutations=10000,
        #                           tail=0, n_jobs=5)
        stats_stim = mne.VolSourceEstimate(data=t_stim,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')

        stats_stim_vol_unc = stats_stim.as_volume(src=fwd['src'])

        stats_stim = mne.VolSourceEstimate(data=t_stim_fdr,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')
        stats_stim_vol_fdr = stats_stim.as_volume(src=fwd['src'])


        t_rate, p_rate = ttest_1samp(rate_betas, 0)
        p_rate_fdr = fdr_correction(p_rate, alpha=0.05)[1]
        t_rate_fdr = np.where(p_rate_fdr < 0.05, t_rate, 0)

        stats_rate = mne.VolSourceEstimate(data=t_rate,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')

        stats_rate_vol_unc = stats_rate.as_volume(src=fwd['src'])

        stats_rate = mne.VolSourceEstimate(data=t_rate_fdr,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')
        stats_srate_vol_fdr = stats_rate.as_volume(src=fwd['src'])


        # Make some quick plots
        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_rate_vol_unc, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' rating (unc. t > 2)')

        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'rating_unc_norob.png'))

        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_stim_vol_unc, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' stimulation  (unc. t > 2)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'stim_unc_norob.png'))
        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_srate_vol_fdr, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' rating (FDR)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'rating_fdr_norob.png'))
        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_stim_vol_fdr, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' stimulation (FDR)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'stim_fdr_norob.png'))



###################################################################
# Contrasts
###################################################################


cond1 = 'thermal_start'
cond2 = 'thermalrate_start'
band = 'gamma'

betas1 = np.load(opj(outpath, bname + '_'
                            + cond1 + '_betas_stim.npy'))
