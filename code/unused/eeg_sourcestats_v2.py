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
from scipy.stats import linregress, zscore, ttest_1samp
from mne.stats import fdr_correction
from mne.stats import permutation_t_test, permutation_cluster_1samp_test
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
###############################
# Parameters
###############################

source_path = '/media/mp/lxhdd/2020_embcp/source'
layout = BIDSLayout(source_path)
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/source_analysis_lmm'

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



subject_dir = 'fsaverage'
fwd = mne.read_forward_solution(opj(outpath, 'forward_solution-fwd.fif'))
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# src = opj(fs_dir, 'bem', 'fsaverage-ico-4-src.fif')
bem = opj(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

src = mne.read_source_spaces(opj(outpath, 'forward_solution-fwd.fif'))


src_est = mne.read_source_estimate(opj(outpath, 'template-src-vol.stc'))

part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]

adjacency = mne.spatial_src_adjacency(fwd['src'])

epo_types = ['thermal_start',
             'auditory_start',
             'thermalrate_start',
             'auditoryrate_start']


# Loop bands

from bambi import Model

def fit_lmm(epo_type, bname, src_est, fwd):
        # Load lmm data
        lmm_dat = pd.read_csv(opj(outpath, bname + '_'
                                  + epo_type + '_lmm_data.csv'))


        numeric_cols = lmm_dat.select_dtypes(include=[np.number]).columns
        lmm_dat[numeric_cols] = lmm_dat[numeric_cols].apply(zscore, nan_policy='omit')

        # Mixed models
        tvals_rate, pvals_rate, slopes_rate =  np.zeros(src_est.shape[0]), np.zeros(src_est.shape[0]), np.zeros(src_est.shape[0])
        tvals_stim, pvals_stim, slopes_stim =  np.zeros(src_est.shape[0]), np.zeros(src_est.shape[0]), np.zeros(src_est.shape[0])

        for vox in tqdm(range(src_est.shape[0])):
            mod_dat = lmm_dat[["tcourse_vox" + str(vox), 'ratings', 'stimulation', 'participant_id']]
            model = Model(mod_dat, dropna=True)
            results = model.fit("tcourse_vox" + str(vox) + " ~ ratings + (ratings|participant_id)")

            
            # md = smf.mixedlm("ratings ~ tcourse_vox" + str(vox), lmm_dat,
            #                 groups=lmm_dat["participant_id"],
            #                 re_formula=  "~ tcourse_vox" + str(vox),
            #                 missing='drop').fit(method='lbfgs', maxiter=9999999)
            # assert md.converged
            tvals_rate[vox] = md.tvalues[1]
            slopes_rate[vox] = md.fe_params[1]
            pvals_rate[vox] = md.pvalues[1]

            md = smf.mixedlm("stimulation ~ tcourse_vox" + str(vox), lmm_dat, groups=lmm_dat["participant_id"],
                            re_formula=  "~ tcourse_vox" + str(vox),
                    missing='drop').fit(method='lbfgs', maxiter=9999999)
            # assert md.converged
            tvals_stim[vox] = md.tvalues[1]
            slopes_stim[vox] = md.fe_params[1]
            pvals_stim[vox] = md.pvalues[1]


        # T-tests on betas
        p_stim_fdr = fdr_correction(pvals_stim, alpha=0.05)[1]
        t_stim_fdr = np.where(p_stim_fdr < 0.05, tvals_stim, 0)
        # t, p ,h = permutation_t_test(betas_stim_all_np, n_permutations=10000,
        #                           tail=0, n_jobs=5)
        stats_stim = mne.VolSourceEstimate(data=tvals_stim,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')

        stats_stim_vol_unc = stats_stim.as_volume(src=fwd['src'])

        stats_stim = mne.VolSourceEstimate(data=t_stim_fdr,
                                           vertices=src_est.vertices,
                                           tmin=0, tstep=1,
                                           subject='fsaverage')
        stats_stim_vol_fdr = stats_stim.as_volume(src=fwd['src'])


        p_rate_fdr = fdr_correction(pvals_rate, alpha=0.05)[1]
        t_rate_fdr = np.where(p_rate_fdr < 0.05, tvals_rate, 0)

        stats_rate = mne.VolSourceEstimate(data=tvals_rate,
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

        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'rating_unc.png'))

        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_stim_vol_unc, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' stimulation  (unc. t > 2)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'stim_unc.png'))
        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_srate_vol_fdr, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' rating (FDR)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'rating_fdr.png'))
        plt.figure(figsize=(10, 5))
        fig = plotting.plot_stat_map(stats_stim_vol_fdr, threshold=2,
                               title=bname + ' ' + epo_type + ' ' + ' stimulation (FDR)')
        fig.savefig(opj(outpath, bname + '_' + epo_type + '_' + 'stim_fdr.png'))




params = []
from joblib import Parallel, delayed
for epo_type in epo_types:
    for bname, bedge in bands.items():
        # Loop parts
        params.append((epo_type, bname, src_est, fwd))


# Run it all
Parallel(n_jobs=2)(delayed(fit_lmm)(params[i][0], params[i][1], params[i][2], params[i][3])
                   for i  in tqdm(range(len(params))))

