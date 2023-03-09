from mne.datasets import sample
from mne import read_source_estimate
import mne
import os
from os.path import join as opj
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import HuberRegressor, LinearRegression
from scipy.stats import zscore
from joblib import Parallel, delayed
import multiprocessing

njobs = multiprocessing.cpu_count()
# %matplotlib qt5


# Setup paths
bidsroot = "/media/mp/mpx6/2023_embcp/"
derivpath = opj(bidsroot, "derivatives")
if not os.path.exists(opj(derivpath, 'source')):
    os.mkdir(opj(derivpath, 'source'))

# Freesurfer templates
fs_surfer_dir = opj(derivpath, 'freesurfer')
if not os.path.exists(opj(derivpath, 'freesurfer')):
    os.makedirs(opj(derivpath, 'freesurfer', 'subjects'))
    # Download templates if not available on disk
    mne.datasets.fetch_fsaverage(opj(derivpath, 'freesurfer', 'subjects'))

# Get participants
part = [s for s in os.listdir(derivpath) if "sub-" in s]

# Frequencies to test
freqdict = {'alpha': (8, 13),
            'beta': (14, 29),
            'gamma': (30, 100)}

# Setup freesurfer
os.environ['FREESURFER_HOME'] = fs_surfer_dir
os.environ['SUBJECTS_DIR'] = opj(fs_surfer_dir, 'subjects')

# Create source space
subjects = 'fsaverage'

raw = mne.io.read_raw_fif(opj(derivpath, 'sub-003/eeg', 'sub-003'
                              + "_task-painaudio_rest_start"
                              + "_cleanedeeg-raw.fif"), preload=True)

##################################
# Setup source space
##################################

# Create template source space
src = mne.setup_source_space('fsaverage',
                             spacing="oct6",  # Mne bids default
                             subjects_dir=None,
                             add_dist="patch")

bem = opj(os.environ['SUBJECTS_DIR'], 'fsaverage',
          'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# PLot source space and alignment
# src.plot(subjects_dir=None)

# mne.viz.plot_alignment(
#     raw.info, src=src, eeg=['original', 'projected'], trans='fsaverage',
#     show_axes=True, mri_fiducials=True, dig='fiducials')

##################################
# Make foward models for all part
##################################


fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=None)
adjacency = mne.spatial_src_adjacency(src)
src = None  # save memory


# MNE-BIDS pipeline values
snr = 3.0
lambda2 = 1.0 / snr**2


all_inverse_operators = dict()
for p in tqdm(part, total=len(part)):

    rest_raw = mne.io.read_raw_fif(opj(derivpath, p,  'eeg', p
                                       + "_task-painaudio_" + 'rest' + "_start"
                                       + "_cleanedeeg-raw.fif"), preload=True)

    # Make noise covariance from resting state
    rest_noise_cov = mne.cov.compute_raw_covariance(rest_raw, verbose=None)

    # noise_cov.plot(raw.info, show_svd=False, show=False)
    # Use only diagonal as in Brainstorm
    rest_noise_cov.as_diag()

    all_inverse_operators[p] = make_inverse_operator(rest_raw.info, fwd,
                                                     rest_noise_cov,
                                                     depth=None, verbose=True)

fwd = None  # save memory


##################################
# Inverse solution
##################################
def regressor(data, dipole, Y, method, zscoreY=False, zscoreX=True):
    # Randomly select observations
    X = data[dipole, :].reshape(-1, 1)
    if zscoreX:
        X = zscore(X)
    if zscoreY:
        Y = zscore(Y)
    # Return the weights and stats
    return method.fit(X, Y).coef_[0]


# TODO change loop to avoid doing mutliple times the same thing
# Loop tasks
for task in ['auditory', 'auditoryrate', 'thermal', 'thermalrate']:
    # Loop bands
    betas_rate_out = dict()
    betas_intens_out = dict()
    for name, band in freqdict.items():
        betas_rate_out[name] = []
        betas_intens_out[name] = []
    # Loop participants
    for p in part:

        raw = mne.io.read_raw_fif(opj(derivpath, p,  'eeg', p
                                      + "_task-painaudio_" + task + "_start"
                                        + "_cleanedeeg-raw.fif"), preload=True)

        # Downsample to 250 Hz
        raw.resample(250)

        # Set EEG reference again, # needed for inverse modeling
        raw.set_eeg_reference('average', projection=True)

        # Apply inverse
        stc = apply_inverse_raw(raw, all_inverse_operators[p], lambda2,
                                method="dSPM",
                                pick_ori=None)

        # Get behavioural data
        intensity = zscore(raw.get_data()[-1, :])
        rating = zscore(raw.get_data()[-2, :])

        # Put in a raw array to filter
        src_info = mne.create_info([str(i) for i in np.arange(stc.data.shape[0])],
                                   raw.info['sfreq'],
                                   ch_types='eeg',
                                   verbose=None)
        raw_stc = mne.io.RawArray(stc.data, src_info)

        for name, band in freqdict.items():
            raw_fh = raw_stc.copy()
            # Filter and hilbert source data
            raw_stc.filter(band[0], band[1])
            raw_stc.apply_hilbert(envelope=True)
            # Replace source data with hilbert
            stc.data = raw_stc.get_data()

            # regress with ratings and get betas

            betas_rate = Parallel(n_jobs=njobs,
                                  verbose=0)(delayed(regressor)(data=stc.data,
                                                                Y=rating,
                                                                method=HuberRegressor(),
                                                                dipole=dipole,)
                                             for dipole in tqdm(range(stc.data.shape[0])))

            betas_intensity = Parallel(n_jobs=njobs, verbose=0)(delayed(regressor)(data=stc.data,
                                                                                   Y=intensity,
                                                                                   method=HuberRegressor(),
                                                                                   dipole=dipole,)
                                                                for dipole in tqdm(range(stc.data.shape[0])))
            # betas_rate2 = np.zeros(stc.data.shape[0])
            # for dipole in tqdm(range(stc.data.shape[0])):
            #     # regress
            #     betas_rate2[dipole] = HuberRegressor().fit(zscore(stc.data[dipole, :]).reshape(-1, 1),
            #                                                 rating
            #                                               ).coef_
            #     betas_intensity[dipole] = HuberRegressor().fit(intensity,
            #                                                    zscore(stc.data[dipole, :])).coef_

            # beta_src = mne.SourceEstimate(betas, stc.vertices, 0, 0)
            # beta_src.plot(**kwargs)
            betas_rate_out[name].append(betas_rate)
            betas_intens_out[name].append(betas_intensity)

        # Save betas
    for name, band in freqdict.items():

        np.save(opj(derivpath, 'source', 'betas_rate_' + task + '_' + name),
                np.vstack(betas_rate_out[name]))
        np.save(opj(derivpath, 'source', 'betas_intens_' + task + '_' + name),
                np.vstack(betas_intens_out[name]))

        # T-test betas
        t_obs, _, pvals, _ = mne.stats.permutation_cluster_1samp_test(np.vstack(betas_intens_out[name]),
                                                                      adjacency=adjacency,
                                                                      n_permutations=1000,
                                                                      n_jobs=njobs,
                                                                      threshold=dict(start=0, step=0.2))
        # Save to file
        mne.SourceEstimate(t_obs, stc.vertices, 0, 0).save(
            opj(derivpath, 'source', 'tvals_intens_' + task + '_' + name + '.fif'), overwrite=True)
        mne.SourceEstimate(pvals, stc.vertices, 0, 0).save(
            opj(derivpath, 'source', 'pvals_intens_' + task + '_' + name + '.fif'), overwrite=True)

        t_obs, _, pvals, _ = mne.stats.permutation_cluster_1samp_test(np.vstack(betas_rate_out[name]),
                                                                      adjacency=adjacency,
                                                                      threshold=dict(start=0, step=0.2))
        mne.SourceEstimate(t_obs, stc.vertices, 0, 0).save(
            opj(derivpath, 'source', 'tvals_rate_' + task + '_' + name + '.fif'), overwrite=True)
        mne.SourceEstimate(pvals, stc.vertices, 0, 0).save(
            opj(derivpath, 'source', 'pvals_rate_' + task + '_' + name + '.fif'), overwrite=True)



# TODO loop over tasks and create figures
# # Plot betas
# test.plot(**kwargs)
kwargs = dict(subject='fsaverage',
               subjects_dir=os.environ['SUBJECTS_DIR'],
               hemi='split', smoothing_steps=4,
               time_unit='s', initial_time=1, size=1200,
               views=['lat', 'med'])

 # # Plot betas
stct = mne.read_source_estimate(
     opj(derivpath, 'source', 'tvals_intens_thermal_gamma.fif'))

 np.max(np.abs(stct.data))
 stcp = mne.read_source_estimate(
     opj(derivpath, 'source', 'pvals_intens_thermal_alpha.fif'))

 stct.data[stcp.data > 0.05] = 0
# # # save betas / append for group analysis
 stct.plot(**kwargs)
# test = mne.SourceEstimate(np.mean(betas_intens_out, 0).squeeze(), stc.vertices, 0, 0)
# # # t-test betas
# # # save t-vals


# test = mne.SourceEstimate(np.stack([t_obs, t_obs], 1), stc.vertices, 0, 1,
#                           subject='fsaverage')



##################################
# Make report
##################################
# report.add_bem(
#     subject='fsaverage'
#     title="BEM",
#     width=256,
#     decim=8,
#     replace=True,
#     n_jobs=1,  # prevent automatic parallelization
# )

