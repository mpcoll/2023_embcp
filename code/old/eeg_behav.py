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
import seaborn as sns

###############################
# Parameters
###############################

source_path = '/media/mp/lxhdd/2020_embcp/source'
layout = BIDSLayout(source_path)
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives'
outpath = '/media/mp/lxhdd/2020_embcp/derivatives/behav'

if not os.path.exists(outpath):
    os.mkdir(outpath)

part = ['sub-' + s for s in layout.get_subject()]
part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027']]


all_prate, all_arate, stim = [], [], []
for p in part:
    print(p)
    pdir = opj(derivpath, p, 'eeg')
    # Load ratings
    behav = pd.read_csv(opj(source_path, p, 'eeg',
                            p + '_task-painaudio_beh.tsv'), sep='\t')

    all_prate.append(np.asarray(behav['pain_rating']))
    all_arate.append(np.asarray(behav['audio_rating']))
    stim.append(behav['stim_intensity'])
    # for epo_type in epo_types:

all_prate = np.vstack(all_prate)
all_arate = np.vstack(all_arate)
stim = np.vstack(stim)

# Downsample a bit
all_prate_d = all_prate[:, 0:all_prate.size:100]
all_arate_d = all_arate[:, 0:all_arate.size:100]
stim_d = stim[:, 0:stim.size:100]


plot_prate = pd.DataFrame(index=part, data=all_prate_d).reset_index().melt(id_vars='index', value_name='pain')
plot_arate = pd.DataFrame(index=part, data=all_arate_d).reset_index().melt(id_vars='index', value_name='audio')
plot_stim = pd.DataFrame(index=part, data=stim_d).reset_index().melt(id_vars='index', value_name='stim')

plot_stim['variable'] = plot_stim['variable']/(500/100)
plot_arate['variable'] = plot_arate['variable']/(500/100)
plot_prate['variable'] = plot_prate['variable']/(500/100)

# plot_stim  = plot_stim.groupby(['index', 'variable']).mean()

fig, ax = plt.subplots(figsize=(4, 3))
sns.lineplot(x='variable', y='pain', data=plot_prate, ci=None)
sns.lineplot(x='variable', y='audio', data=plot_arate, ci=None)
sns.lineplot(x='variable', y='stim', data=plot_stim, ci=None, color='k', alpha=0.8, linestyle='dotted')
ax.set_ylabel('Intensity rating',)
ax.set_xlabel('Time (s)')
ax.set_ylim(0, 200)
plt.tight_layout()

fig.savefig(opj(outpath, 'all_rate.png'), dpi=600)
# ax.axhline(100, color='gray', linestyle='--', alpha=0.5)


fig, ax = plt.subplots(figsize=(4, 3))
sns.lineplot(x='variable', y='pain', data=plot_prate, ci=95)
sns.lineplot(x='variable', y='stim', data=plot_stim, ci=None, color='k', alpha=0.8, linestyle='dotted')
ax.set_ylabel('Intensity rating',)
ax.set_xlabel('Time (s)')
ax.set_ylim(0, 200)

ax.set_title('Pain intensity rating')
plt.tight_layout()
fig.savefig(opj(outpath, 'pain_rate.png'), dpi=600)


fig, ax = plt.subplots(figsize=(4, 3))
sns.lineplot(x='variable', y='audio', data=plot_arate, ci=95)
sns.lineplot(x='variable', y='stim', data=plot_stim, ci=None, color='k', alpha=0.8, linestyle='dotted')
ax.set_ylabel('Intensity rating',)
ax.set_xlabel('Time (s)')
ax.set_ylim(0, 200)
ax.set_title('Audio intensity rating')
plt.tight_layout()
fig.savefig(opj(outpath, 'audio_rate.png'), dpi=600)


plot_prate['stim'] = plot_stim['stim']

fig = sns.lmplot(data=plot_prate, y='pain', x='stim', hue='index', scatter=False,
            palette='plasma', legend=False, height=3, aspect=4/3)
plt.xlabel('Stimulation intensity')
plt.ylabel('Pain rating')
# plt.ylim(0, 200)
plt.tight_layout()

fig.savefig(opj(outpath, 'pain_slopes.png'), dpi=600)

plot_arate['stim'] = plot_stim['stim']

fig = sns.lmplot(data=plot_arate, y='audio', x='stim', hue='index', scatter=False,
            palette='plasma', legend=False, height=3, aspect=4/3)
plt.xlabel('Stimulation intensity')
plt.ylabel('Audio rating')
# plt.ylim(0, 200)
plt.tight_layout()

fig.savefig(opj(outpath, 'audio_slopes.png'), dpi=600)
