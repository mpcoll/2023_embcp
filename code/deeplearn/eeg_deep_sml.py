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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from mne.decoding import Vectorizer, Scaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, make_scorer

###############################
# Parameters
###############################
layout = BIDSLayout("/media/mp/lxhdd/2020_embcp/source")
part = sorted(["sub-" + s for s in layout.get_subject()])
datapath = "/media/mp/lxhdd/2020_embcp/deep_derivatives"


###############################
# Load data
###############################

x_data = []
for p in tqdm(part):
    epochs = mne.read_epochs(opj(datapath, p + '-epo.fif.gz'))
    x_data.append(epochs)

x_data = mne.concatenate_epochs(x_data)
x_data = x_data.drop_channels(['VEOG', 'HEOG'])

x_data.save(opj(datapath, 'alldata-epo.fif'), overwrite=True)
print(x_data.get_data().shape) # Epochs x channels x time


param = {'visualinspect': False,
         # Visually identified bad channels
         'badchannels': {'sub-003': ['PO3', 'C1', 'F3', 'TP7', 'FT9'],
                         'sub-004': ['TP10'], # None
                         'sub-005': [], # None
                         'sub-006': [], # None
                         'sub-007': [], # None
                         'sub-008': ['TP10', 'O1', 'Oz', 'O2', 'PO8'], # None, quite noisy ###
                         'sub-009': ['P7', 'TP7', 'PO7', 'F7', 'F8'],
                         'sub-011': ['PO8', 'O1', 'O2', 'PO7', 'FT8'], # None
                         'sub-012': [], # Data looks weird (Bad ref?), exclude
                         'sub-013': ['T7', 'FT8', 'T8'], # None
                         'sub-014': ['T7', 'TP7'], # None
                         'sub-015': [], # Too noisy, exclude
                         'sub-016': ['T8'], # None
                         'sub-017': ['F8'], # None
                         'sub-018': ['F6', 'AF8', 'Fp2', 'AF7', 'AF4'], # None
                         'sub-019': ['Fp1'], # None
                         'sub-020': ['FT7', 'FC5'], # None
                         'sub-021': ['P6'], # None
                         'sub-022': ['FC5'], # None
                         'sub-023': ['O1'], # None
                         'sub-024': [], # None
                         'sub-025': ['AF8'], # None
                         'sub-026': ['F4', 'FC6', 'C6', 'F6', 'FC4'], # None
                         'sub-027': [], # Data looks weird (Bad ref?), exclude
                         'sub-028': ['FC6', 'F8', 'T8'], # None
                         'sub-029': ['FT7'], # None
                         'sub-030': [], # None
                         'sub-031': ["T7", 'FT8', 'F8', 'FC6'], # None, but noisy ####
                         'sub-032': ['O1'], # None
                         'sub-033': [], # None
                         'sub-034': ['O1', 'O2', 'PO7'], # None
                         'sub-036': ['FT8', 'T8', 'T7'], # None
                         'sub-037': ['TP8', 'T8', 'T7'], # None
                         'sub-038': ['TP10'], # None
                         'sub-039': ['FT10', 'Fp1', 'T8'], # None
                         'sub-040': ['PO3', 'Fp1'],
                         'sub-041': ['CP6', 'C5', 'T7', 'T8'], # None
                         'sub-042': ['Fp2', 'AF8', 'Fp1', 'AF7', 'F6'], # None
                         'sub-043': ['AF7', 'T8'],
                         'sub-044': ['TP10'], # None
                         'sub-045': ['O2', 'P8', 'P1', 'T8', 'PO8'],
                         'sub-046': ['AF8', 'Fp2'], # None
                         'sub-047': ['O2', 'AF7']
                         }}

###############################
# Test LogisticRegression to classify condition
###############################
# Reduce size and conditions
x_data = x_data[x_data.metadata['condition'].isin(['rest_start',
                                                   'thermal_start'])]

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
all_acc = []
for s in list(x_data.metadata['subject_id'].unique()):
    debug_data = x_data[x_data.metadata['subject_id'] == s]

    # Sklearn Pipeline

    clf = Pipeline([('vectorize', Vectorizer()),  # Vectorize
                    ('pca', PCA()),
                    ('lda', Lasso())])  # Classify

    freqs = np.arange(3, 100, 1)
    epochs_tf = mne.time_frequency.tfr_morlet(debug_data,
                                              freqs=np.arange(3, 100, 1),
                                              n_cycles = freqs / 2,
                                              n_jobs=-2,
                                              return_itc=False,
                                              average=False)
    epochs_tf.drop_channels(param['badchannels'][s])
    print(epochs_tf)
    # Participant-wise cross-validation
    group_cv = GroupKFold(n_splits=5).split(epochs_tf.data,
                                            epochs_tf.metadata['condition'],
                                            groups=epochs_tf.metadata['subject_id'])


    epochs_tf_pain = epochs_tf[epochs_tf.metadata['condition'] == 'thermal_start']
    cvs = cross_val_score(clf, X=epochs_tf_pain.data,
                        y=epochs_tf_pain.metadata['stim_intensity'],
                        #   groups=x_data.metadata['subject_id'],
                        scoring='r2',
                        cv=5)

    all_acc.append(str(np.mean(cvs)))

    print(s)
    print('Accuracy %' + str(np.mean(cvs)*100))
    stophere