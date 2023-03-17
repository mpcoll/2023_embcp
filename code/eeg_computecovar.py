# Authors: mpcoll
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances, Shrinkage
from mne.decoding import Scaler, Vectorizer
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mne.datasets import sample
from mne import io
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, HankelCovariances
import numpy as np
import pandas as pd
import mne
import torch
from os.path import join as opj
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from braindecode import EEGRegressor, EEGClassifier
from braindecode.preprocessing import exponential_moving_standardize, Preprocessor, preprocess
from braindecode.visualization import plot_confusion_matrix
from braindecode.datasets import create_from_X_y
from braindecode.models import Deep4Net, ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.augmentation import FTSurrogate, AugmentedDataLoader

from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from skorch.helper import SliceDataset, predefined_split
from skorch.dataset import Dataset

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             confusion_matrix)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split
from sklearn.base import clone
import faulthandler
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import coffeine

faulthandler.enable()

# Check if on ccanada
if 'lustre' in os.getcwd():
    ccanada = 1
    bidsout = '/lustre04/scratch/mpcoll/2023_embcp'
else:
    ccanada = 0
    bidsout = '/Users/mp/data/2023_embcp'

derivpath = opj(bidsout, 'derivatives')
if not os.path.exists(opj(derivpath, 'machinelearning')):
    os.mkdir(opj(derivpath, 'machinelearning'))


# Hide data loading info
mne.set_log_level('WARNING')


####################################################
# Load data
####################################################
print('Loading data...')
dataset = mne.read_epochs(opj(derivpath, 'all_epochs-epo.fif'))


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


dataset = dataset[1:100]
covs = Parallel(n_jobs=10,
                verbose=0)(delayed(compute_covs_epoch)(epoch=dataset[i],
                                                       frequency_bands=frequency_bands)
                           for i in tqdm(range(len(dataset))))


dataset_cov = []
for i in tqdm(range(len(dataset))):
    epoch_cv = coffeine.power_features._compute_covs_epochs(
        dataset[i].apply_baseline((None, None)), frequency_bands)
    dataset_cov.append(np.expand_dims(epoch_cv, 0))

dataset_cov = np.vstack(dataset_cov)

X = pd.DataFrame({band: list(dataset_cov[:, ii])
                 for ii, band in enumerate(frequency_bands)})
filter_bank_transformer = coffeine.make_filter_bank_transformer(
    names=list(frequency_bands),
    method='riemann',
    projection_params=dict(scale='auto', n_compo=rank)
)
model = make_pipeline(
    filter_bank_transformer, StandardScaler(),
    RidgeCV(alphas=np.logspace(-5, 10, 100)))


dataset_cov = np.vstack(dataset_cov)
test = coffeine.power_features._compute_covs_epochs(
    dataset_test, frequency_bands)

features, meta_info = coffeine.compute_features(
    dataset_test, features=('covs',), n_fft=1024, n_overlap=512,
    fs=dataset.info['sfreq'], fmax=100, frequency_bands=frequency_bands)


freq_bands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
n_freq_bands = len(freq_bands)
n_subjects = 10
n_channels = 4
X_cov = np.random.randn(n_subjects, n_freq_bands, n_channels, n_channels)
for sub in range(n_subjects):
    for fb in range(n_freq_bands):
        X_cov[sub, fb] = X_cov[sub, fb] @ X_cov[sub, fb].T
X_df = pd.DataFrame(
    {band: list(X_cov[:, ii]) for ii, band in enumerate(freq_bands)})
