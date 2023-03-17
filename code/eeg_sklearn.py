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
from sklearn.naive_bayes import GaussianNB
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

from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             confusion_matrix)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split
from sklearn.base import clone
import faulthandler
from tqdm import tqdm

import os
# import coffeine

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
print(dataset)
# debug = 0
# if debug:
#     # If debug, keep only 10 participants
#     dataset = dataset[]


####################################################
# set up model and parameters
####################################################

# Check if GPUs
cuda = torch.cuda.is_available()
if ccanada:
    assert cuda


if cuda:  # If GPU
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
    print('Using ' + str(n_gpus) + ' GPUs')
else:
    device = 'cpu'
    n_gpus = 0


# For macbook M1, use MPS backend
if torch.backends.mps.is_available():
    print("Using MPS")
    device = 'mps'

# Set seed
seed = 42
set_random_seeds(seed=seed, cuda=cuda)

# Get data info
n_chans = dataset.get_data().shape[1]
input_window_samples = dataset.get_data().shape[2]

batch_size = 32
n_epochs = 10

model = 'braindecode_shallow'


def initiate_clf(model_name, n_classes, n_chans=n_chans,
                 input_window_samples=input_window_samples,
                 batch_size=batch_size, device=device,
                 early_stop_n=15,
                 braindecode=True, path_out='',
                 augmentation=False):
    """_summary_

    Args:
        model (_type_): _description_
        n_classes (_type_): _description_
        n_chans (_type_, optional): _description_. Defaults to n_chans.
        input_window_samples (_type_, optional): _description_. Defaults to input_window_samples.
        batch_size (_type_, optional): _description_. Defaults to batch_size.
        device (_type_, optional): _description_. Defaults to device.
        braindecode (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    if braindecode:
        if model_name == 'braindecode_deep':
            lr = 0.01
            weight_decay = 0.0005
            # Model
            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
        elif model_name == 'braindecode_shallow':
            # Model parameters
            lr = 0.0625 * 0.01
            weight_decay = 0
            # Model
            model = ShallowFBCSPNet(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
        # If multiple GPUs, split
        if n_gpus > 1:
            model = torch.nn.parallel.DistributedDataParallell(model)

        if augmentation:
            fts = FTSurrogate(0.5, phase_noise_magnitude=1, channel_indep=False,
                              random_state=seed)
            transforms = [fts]
            iterator_train = AugmentedDataLoader
        else:
            transforms = None
            iterator_train = DataLoader

        if n_classes == 1:  # If regression
            # Remove softmax layer
            new_model = torch.nn.Sequential()
            for name, module_ in model.named_children():
                if "softmax" in name:
                    continue
                new_model.add_module(name, module_)
            model = new_model

            clf_deep = EEGRegressor(
                model,
                optimizer=torch.optim.AdamW,
                iterator_train=iterator_train,
                # iterator_train__transforms=transforms, # TODO check why not working
                optimizer__lr=lr,
                train_split=group_train_valid_split,
                optimizer__weight_decay=weight_decay,
                iterator_valid__shuffle=False,
                iterator_train__shuffle=True,
                batch_size=batch_size,
                callbacks=[
                    "neg_root_mean_squared_error",
                    ("checkpoint", Checkpoint(dirname=path_out, f_criterion=None,
                                              f_optimizer=None, f_history=None,
                                              load_best=True)),
                    # ("lr_scheduler", LRScheduler(policy=ReduceLROnPlateau)),
                    ("lr_scheduler",   LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),
                    ("early_stopping", EarlyStopping(patience=early_stop_n)),
                ],
                device=device,
            )
            clf = Pipeline(steps=[('mnescaler', Scaler(scalings='median')),
                                  ('deeplearn', clf_deep)])

        else:  # If classification
            clf_deep = EEGClassifier(
                model,
                iterator_train=iterator_train,
                iterator_train__transforms=transforms,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                train_split=None,
                optimizer__weight_decay=weight_decay,
                iterator_valid__shuffle=False,
                iterator_train__shuffle=True,
                batch_size=batch_size,
                callbacks=[
                    "accuracy",
                    ("checkpoint", Checkpoint(dirname=path_out, f_criterion=None,
                                              f_optimizer=None, f_history=None,
                                              load_best=True)),
                    # ("lr_scheduler", LRScheduler(policy=ReduceLROnPlateau)),
                    ("lr_scheduler",   LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),

                ],
                device=device,
            )
            clf = Pipeline(steps=[('mnescaler', Scaler(scalings='median')),
                                  ('deeplearn', clf_deep)])

    else:
        if model_name == 'cov_MDM':
            # MDM covariance
            clf = make_pipeline(
                Covariances(),
                Shrinkage(),
                MDM(metric=dict(mean='riemann', distance='riemann'))
            )

        elif model_name == 'cov_SVM':
            # SVC covariance
            clf = make_pipeline(
                Covariances(),
                Shrinkage(),
                Vectorizer(),
                SVC(),
            )

        elif model_name == 'SVC':
            # SVC covariance
            clf = make_pipeline(
                Scaler(scalings='mean'),
                Vectorizer(),
                PCA(n_components=0.8),
                SVC(),
            )
        elif model_name == 'filterbank_SVM':
            frequency_bands = {
                "alpha": (8.0, 15.0),
                "beta_low": (15.0, 30.0),
                "beta_high": (30.0, 45.0),
                "gamma_low": (45.0, 65.0),
                "gamma_high": (65.0, 100.0)}

            filter_bank_transformer = coffeine.make_filter_bank_transformer(
                names=list(frequency_bands.keys()),
                method='riemann',
                projection_params=dict(scale='auto', n_compo=64)
            )
            clf = make_pipeline(filter_bank_transformer,
                                StandardScaler(),
                                GaussianNB()
                                )

    return clf


def group_train_valid_split(X, y, groups, proportion_valid=0.2):
    splitter = GroupShuffleSplit(
        test_size=proportion_valid, n_splits=2, random_state=42)
    split = splitter.split(X, groups=groups)
    train_inds, valid_inds = next(split)
    return (X[train_inds], y[train_inds]), (X[valid_inds], y[valid_inds])


def GroupKfold_train(X, y, participant_id, clf, n_epochs=4, n_splits=1,
                     n_classes=2, valid_prop=0.2, scaling_factor=1e6,
                     filterbank=False):
    """_summary_

    Args:
        windows_dataset (_type_): _description_
        clf (_type_): _description_
        n_epochs (int, optional): _description_. Defaults to 4.
        n_splits (int, optional): _description_. Defaults to 1.
        n_classes (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    if scaling_factor:
        X = X*scaling_factor
    # Perform cross validation
    if n_splits == 'loo':
        kf = LeaveOneGroupOut()
    else:
        kf = GroupKFold(n_splits=n_splits)

    train_val_split = kf.split(X, y, groups=participant_id)

    fold_accuracy = []
    y_pred = np.asarray([99999]*len(y))

    count = 0
    for train_index, test_index in tqdm(train_val_split,
                                        total=kf.get_n_splits(
                                            groups=participant_id),
                                        desc="k-fold"):
        count += 1

        # Copy untrained model
        clf_fold = clone(clf)

        # Split train and test
        if filterbank:
            X_train_fold, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test = y[train_index], y[test_index]

        else:
            X_train_fold, X_test = X[train_index], X[test_index]
            y_train_fold, y_test = y[train_index], y[test_index]

        # If validation set, futrher split train set
        if valid_prop != 0:
            train, valid = group_train_valid_split(X_train_fold, y_train_fold,
                                                   participant_id[train_index],
                                                   valid_prop)

            X_train_fold, y_train_fold = train
            print(valid[0].shape)
            valid_set = Dataset(valid[0], valid[1])
            clf_fold.set_params(
                **{'deeplearn__train_split': predefined_split(valid_set)})

        # Fit
        if n_epochs != 0:
            clf_fold.fit(X_train_fold, y=y_train_fold,
                         deeplearn__epochs=n_epochs)
            y_pred[test_index] = clf_fold.predict(X_test).flatten()

        else:
            clf_fold.fit(X_train_fold, y=y_train_fold)
            y_pred[test_index] = clf_fold.predict(X_test).flatten()

        if n_classes == 1:
            fold_accuracy.append(mean_absolute_error(
                y_test, y_pred[test_index]))
            print('fold MAE: ', fold_accuracy[-1])

        else:
            fold_accuracy.append(balanced_accuracy_score(
                y_test, y_pred[test_index]))
            print('fold accuracy: ', fold_accuracy[-1])
            if valid_prop != 0:
                valid_pred = clf_fold.predict(valid[0])
                print(valid_pred)
                valid_acc = balanced_accuracy_score(valid[1], valid_pred)
                print(valid_acc)

    return fold_accuracy, y, y_pred


def Kfold_train(X, y, clf, n_epochs=4, n_splits=1,
                n_classes=2, valid_prop=0.2, scaling_factor=1e6,
                shuffle=True, random_state=seed):
    """_summary_

    Args:
        windows_dataset (_type_): _description_
        clf (_type_): _description_
        n_epochs (int, optional): _description_. Defaults to 4.
        n_splits (int, optional): _description_. Defaults to 1.
        n_classes (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    if scaling_factor:
        X = X*scaling_factor
    # Perform cross validation
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_val_split = kf.split(X, y)

    fold_accuracy = []
    y_pred = np.asarray([99999]*len(y))

    count = 0
    for train_index, test_index in tqdm(train_val_split,
                                        total=kf.get_n_splits(),
                                        desc="k-fold"):
        count += 1

        # Copy untrained model
        clf_fold = clone(clf)

        # Split train and test
        X_train_fold, X_test = X[train_index], X[test_index]
        y_train_fold, y_test = y[train_index], y[test_index]

        # If validation set, futrher split train set
        if valid_prop != 0:
            X_train_fold,  X_valid, y_train_fold, y_valid = train_test_split(X_train_fold, y_train_fold, test_size=valid_prop,
                                                                             shuffle=True, random_state=random_state)

            valid_set = Dataset(X_valid, y_valid)
            clf_fold.set_params(
                **{'deeplearn__train_split': predefined_split(valid_set)})

        # Fit
        if n_epochs != 0:
            clf_fold.fit(X_train_fold, y=y_train_fold,
                         deeplearn__epochs=n_epochs)
            y_pred[test_index] = clf_fold.predict(X_test).flatten()

        else:
            clf_fold.fit(X_train_fold, y=y_train_fold)
            y_pred[test_index] = clf_fold.predict(X_test).flatten()

        if n_classes == 1:
            fold_accuracy.append(mean_absolute_error(
                y_test, y_pred[test_index]))
            print('fold MAE: ', fold_accuracy[-1])

        else:
            fold_accuracy.append(balanced_accuracy_score(
                y_test, y_pred[test_index]))
            print('fold accuracy: ', fold_accuracy[-1])

            # valid_pred = clf_fold.predict(valid[0])
            # print(valid_pred)
            # valid_acc = balanced_accuracy_score(valid[1], valid_pred)
            # print(valid_acc)

    return fold_accuracy, y, y_pred

####################################################


# Collect all accuracies in a dataframe
all_accuracies = pd.DataFrame(
    index=dataset.metadata['participant_id'].unique())

# ####################################################
# # Within classification for all tasks
# ####################################################

# for p in dataset.metadata['participant_id'].unique():
#     print(p)
#     data_sub = dataset[dataset.metadata['participant_id'] == p]

#     data_sub.metadata['target'] = LabelEncoder().fit_transform(
#         list(data_sub.metadata['task'].values))
#     windows_dataset = create_from_X_y(normalize(data_sub),
#                                       y=data_sub.metadata['target'],
#                                       sfreq=data_sub.info['sfreq'],
#                                       ch_names=data_sub.ch_names,
#                                       drop_last_window=False)
#     windows_dataset.set_description(data_sub.metadata, overwrite=True)

#     n_classes = len(np.unique(windows_dataset.description['target']))

#     # Initiate classifier
#     clf = initiate_clf(model, n_classes)

#     # Train and test
#     fold_accuracy, y_train, y_pred = KFold_train(windows_dataset, clf)

#     # Make condution matrix
#     confusion_mat = confusion_matrix(y_train, y_pred)

#     plot_confusion_matrix(confusion_mat, figsize=(10, 10))
#     plt.title(p + ' 5 tasks classification')
#     plt.show()
#     all_accuracies.loc[p,
#                        'within_classification_5-tasks'] = np.mean(fold_accuracy)


# ####################################################
# # Within classification for 2 active tasks
# ####################################################

# for p in dataset.metadata['participant_id'].unique():

#     data_sub = dataset[dataset.metadata['participant_id'] == p]

#     # Keep only task with 2 active tasks
#     keep = [True if e in ['auditoryrate', 'thermalrate']
#             else False for e in data_sub.metadata['task']]

#     data_sub = data_sub[keep]

#     targets = LabelEncoder().fit_transform(
#         list(data_sub.metadata['task'].values))

#     n_classes = len(np.unique(targets))

#     path_out = opj(derivpath, 'machinelearning', 'within_2tasks')

#     clf = initiate_clf('cov_MDM', n_classes, braindecode=False,
#                        path_out=path_out)

#     fold_accuracy, y_train, y_pred = Kfold_train(X=data_sub.get_data(),
#                                                  y=targets,
#                                                  clf=clf,
#                                                  n_splits=10,
#                                                  valid_prop=0,
#                                                  n_classes=n_classes,
#                                                  n_epochs=0)


# ####################################################
# # Within regression for thermal intensity
# ####################################################
# # TODO remove epochs with too much variation
# # TODO try as a classification task
# for p in dataset.metadata['participant_id'].unique():

#     data_sub = dataset[dataset.metadata['participant_id'] == p]

#     # Keep only tasks with fixed intesity
#     keep = [True if e in ['thermalrate', 'thermal']
#             else False for e in data_sub.metadata['task']]

#     data_sub = data_sub[keep]

#     data_sub.metadata['target'] = data_sub.metadata['intensity']
#     windows_dataset = create_from_X_y(normalize(data_sub),
#                                       y=data_sub.metadata['target'],
#                                       sfreq=data_sub.info['sfreq'],
#                                       ch_names=data_sub.ch_names,
#                                       drop_last_window=False)
#     windows_dataset.set_description(data_sub.metadata, overwrite=True)

#     n_classes = 1

#     # Initiate classifier
#     clf = initiate_clf(model, n_classes)

#     # Train and test
#     fold_accuracy, y_train, y_pred = KFold_train(
#         windows_dataset, clf, n_classes=1, n_epochs=10)

#     sns.regplot(x=y_train, y=y_pred)

#     plt.figure()
#     plt.plot(y_train, label='true')
#     plt.plot(y_pred, label='pred')

#     all_accuracies.loc[p, 'within_regression_intensity'] = np.mean(
#         fold_accuracy)

####################################################
# Between particiapnt classification for all tasks
####################################################

# le = LabelEncoder()
# dataset.metadata['target'] = le.fit_transform(
#     list(dataset.metadata['task'].values))
# windows_dataset = create_from_X_y(normalize(dataset),
#                                   y=dataset.metadata['target'],
#                                   sfreq=dataset.info['sfreq'],
#                                   ch_names=dataset.ch_names,
#                                   drop_last_window=False)
# windows_dataset.set_description(dataset.metadata, overwrite=True)


# n_classes = len(np.unique(windows_dataset.description['target']))


# clf = initiate_clf(model, n_classes)

# fold_accuracy, y_train, y_pred = GroupKfold_train(windows_dataset, clf,
#                                                   n_splits='loo',
#                                                   n_classes=n_classes,
#                                                   n_epochs=4)


# # Make condution matrix
# confusion_mat = confusion_matrix(y_train, y_pred)

# plot_confusion_matrix(confusion_mat, figsize=(10, 10),
#                       class_names=le.classes_)
# plt.tick_params(axis='x', rotation=90)
# plt.tick_params(axis='y', rotation=0)

# plt.title('between - 5 tasks classification')
# plt.show()


# ####################################################
# # Between classification for thermal vs auditory
# ####################################################

# # Keep only task with 4 active tasks
# keep = [True if e in ['thermalrate', 'thermal',
#                       'auditory', 'auditoryrate']
#         else False for e in dataset.metadata['task']]

# dataset_class = dataset[keep]

# targets = np.array([0 if 'thermal' in l else 1 for l in
#                     dataset_class.metadata['task'].values])

# participant_id = dataset_class.metadata['participant_id'].values

# n_classes = len(np.unique(targets))

# path_out = opj(derivpath, 'machinelearning', 'between_4tasks')


# clf = initiate_clf('braindecode_shallow', n_classes, braindecode=True,
#                    path_out=path_out, early_stop_n=20, augmentation=False)

# fold_accuracy, y_train, y_pred = GroupKfold_train(X=dataset_class.get_data(),
#                                                   y=targets,
#                                                   participant_id=participant_id,
#                                                   clf=clf,
#                                                   n_splits=10,
#                                                   valid_prop=0.2,
#                                                   n_classes=n_classes,
#                                                   n_epochs=35)

# # Make confusion matrix
# confusion_mat = confusion_matrix(y_train, y_pred)

# plot_confusion_matrix(confusion_mat, figsize=(10, 10),
#                       class_names=le.classes_)
# plt.tick_params(axis='x', rotation=90)
# plt.tick_params(axis='y', rotation=0)
# plt.show()


# ####################################################
# # Between classification for 4 active tasks
# ####################################################

# # Keep only task with 4 active tasks
# keep = [True if e in ['thermalrate', 'thermal',
#                       'auditory', 'auditoryrate']
#         else False for e in dataset.metadata['task']]

# dataset_class = dataset[keep]

# le = LabelEncoder()
# targets = le.fit_transform(
#     list(dataset_class.metadata['task'].values))

# n_classes = len(np.unique(targets))
# participant_id = dataset_class.metadata['participant_id'].values
# clf = initiate_clf('braindecode_shallow', n_classes,
#                    braindecode=True, early_stop=35)


# clf = initiate_clf('braindecode_shallow', n_classes, braindecode=True,
#                    path_out=path_out, early_stop_n=20, augmentation=False)

# fold_accuracy, y_train, y_pred = GroupKfold_train(X=dataset_class.get_data(),
#                                                   y=targets,
#                                                   participant_id=participant_id,
#                                                   clf=clf,
#                                                   n_splits=2,
#                                                   n_classes=n_classes,
#                                                   n_epochs=35)

# # Make confusion matrix
# confusion_mat = confusion_matrix(y_train, y_pred)

# plot_confusion_matrix(confusion_mat, figsize=(10, 10),
#                       class_names=le.classes_)
# plt.tick_params(axis='x', rotation=90)
# plt.tick_params(axis='y', rotation=0)
# plt.show()


# ######################################################################
# # Between classification for 4 active tasks with frequency bands
# #######################################################################

# frequency_bands = {
#     "alpha": (8.0, 15.0),
#     "beta_low": (15.0, 30.0),
#     "beta_high": (30.0, 45.0),
#     "gamma_low": (45.0, 65.0),
#     "gamma_high": (65.0, 100.0)}

# # Keep only task with 4 active tasks
# keep = [True if e in ['thermalrate', 'thermal',
#                       'auditory', 'auditoryrate']
#         else False for e in dataset.metadata['task']]


# le = LabelEncoder()
# targets = le.fit_transform(
#     list(dataset_class.metadata['task'].values))

# n_classes = len(np.unique(targets))
# participant_id = dataset_class.metadata['participant_id'].values

# covs = np.load(opj(derivpath, 'all_epochs-covs.npy'))
# covs = covs[keep, ::]
# cov_data = pd.DataFrame({band: list(covs[:, ii]) for ii, band in
#                          enumerate(list(frequency_bands.keys()))})

# clf = initiate_clf('filterbank_SVM', 4, braindecode=False,
#                    path_out=path_out)

# fold_accuracy, y_train, y_pred = GroupKfold_train(X=cov_data,
#                                                   y=targets,
#                                                   participant_id=participant_id,
#                                                   clf=clf,
#                                                   valid_prop=0,
#                                                   n_splits=10,
#                                                   n_classes=4,
#                                                   n_epochs=0,
#                                                   filterbank=True)


####################################################
# Between regression for thermal intensity
####################################################
path_out = opj(derivpath, 'machinelearning', 'thermal_intensity_regression')

# Keep only tasks with fixed intesity
keep = [True if e in ['thermal', 'thermalrate']
        else False for e in dataset.metadata['task']]

dataset_class = dataset[keep]

targets = np.asarray(dataset_class.metadata['intensity'])


# Initiate classifier
participant_id = dataset_class.metadata['participant_id'].values


clf = initiate_clf('braindecode_shallow', n_classes=1, braindecode=True,
                   path_out=path_out, early_stop_n=20, augmentation=False)

fold_accuracy, y_train, y_pred = GroupKfold_train(X=dataset_class.get_data(),
                                                  y=np.expand_dims(targets, 1),
                                                  participant_id=participant_id,
                                                  clf=clf,
                                                  n_splits=5,
                                                  n_classes=1,
                                                  n_epochs=35)


sns.regplot(x=y_train, y=y_pred)
plt.figure()
plt.plot(y_train, label='true')
plt.plot(y_pred, label='pred')


all_accuracies.loc['between_regression_intensity'] = np.mean(
    fold_accuracy)
