# Authors: mpcoll
import numpy as np
import pandas as pd
import mne
import torch
from os.path import join as opj
import seaborn as sns
import matplotlib.pyplot as plt

from braindecode import EEGRegressor, EEGClassifier
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.visualization import plot_confusion_matrix
from braindecode.datasets import create_from_X_y
from braindecode.models import Deep4Net, ShallowFBCSPNet
from braindecode.util import set_random_seeds

from skorch.callbacks import LRScheduler
from skorch.helper import SliceDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             confusion_matrix)
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.base import clone
import faulthandler
from tqdm import tqdm

import os

faulthandler.enable()

bidsout = '/Users/mp/data/2023_embcp'
derivpath = opj(bidsout, 'derivatives')
if not os.path.exists(opj(derivpath, 'machinelearning')):
    os.mkdir(opj(derivpath, 'machinelearning'))

# Hide data loading info
mne.set_log_level('WARNING')


####################################################
# Load data
####################################################

dataset = mne.read_epochs(opj(derivpath, 'all_epochs-epo.fif'))

# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

# Standardize


def normalize(dataset):
    X = dataset.get_data()
    X = np.multiply(X, factor)
    print('standardizing data...')
    for i in tqdm(range(X.shape[0])):
        X[i, :, :] = exponential_moving_standardize(X[i, :, :],
                                                    factor_new=factor_new,
                                                    init_block_size=init_block_size)
    return X

####################################################
# set up model and parameters
####################################################


# Device
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
# For macbook M1
if torch.backends.mps.is_available():
    print("Using MPS")
    device = 'mps'

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

# Get data info
n_chans = dataset.get_data().shape[1]
input_window_samples = dataset.get_data().shape[2]


batch_size = 32
n_epochs = 10

model = 'shallow'

# TODO add other non deep models (LASSO, SVM, RF)


def initiate_clf_sklearn(model, n_classes, n_chans=n_chans,
                         input_window_samples=input_window_samples,
                         batch_size=batch_size, device=device,
                         braindecode=True):
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

    if braindecode:
        if model == 'deep':
            lr = 0.01
            weight_decay = 0.0005
            # Model
            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
        elif model == 'shallow':
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

        if n_classes == 1:  # If regression
            # Remove softmax layer
            new_model = torch.nn.Sequential()
            for name, module_ in model.named_children():
                if "softmax" in name:
                    continue
                new_model.add_module(name, module_)
            model = new_model

            clf = EEGRegressor(
                model,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                train_split=None,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "neg_root_mean_squared_error",
                    # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
                    ("lr_scheduler", LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
            )
        else:  # If classification
            clf = EEGClassifier(
                model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                train_split=None,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "accuracy",
                    # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
                    ("lr_scheduler", LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
            )

    return clf


def initiate_clf(model, n_classes, n_chans=n_chans,
                 input_window_samples=input_window_samples,
                 batch_size=batch_size, device=device,
                 braindecode=True):
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

    if braindecode:
        if model == 'deep':
            lr = 0.01
            weight_decay = 0.0005
            # Model
            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
        elif model == 'shallow':
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

        if n_classes == 1:  # If regression
            # Remove softmax layer
            new_model = torch.nn.Sequential()
            for name, module_ in model.named_children():
                if "softmax" in name:
                    continue
                new_model.add_module(name, module_)
            model = new_model

            clf = EEGRegressor(
                model,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                train_split=None,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "neg_root_mean_squared_error",
                    # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
                    ("lr_scheduler", LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
            )
        else:  # If classification
            clf = EEGClassifier(
                model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                train_split=None,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "accuracy",
                    # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
                    ("lr_scheduler", LRScheduler(
                        'CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
            )

    return clf


def KFold_train(windows_dataset, clf, n_epochs=4, n_splits=5,
                shuffle=True, n_classes=2):
    """_summary_

    Args:
        windows_dataset (_type_): _description_
        clf (_type_): _description_
        n_epochs (int, optional): _description_. Defaults to 4.
        n_splits (int, optional): _description_. Defaults to 5.
        shuffle (bool, optional): _description_. Defaults to True.
        n_classes (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    # Put in skorch for cross validation
    X_train = SliceDataset(windows_dataset, idx=0)
    y_train = np.array([y for y in SliceDataset(windows_dataset, idx=1)])

    # Perform cross validation
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    fold_accuracy = []
    y_pred = np.asarray([99999]*len(y_train))
    count = 0
    for train_index, test_index in tqdm(kf.split(X_train, y_train),
                                        total=kf.get_n_splits(), desc="k-fold"):
        count += 1

        # Copy untrained model
        clf_fold = clone(clf)

        # Get data
        X_train_fold, X_test = X_train[train_index], X_train[test_index]
        y_train_fold, y_test = y_train[train_index], y_train[test_index]

        # Fit
        clf_fold.fit(X_train_fold, y=y_train_fold, epochs=n_epochs)

        # Test
        y_pred[test_index] = clf_fold.predict(X_test).flatten()

        if n_classes == 1:
            fold_accuracy.append(mean_absolute_error(
                y_test, y_pred[test_index]))
            print('fold MAE: ', fold_accuracy[-1])

        else:
            fold_accuracy.append(balanced_accuracy_score(
                y_test, y_pred[test_index]))
            print('fold accuracy: ', fold_accuracy[-1])
    return fold_accuracy, y_train, y_pred


def GroupKfold_train(windows_dataset, clf, n_epochs=4, n_splits=1,
                     n_classes=2):
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

    # Put in skorch for cross validation
    X_train = SliceDataset(windows_dataset, idx=0)
    y_train = np.array([y for y in SliceDataset(windows_dataset, idx=1)])
    participant_id = windows_dataset.description['participant_id']

    # Perform cross validation
    if n_splits == 'loo':
        kf = LeaveOneGroupOut()
    else:
        kf = GroupKFold()

    train_val_split = kf.split(X_train, y_train, groups=participant_id)

    fold_accuracy = []
    y_pred = np.asarray([99999]*len(y_train))

    count = 0
    for train_index, test_index in tqdm(train_val_split,
                                        total=kf.get_n_splits(
                                            groups=participant_id),
                                        desc="k-fold"):
        count += 1

        # Copy untrained model
        clf_fold = clone(clf)

        # Get data
        X_train_fold, X_test = X_train[train_index], X_train[test_index]
        y_train_fold, y_test = y_train[train_index], y_train[test_index]

        # Fit
        clf_fold.fit(X_train_fold, y=y_train_fold, epochs=n_epochs)

        # Test
        y_pred[test_index] = clf_fold.predict(X_test).flatten()

        if n_classes == 1:
            fold_accuracy.append(mean_absolute_error(
                y_test, y_pred[test_index]))
            print('fold MAE: ', fold_accuracy[-1])

        else:
            fold_accuracy.append(balanced_accuracy_score(
                y_test, y_pred[test_index]))
            print('fold accuracy: ', fold_accuracy[-1])

    return fold_accuracy, y_train, y_pred

####################################################


# Collect all accuracies in a dataframe
all_accuracies = pd.DataFrame(
    index=dataset.metadata['participant_id'].unique())

# ####################################################
# # Within classification for all tasks
# ####################################################

for p in dataset.metadata['participant_id'].unique():
    print(p)
    data_sub = dataset[dataset.metadata['participant_id'] == p]

    data_sub.metadata['target'] = LabelEncoder().fit_transform(
        list(data_sub.metadata['task'].values))
    windows_dataset = create_from_X_y(normalize(data_sub),
                                      y=data_sub.metadata['target'],
                                      sfreq=data_sub.info['sfreq'],
                                      ch_names=data_sub.ch_names,
                                      drop_last_window=False)
    windows_dataset.set_description(data_sub.metadata, overwrite=True)

    n_classes = len(np.unique(windows_dataset.description['target']))

    # Initiate classifier
    clf = initiate_clf(model, n_classes)

    # Train and test
    fold_accuracy, y_train, y_pred = KFold_train(windows_dataset, clf)

    # Make condution matrix
    confusion_mat = confusion_matrix(y_train, y_pred)

    plot_confusion_matrix(confusion_mat, figsize=(10, 10))
    plt.title(p + ' 5 tasks classification')
    plt.show()
    all_accuracies.loc[p,
                       'within_classification_5-tasks'] = np.mean(fold_accuracy)


# ####################################################
# # Within classification for 4 active tasks
# ####################################################

for p in dataset.metadata['participant_id'].unique():

    data_sub = dataset[dataset.metadata['participant_id'] == p]

    # Keep only task with 2 active tasks
    keep = [True if e in ['auditoryrate', 'thermalrate']
            else False for e in data_sub.metadata['task']]

    data_sub = data_sub[keep]

    data_sub.metadata['target'] = LabelEncoder().fit_transform(
        list(data_sub.metadata['task'].values))
    windows_dataset = create_from_X_y(normalize(data_sub),
                                      y=data_sub.metadata['target'],
                                      sfreq=data_sub.info['sfreq'],
                                      ch_names=data_sub.ch_names,
                                      drop_last_window=False)
    windows_dataset.set_description(data_sub.metadata, overwrite=True)

    n_classes = len(np.unique(windows_dataset.description['target']))
    # Initiate classifier
    clf = initiate_clf(model, n_classes)

    # Train and test
    fold_accuracy, y_train, y_pred = KFold_train(windows_dataset, clf,
                                                 n_epochs=6)

    # Make condution matrix
    confusion_mat = confusion_matrix(y_train, y_pred)

    plot_confusion_matrix(confusion_mat, figsize=(10, 10))
    plt.title(p + ' 9 tasks classification')
    plt.show()
    all_accuracies.loc[p,
                       'within_classification_4-tasks'] = np.mean(fold_accuracy)


####################################################
# Within regression for thermal intensity
####################################################
# TODO remove epochs with too much variation
# TODO try as a classification task
for p in dataset.metadata['participant_id'].unique():

    data_sub = dataset[dataset.metadata['participant_id'] == p]

    # Keep only tasks with fixed intesity
    keep = [True if e in ['thermalrate', 'thermal']
            else False for e in data_sub.metadata['task']]

    data_sub = data_sub[keep]

    data_sub.metadata['target'] = data_sub.metadata['intensity']
    windows_dataset = create_from_X_y(normalize(data_sub),
                                      y=data_sub.metadata['target'],
                                      sfreq=data_sub.info['sfreq'],
                                      ch_names=data_sub.ch_names,
                                      drop_last_window=False)
    windows_dataset.set_description(data_sub.metadata, overwrite=True)

    n_classes = 1

    # Initiate classifier
    clf = initiate_clf(model, n_classes)

    # Train and test
    fold_accuracy, y_train, y_pred = KFold_train(
        windows_dataset, clf, n_classes=1, n_epochs=10)

    sns.regplot(x=y_train, y=y_pred)

    plt.figure()
    plt.plot(y_train, label='true')
    plt.plot(y_pred, label='pred')

    all_accuracies.loc[p, 'within_regression_intensity'] = np.mean(
        fold_accuracy)

####################################################
# Between particiapnt classification for all tasks
####################################################

le = LabelEncoder()
dataset.metadata['target'] = le.fit_transform(
    list(dataset.metadata['task'].values))
windows_dataset = create_from_X_y(normalize(dataset),
                                  y=dataset.metadata['target'],
                                  sfreq=dataset.info['sfreq'],
                                  ch_names=dataset.ch_names,
                                  drop_last_window=False)
windows_dataset.set_description(dataset.metadata, overwrite=True)


n_classes = len(np.unique(windows_dataset.description['target']))


clf = initiate_clf(model, n_classes)

fold_accuracy, y_train, y_pred = GroupKfold_train(windows_dataset, clf,
                                                  n_splits='loo',
                                                  n_classes=n_classes,
                                                  n_epochs=4)


# Make condution matrix
confusion_mat = confusion_matrix(y_train, y_pred)

plot_confusion_matrix(confusion_mat, figsize=(10, 10),
                      class_names=le.classes_)
plt.tick_params(axis='x', rotation=90)
plt.tick_params(axis='y', rotation=0)

plt.title('between - 5 tasks classification')
plt.show()


# ####################################################
# # Between classification for 4 active tasks
# ####################################################

# Keep only task with 4 active tasks
keep = [True if e in ['thermalactive', 'audioactive',
                      'chemicalactive', 'disgustactive']
        else False for e in dataset.metadata['task']]

dataset_class = dataset[keep]

le = LabelEncoder()
dataset_class.metadata['target'] = le.fit_transform(
    list(dataset_class.metadata['task'].values))

windows_dataset = create_from_X_y(normalize(dataset_class),
                                  y=dataset_class.metadata['target'],
                                  sfreq=dataset_class.info['sfreq'],
                                  ch_names=dataset_class.ch_names,
                                  drop_last_window=False)
windows_dataset.set_description(dataset_class.metadata, overwrite=True)

n_classes = len(np.unique(windows_dataset.description['target']))

clf = initiate_clf(model, n_classes)

fold_accuracy, y_train, y_pred = GroupKfold_train(windows_dataset, clf,
                                                  n_splits='loo',
                                                  n_classes=n_classes,
                                                  n_epochs=4)

# Make confusion matrix
confusion_mat = confusion_matrix(y_train, y_pred)

plot_confusion_matrix(confusion_mat, figsize=(10, 10),
                      class_names=le.classes_)
plt.tick_params(axis='x', rotation=90)
plt.tick_params(axis='y', rotation=0)
plt.show()


# ####################################################
# # Between regression for thermal intensity
# ####################################################

# Keep only tasks with fixed intesity
keep = [True if e in ['thermal', 'thermalrate']
        else False for e in dataset.metadata['task']]

data_class = dataset[keep]

data_class.metadata['target'] = data_class.metadata['intensity']
windows_dataset = create_from_X_y(normalize(data_class),
                                  y=data_class.metadata['target'],
                                  sfreq=data_class.info['sfreq'],
                                  ch_names=data_class.ch_names,
                                  drop_last_window=False)
windows_dataset.set_description(data_class.metadata, overwrite=True)

n_classes = 1

# Initiate classifier
clf = initiate_clf(model, n_classes)

# Train and test
fold_accuracy, y_train, y_pred = GroupKfold_train(windows_dataset, clf,
                                                  n_splits='loo',
                                                  n_classes=1,
                                                  n_epochs=4)

sns.regplot(x=y_train, y=y_pred)
plt.figure()
plt.plot(y_train, label='true')
plt.plot(y_pred, label='pred')


all_accuracies.loc['between_regression_intensity'] = np.mean(
    fold_accuracy)
