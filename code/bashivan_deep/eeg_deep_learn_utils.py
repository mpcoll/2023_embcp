# data path

import pandas as pd
import numpy as np
from os.path import join as opj
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
sys.path.append('deeplearn')
from models_3d import AlexNet3D_Dropout, AlexNet3D_Deeper_Dropout, AlexNet3D_Dropout_Regression, resnet50, resnet34, resnet152
# from models_3d import densenet121_3D_DropOut, densenet161_3D_DropOut
# from models_3d import generate_densenet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, explained_variance_score,
                             mean_squared_error, r2_score, log_loss, roc_auc_score)
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import time
import queue
import threading
import matplotlib.pyplot as plt
from nilearn import image
from sklearn.svm import SVC, LinearSVC
from sklearn.random_projection import GaussianRandomProjection
from hypopt import GridSearch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from nilearn import image
from nilearn import plotting
from scipy.stats import pearsonr, zscore
from itertools import combinations
from ray import tune
import mne
import math
from models_eeglearn import MaxCNN, Mix, TempCNN, LSTM
mne.set_log_level('ERROR')


def train_test_split_group(df, test_prop=0.3,
                           group_id='subject_id', random_state=7):
    """Split a df in three sets, train, validation and test according to
       groups (i.e. participants)

    Args:
        df (pandas df): data frame to split
        test_prop (float, optional): Proportion of observations to use as validation/test. 
                                     Note that will be split in two for val/test. Defaults to 0.4.
        group_id (str, optional): Column with group label. Defaults to 'subject_id'.
        random_state (int, optional): random state to replicate. Defaults to 7.

    Returns:
        train, validation and test data frames
    """
    # Split main df in two
    train_ids, test_ids = next(GroupShuffleSplit(n_splits=2,
                                        test_size=test_prop,
                                        random_state=random_state).split(df,
                                                            groups=df[group_id]))
    df_tr = df.iloc[train_ids]
    df_te = df.iloc[test_ids]
    # REsplit test in validation and test
    va_ids, test_ids = next(GroupShuffleSplit(n_splits=2,
                                        test_size=0.5,
                                        random_state=random_state).split(df_te,
                                                            groups=df_te[group_id]))
    df_va = df_te.iloc[va_ids]
    df_te = df_te.iloc[test_ids]

    return df_tr, df_va, df_te


def to_gpu(batch):
    """Push data to gpu

    Args:
        batch (pytorch batch): batch of data

    Returns:
        inputs, labels as tensors on the GPU
    """
    inputs = Variable(batch[0].cuda(non_blocking=True))
    labels = Variable(batch[1].cuda(non_blocking=True))
    return inputs, labels


def _mapper_loop(func, input_iter, result_q, error_q, done_event):
    """Map a function?

    Args:
        func ([type]): [description]
        input_iter ([type]): [description]
        result_q ([type]): [description]
        error_q ([type]): [description]
        done_event ([type]): [description]
    """
    try:
        for x in input_iter:
            result = func(x)
            result_q.put(result)
    except BaseException:
        error_q.put(sys.exc_info())
    finally:
        done_event.set()


def read_tf_idx(df, idx, scorename, regression, transform, scale_param=None):
    """Fetch an X, y observation from the EEG dataset using idx

    Args:
        df (dataframe): pandas df with score in a column and img path in another
        idx (int): index of the observation to fetch
        scorename (string): column for the scorename in the df
        regression (bool): Wheter to return the score as float for regression
        transform (string): 'zscore' to standardisize, minmax to get minmanx, None for raw data

    Returns:
        X : numpy array with the MRI data
        y : score as float or int depending on regression arg
    """

    X, y = [], []
    fN = df['filepath'].iloc[idx]  # Path to image
    la = df[scorename].iloc[idx]

    eeg = np.load(fN).squeeze()

    if transform == 'minmax':
        eeg = (eeg - eeg.min()) / (eeg.max() - eeg.min()) # Normalize
    if transform == 'zscore':
        eeg = (eeg - eeg.mean())/eeg.std() # Standardize
    if transform == 'standardscaler': # Standardize with mean/sd calcualted elsewhere
        eeg = (eeg - scale_param[0])/scale_param[1]

    # Add sample dimension
    # eeg = np.reshape(eeg, (1, eeg.shape[0], eeg.shape[1], eeg.shape[2], eeg.shape[3]))

    eeg = np.expand_dims(eeg, 0)

    X = np.float32(eeg)

    if regression:
        y = np.array(np.float32(la))
    else:
        y = np.array(np.int(la))
    return X, y


def read_tf(df, scorename, regression, transform='zscore', scale_param=None):
    X,y = [],[]
    for sN in np.arange(df.shape[0]):
        fN = df['filepath'].iloc[sN]
        la = df[scorename].iloc[sN]

        im = np.load(fN).squeeze()
        if transform == 'minmax':
            im = (im - im.min()) / (im.max() - im.min()) # Normalize
        if transform == 'zscore':
            im = (im - im.mean())/im.std() # Standardize
        im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        X.append(im)
        y.append(la)
    X = np.float32(np.array(X))
    if regression:
        y = np.array(y).astype(float)
    else:
        y = np.array(y).astype(int)

    # print('X: ',X.shape,' y: ',y.shape)    
    return X, y



class EEGdataset(Dataset):

    def __init__(self, df, scorename, regression=False, transform='zscore',
                 scale_param=None):
        """Initialize an MRI dataset

        Args:
            df (dataframe): pandas df with score in a column and img path in another
            scorename (string): column for the scorename in the df
            regression (bool): Wheter to return the score as float
                               for regression. Defaults to False.
        """
        self.df = df
        self.scorename = scorename
        self.regression = regression
        self.transform = transform
        self.scale_param = scale_param


    def __len__(self):
        """Helper function to get size"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Helper function for dataloader"""
        X, y = read_tf_idx(self.df, idx, self.scorename, self.regression,
                       self.transform, self.scale_param)
        return [X, y]


def prefetch_map(func, input_iter, prefetch=1, check_interval=5):
    """
    Map a function (func) on a iterable (input_iter), but
    prefetch input values and map them asyncronously as output
    values are consumed.
    prefetch: int, the number of values to prefetch asyncronously
    check_interval: int, the number of seconds to block when waiting
                    for output values.
    """
    result_q = queue.Queue(prefetch)
    error_q = queue.Queue(1)
    done_event = threading.Event()

    mapper_thread = threading.Thread(target=_mapper_loop, args=(func, input_iter, result_q, error_q, done_event))
    mapper_thread.daemon = True
    mapper_thread.start()

    while not (done_event.is_set() and result_q.empty()):
        try:
            result = result_q.get(timeout=check_interval)
        except queue.Empty:
            continue

        yield result

    if error_q.full():
        raise error_q.get()[1]


def train(dataloader, net, optimizer, criterion, cuda_avl, num_prefetch=1,
          nbatches=1):
    """Train the network

    Args:
        dataloader (torch dataloader): dataloader for training
        net (pytorch model): moderl to use
        optimizer (torch optim): optimizer to use
        criterion (torch criterion): criterion to use
        cuda_avl (book): Whether to use cuda
        num_prefetch (int, optional): Number of observations to prefecth in parrallel. Defaults to 16.

    Returns:
        loss [tensor]: training loss
    """
    # Debug if true
    # torch.autograd.set_detect_anomaly(True)
    if cuda_avl:
        dataloader = prefetch_map(to_gpu, dataloader, prefetch=num_prefetch)
    net.train()
    for i, data in enumerate(tqdm(dataloader, total=nbatches), 0):

        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        # print(inputs.shape)
        outputs = net(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    return loss


def test(dataloader, net, cuda_avl, criterion):
    """Predict using a model
    Args:
        dataloader (torch dataloader): dataloader for test
        net (pytorch model): moderl to use
        cuda_avl (book): Whether to use cuda
        criterion (torch criterion): criterion to use

    Returns:
        true labels, predicted labels, test loss
    """
    net.eval()
    y_pred = np.array([])
    y_true = np.array([])
    y_proba = np.array([])
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if cuda_avl:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_loss = criterion(outputs, labels)    
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
            y_true = np.concatenate((y_true, labels.data.cpu().numpy()))
    return y_true, y_pred, test_loss



def test_reg(dataloader, net, cuda_avl):
    net.eval()
    y_pred = np.array([])
    y_true = np.array([])
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)       
        outputs = net(inputs)
        y_pred = np.concatenate((y_pred, outputs.squeeze().data.cpu().numpy()))
        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))
    return y_true, y_pred


def generate_validation_model(df_tr, df_val, scorename, model,
                              num_classes,
                              n_epochs=30,
                              lr=0.001,
                              early_stopping=True,
                              early_plateau=10,
                              class_weights=None,
                              linear_size=4096,
                              imbalance_sampler=False,
                              x_transform='zscore',
                              outpath='/data/models/model_name',
                              num_workers=0, batch_size=32):

    # Create model path if not exists
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Get the model
    regression = False
    if model == 'AlexNet3D':
        net = AlexNet3D(num_classes=num_classes)
    elif model == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=num_classes)
    elif model == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=num_classes)
    elif model == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=num_classes)
    elif model == 'ResNet_50':
        net = resnet50(num_classes=num_classes, linear_size=linear_size)
    elif model == 'ResNet_34':
        net = resnet34(num_classes=num_classes, linear_size=linear_size)
    elif model == 'DenseNet':
        net = densenet(num_classes=num_classes)
    elif model == 'MaxCNN':
        net = MaxCNN(n_classes=num_classes)
    elif model == 'CNN+RNN':
        net = Mix(n_classes=num_classes)
    elif model == 'TempCNN':
        net = TempCNN(n_classes=num_classes)
    else:
        print('Check model type')

    # Choose the criterion
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    # Get the training and validation data
    trainset = EEGdataset(df_tr, scorename, regression=regression,
                          transform=x_transform)

    if imbalance_sampler:
        sampler = WeightSampler(df_tr[scorename])
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                sampler=sampler,
                                num_workers=num_workers, drop_last=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True)



    # Scale using training mean and sd
    if x_transform == 'standardscaler':

        # Load raw data
        trainset = EEGdataset(df_tr, scorename, regression=regression,
                              transform=None)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True)

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        print('Calculating trainset scaling parameters')
        for batch in tqdm(trainloader, total=len(trainset)//batch_size):
            batch = batch[0]
            batch_flat = batch.reshape((batch.shape[0], np.prod(batch.shape[1:])))

            scaler.partial_fit(batch_flat)

        scale_param = [scaler.mean_.reshape((batch.shape[1:])).squeeze(), np.sqrt(scaler.var_).reshape((batch.shape[1:])).squeeze()]
                # Load raw data
        trainset = EEGdataset(df_tr, scorename, regression=regression,
                              transform=x_transform, scale_param=scale_param)
    else:
        scale_param = None


    # IF standardize across

    validset = EEGdataset(df_val, scorename, regression=regression,
                          transform=x_transform, scale_param=scale_param)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)


    # Use GPU is cuda available
    cuda_avl = torch.cuda.is_available()

    cuda_avl = True
    if cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net,
                                    device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Load optimizer (use ADAM for now)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=0.5, patience=5, verbose=True)

    # Early stopping details
    max_val_acc = 0
    epochs_no_improve = 0
    valid_acc = 0


    history = pd.DataFrame(columns=['scorename','iter','epoch',
                                    'val_acc','bal_val_acc','tr_loss',
                                    'val_loss'])


    # Baseline
    for epoch in range(n_epochs):
        # Train
        print('Training epoch ' + str(epoch))
        t0 = time.time()
        loss = train(trainloader, net, optimizer, criterion, cuda_avl,
                     nbatches=len(trainset)//batch_size)
        loss = loss.data.cpu().numpy()
        # Do not get training acc (not useful?)
        # y_true, y_pred, _ = test(trainloader, net, cuda_avl, criterion)
        # train_acc = accuracy_score(y_true, y_pred)
        # bal_train_acc = balanced_accuracy_score(y_true, y_pred)

        # Validate
        # print('Validating')
        y_true, y_pred, val_loss = test(validloader, net, cuda_avl, criterion)
        val_loss = val_loss.data.cpu().numpy()
        valid_acc = accuracy_score(y_true, y_pred)
        bal_valid_acc = balanced_accuracy_score(y_true, y_pred)
        history.loc[epoch] = [scorename, 0, epoch, valid_acc, bal_valid_acc,
                              loss, val_loss]

        history.to_csv(opj(outpath, 'train_history.csv'), index=False)

        print ('scorename: ' + scorename + '_Iter '+str(0)+' Epoch '+str(epoch)
                +' Val. Acc.: '+str(np.round(valid_acc, 2))
                +' Bal. Val. Acc.: '
                +str(np.round(bal_valid_acc, 2))+' Tr. Loss ' +str(np.round(loss, 2))
                + ' Val. Loss ' + str(np.round(val_loss, 2)))
        print('{} seconds'.format(time.time() - t0))

        if early_stopping:
            # If the validation accuracy is at a maximum
            if bal_valid_acc > max_val_acc:
                # Save the model
                torch.save(net.state_dict(), open(opj(outpath,
                                                      'model_state_dict.pt'),'wb'))
                print('Best score, saving model.')
                epochs_no_improve = 0
                max_val_acc = bal_valid_acc
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == early_plateau:
                    print('Early stopping!')
                    fig = plt.figure()
                    plt.plot(history['tr_loss'], label='Training loss')
                    plt.plot(history['val_loss'], label='Validation loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.title('Running loss at early stop '
                              + str(epoch) + ' epochs')
                    fig.savefig(opj(outpath, 'train_history_early.png'))
                    return history, max_val_acc
        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)

    fig = plt.figure()
    plt.plot(history['tr_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Running loss no early stop ' + str(n_epochs) + ' epochs')
    fig.savefig(opj(outpath, 'train_history_full.png'))

    return history, max_val_acc



def generate_validation_model_regression(df_tr, df_val, scorename, model,
                                         num_classes,
                                         n_epochs=30,
                                         lr=0.001,
                                         early_stopping=True,
                                         x_transform='zscore',
                                         early_plateau=10,
                                         linear_size=2048,
                                         cuda_avl=True,
                                         imbalance_sampler=False,
                                         outpath='/data/models/model_name',
                                         num_workers=0, batch_size=32):

    # Create model path if not exists
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if model == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=1)
    elif model == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=1)
    elif model == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=1)
    elif model == 'ResNet_152':
        net = resnet152(num_classes=1, linear_size=linear_size)
    elif model == 'ResNet_50':
        net = resnet50(num_classes=1, linear_size=linear_size)
    elif model == 'ResNet_34':
        net = resnet34(num_classes=1, linear_size=linear_size)
    elif model == 'DenseNet_121_dropout':
        net = densenet121_3D_DropOut(num_classes=1)
    elif model == 'DenseNet_161_dropout':
        net = densenet161_3D_DropOut(num_classes=1)
    elif model == 'LSTM':
        net = LSTM(n_classes=1)
    else:
        print('Check model')

    criterion = nn.MSELoss()

    # Get the training and validation data
    trainset = EEGdataset(df_tr, scorename, regression=True, transform=x_transform)

    if imbalance_sampler:
        sampler = WeightSampler(df_tr[scorename])
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                sampler=sampler,
                                num_workers=num_workers, drop_last=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True)

    validset = EEGdataset(df_val, scorename, regression=True, transform=x_transform)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)

    # Use GPU is cuda available

    if cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net,
                                    device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Load optimizer (use ADAM for now)
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=3, verbose=True)

    # Early stopping details
    min_val_mse = math.inf
    epochs_no_improve = 0
    valid_mae = math.inf


    history = pd.DataFrame(columns=['scorename','epoch',
                                    'val_mae','val_ev','val_r2', 'val_r',
                                    'tr_loss (mse)', 'val_loss (mse)'])

    # Baseline
    for epoch in range(n_epochs):
        # Train
        print('Training epoch ' + str(epoch))
        t0 = time.time()
        loss = train(trainloader, net, optimizer, criterion, cuda_avl, len(trainset))
        loss = loss.data.cpu().numpy()
        # Do not get training acc (not useful?)
        # y_true, y_pred, _ = test(trainloader, net, cuda_avl, criterion)
        # train_acc = accuracy_score(y_true, y_pred)
        # bal_train_acc = balanced_accuracy_score(y_true, y_pred)

        # Validate
        # print('Validating')

        y_true, y_pred = test_reg(validloader, net, cuda_avl)
        valid_mae = mean_absolute_error(y_true, y_pred)
        valid_ev = explained_variance_score(y_true, y_pred)
        val_loss = mean_squared_error(y_true, y_pred)
        valid_r2 = r2_score(y_true, y_pred)
        valid_r = pearsonr(y_true, y_pred)[0]

        history.loc[epoch] = [scorename, epoch, valid_mae, valid_ev,
                              valid_r2, valid_r, loss, val_loss]
        history.to_csv(opj(outpath, 'train_history.csv'), index=False)

        print ('scorename: ' + scorename +' Epoch '+str(epoch)
                +' Tr. MSE.: '+ str(np.round(loss, 2))
                +' Tr. RMSE: ' + str(np.round(np.sqrt(loss), 2))
                + ' Val. MSE: ' + str(np.round(val_loss, 2))
                + ' Val. RMSE: ' + str(np.round(np.sqrt(val_loss), 2))
                + ' Val. MAE ' + str(np.round(valid_mae, 2))
                + ' Val. R2 ' + str(np.round(valid_r2, 2))
                + ' Val. r '  + str(np.round(valid_r, 2)))
        print('{} seconds'.format(time.time() - t0))

        if early_stopping:
            # If the validation accuracy is at a maximum
            if val_loss < min_val_mse:
                # Save the model if there is improvement
                torch.save(net.state_dict(), open(opj(outpath,
                                                      'model_state_dict.pt'),
                                                  'wb'))
                epochs_no_improve = 0
                min_val_mse = val_loss
                print('Best score, saving model.')
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == early_plateau:
                    print('Early stopping!')
                    fig = plt.figure()
                    plt.plot(history['tr_loss (mse)'], label='Training loss')
                    plt.plot(history['val_loss (mse)'], label='Validation loss')
                    plt.xlabel('Epochs')
                    plt.legend()
                    plt.ylabel('Mean squared error')
                    plt.title('Running loss at early stop '
                              + str(epoch) + ' epochs')
                    fig.savefig(opj(outpath, 'train_history_early.png'))
                    return history, min_val_mse
        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_mae)

    fig = plt.figure()
    plt.plot(history['tr_loss (mse)'], label='Training loss')
    plt.plot(history['val_loss (mse)'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Mean squared error')
    plt.title('Running loss no early stop ' + str(epoch) + ' epochs')
    fig.savefig(opj(outpath, 'train_history_full.png'))

    return history, min_val_mse

def load_net_weights2(net, weights_filename):
    state_dict = torch.load(weights_filename,  map_location=lambda storage, loc: storage)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)
    return net




def returnModel(model, outpath, num_classes):
    # Load validated model
    net=0

    if model == 'AlexNet3D':
        net = AlexNet3D(num_classes=num_classes)
    elif model == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=num_classes)
    elif model == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=num_classes)
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = 0

    net = load_net_weights2(model, opj(outpath, 'model_state_dict.pt'))

    return net

# TODO update VARIABLES to tensor
# TODO Crop images for memory



###################################################################
# Visualisation of relevant features (see https://github.com/jrieke/cnn-interpretability)
###################################################################

def sensitivity_analysis(model, image_tensor, target_class=None,
                         postprocess='abs', apply_softmax=True,
                         cuda=False, verbose=False, taskmode='clx'):
    # Adapted from http://arxiv.org/abs/1808.02874
    # https://github.com/jrieke/cnn-interpretability

    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")

    # Forward pass.
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor        print(image_tensor.shape)

    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch

    output = model(X)[0]

    if apply_softmax:
        output = F.softmax(output)
    #print(output.shape)

    # Backward pass.
    model.zero_grad()

    if taskmode == 'reg':
        output.backward(gradient=output)
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data[0]
        if verbose: print('Image was classified as', output_class,
                          'with probability', output.max(1)[0].data[0])
        one_hot_output = torch.zeros(output.size())
        if target_class is None:
            one_hot_output[0, output_class] = 1
        else:
            one_hot_output[0, target_class] = 1
        if cuda:
            one_hot_output = one_hot_output.cuda()
        output.backward(gradient=one_hot_output)

    relevance_map = X.grad.data[0].cpu().numpy()

    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map


def area_occlusion(model, image_tensor, area_masks, target_class=None,
                   occlusion_value=0, apply_softmax=True, cuda=False,
                   verbose=False, taskmode='clx'):

    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor    

    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))[0]

    if apply_softmax:
        output = F.softmax(output)

    if taskmode == 'reg':
        unoccluded_prob = output.data
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data.cpu().numpy()[0]    

        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])

        if target_class is None:
            target_class = output_class
        unoccluded_prob = output.data[0, target_class]

    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()

    for area_mask in area_masks:

        area_mask = torch.FloatTensor(area_mask)

        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)

        output = model(Variable(image_tensor_occluded[None], requires_grad=False))[0]
        if apply_softmax:
            output = F.softmax(output)

        if taskmode == 'reg':
            occluded_prob = output.data
        elif taskmode == 'clx':
            occluded_prob = output.data[0, target_class]

        ins = area_mask.view(image_tensor.shape) == 1
        ins = ins.squeeze()
        relevance_map[ins] = (unoccluded_prob - occluded_prob)

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map


def run_saliency(odir, itrpm, images, net, area_masks, iter_, scorename, taskM):
    for nSub in np.arange(images.shape[0]): 
        print(nSub)
        fname = odir + itrpm + '_' + scorename + '_iter_' + str(iter_) + '_nSub_' + str(nSub) + '.nii'    
        if itrpm == 'AO':
            interpretation_method = area_occlusion
            sal_im = interpretation_method(net, images[nSub], area_masks, occlusion_value=0, apply_softmax=False, cuda=False, verbose=False,taskmode=taskM) 
        elif itrpm == 'BP':
            interpretation_method = sensitivity_analysis
            sal_im = interpretation_method(net, images[nSub], apply_softmax=False, cuda=False, verbose=False, taskmode=taskM)
        else:
            print('Verify interpretation method')
        nib.save(nib.Nifti1Image(sal_im.squeeze() , np.eye(4)), fname)


def forward_pass(X_te,net):
    # used to overcome memory constraints
    net.eval()
    outs30=[]
    ims = X_te.shape
    for n in range(0,ims[0]):
        im = X_te[n].reshape(1, 1, ims[2],ims[3],ims[4])          
        temp = net(im)
        temp0 = temp[0]
        aa = nn.functional.softmax(temp0,dim=-1).data.cpu().numpy().squeeze()
        outs30.append(aa)
    probs = np.vstack((outs30))
    return probs


def evaluate_test_accuracy(model, outpath, df_te, num_classes, scorename,
                           classify=True, x_transform=None, linear_size=None,
                           scale_param=None):

    # Load validated model
    net=0
    regression=False
    if model == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=num_classes)
    elif model == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=num_classes)
    elif model == 'ResNet_34':
        net = resnet34(num_classes=num_classes, linear_size=linear_size)
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, opj(outpath, 'model_state_dict.pt'))

    X_te, y_te = read_tf(df_te, scorename, regression=regression,
                         transform=x_transform, scale_param=scale_param)

    X_te = Variable(torch.from_numpy(X_te))

    # save accuracy as csv file
    outs = pd.DataFrame(columns=['bal_acc_te'])
    probs = forward_pass(X_te,net)

    p_y_te = probs.argmax(1)
    acc_te = balanced_accuracy_score(y_te, p_y_te)

    df_te['Y_pred'] = p_y_te

    mae_te = mean_absolute_error(y_te, p_y_te)
    rmse_te = mean_squared_error(y_te, p_y_te, squared=False)
    ev_te = explained_variance_score(y_te, p_y_te)
    mse_te = mean_squared_error(y_te,p_y_te)
    r2_te = r2_score(y_te,p_y_te)
    r_te = pearsonr(y_te,p_y_te)[0]
    # write test accuracy and clx time

    if classify:
        # Compare all classes one vs one
        comb = list(combinations(list(np.unique(df_te[scorename])), r=2))

        acc_all = []
        for a, b in comb:
            Y = df_te[scorename][df_te[scorename].isin([a, b])]
            pred = p_y_te[df_te[scorename].isin([a, b])]
            Y = np.where(Y == a, 1, 0)
            clf = LinearSVC()
            clf.fit(np.asarray(pred).reshape(-1, 1), Y)
            Y_pred = clf.predict(np.asarray(pred).reshape(-1, 1))
            acc = balanced_accuracy_score(Y, Y_pred)
            df_te[str(a) + '_vs_' + str(b)] = acc
            acc_all.append(acc)

        bal_acc = np.mean(acc_all)
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                     'r_te', 'rmse_te', 'bal_acc_fc', "bal_acc_all"])

        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te,  rmse_te, bal_acc, acc_te]
    else:
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                'r_te', 'rmse_te', "bal_acc_all"])
        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te, rmse_te, acc_te]

    outs.to_csv(opj(outpath, 'test.csv'), index=False)

    # r2 = print(r2_score(y_te, p_y_te))
    # r = print(pearsonr(y_te, p_y_te))
    outs.to_csv(opj(outpath, 'test_es.csv'), index=False)

    return outs, df_te



def forward_pass_reg(X_te,net):
    # used to overcome memory constraints
    net.eval()
    outs30=[]
    ims = X_te.shape
    for n in range(0,ims[0]):
        im = X_te[n].reshape(1,1,ims[2],ims[3],ims[4])
        temp = net(im)
        temp0 = temp[0].data.cpu().numpy().squeeze()
        outs30.append(temp0) 
    probs = np.vstack((outs30)).squeeze()
    return probs


def evaluate_test_accuracy_regressor(model, outpath, df_te, scorename,
                                     classify=True, linear_size=2048):

    # Load validated model
    net=0

    if model == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=1)
    if model == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=1)
    elif model == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=1)
    elif model == 'ResNet_50':
        net = resnet50(num_classes=1, linear_size=linear_size)
    elif model == 'ResNet_34':
        net = resnet34(num_classes=1, linear_size=linear_size)
    elif model == 'ResNet_152':
        net = resnet152(num_classes=1, linear_size=linear_size)
    elif model == 'DenseNet_121_dropout':
        net = densenet121_3D_DropOut(num_classes=1)
    elif model == 'DenseNet_161_dropout':
        net = densenet161_3D_DropOut(num_classes=1)
    else:
        print('Check model type')

    model = torch.nn.DataParallel(net)
    net = load_net_weights2(model, opj(outpath, 'model_state_dict.pt'))
    X_te, y_te = read_X_y_5D(df_te, scorename, regression=True)
    X_te = Variable(torch.from_numpy(X_te))

    # save accuracy as csv file
    # outs = pd.DataFrame(columns=['mae_te','ev_te','mse_te','r2_te', 'r_te',
    #                              'bal_acc', 'acc'])
    p_y_te = forward_pass_reg(X_te, net)
    df_te['Y_pred'] = p_y_te


    mae_te = mean_absolute_error(y_te, p_y_te)
    rmse_te = mean_squared_error(y_te, p_y_te, squared=False)
    ev_te = explained_variance_score(y_te, p_y_te)
    mse_te = mean_squared_error(y_te,p_y_te)
    r2_te = r2_score(y_te,p_y_te)
    r_te = pearsonr(y_te,p_y_te)[0]
    # write test accuracy and clx time

    if classify:
        # Compare all classes one vs one
        comb = list(combinations(list(np.unique(df_te[scorename])), r=2))

        acc_all = []
        for a, b in comb:
            Y = df_te[scorename][df_te[scorename].isin([a, b])]
            Y_pred = np.asarray(p_y_te[df_te[scorename].isin([a, b])])

            Y_true = np.where(Y == b, 1, 0).astype(bool)
            roc = Roc(Y_pred, Y_true)
            roc.calculate()
            # clf = LinearSVC()
            # clf.fit(np.asarray(pred).reshape(-1, 1), Y)
            # Y_pred = clf.predict(np.asarray(pred).reshape(-1, 1))
            # acc = balanced_accuracy_score(Y, Y_pred)
            df_te[str(a) + '_vs_' + str(b)] = roc.auc
            acc_all.append(roc.auc)

        bal_acc = np.mean(acc_all)
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                     'r_te', 'rmse_te', 'bal_acc'])

        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te,  rmse_te, bal_acc]
    else:
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                'r_te', 'rmse_te'])
        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te, rmse_te]

    outs.to_csv(opj(outpath, 'test.csv'), index=False)

    # return outs, p_y_te, Y_pred
    return outs, df_te




def run_SML_Classifiers(meth, x_tr, y_tr, x_va, y_va, x_te, y_te, trd, pp=0,
                        mi=1000, parallelization=False, n_cpu=8):
    # postprocess scaling:
    if pp == 1: # explore and use better results
        ss = StandardScaler().fit(np.concatenate((x_tr, x_va)))
        x_tr = ss.transform(x_tr)
        x_va = ss.transform(x_va)
        x_te = ss.transform(x_te)

    nt = n_cpu
    # Hyperparameter grids
    C_range_lin = np.logspace(-20, 10, 10, base=2)
    C_range_ker = np.logspace(-10, 20, 10, base=2)
    Y_range = np.logspace(-25, 5, 10, base=2)
    coef0Vals = [-1,0,1] # Coefficients for Poly and Sigmoid Kernel SVMs

    param_grid_lr = [{'C': C_range_lin}]
    param_grid_svml = [{'C': C_range_lin, 'gamma': Y_range}]
    param_grid_svmk = [{'C': C_range_ker, 'gamma': Y_range, 'coef0': coef0Vals}]

    if meth == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif meth == 'LR':
        gs = GridSearch(model = LogisticRegression(), param_grid = param_grid_lr, parallelize=parallelization, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = LogisticRegression(C = gs.best_params['C'])
    elif meth == 'SVML':
        gs = GridSearch(model = SVC(kernel="linear", max_iter=mi),
                        param_grid=param_grid_svml,
                        parallelize=parallelization,
                        num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="linear", C = gs.best_params['C'], gamma = gs.best_params['gamma'], max_iter=mi)
    elif meth == 'SVMR':
        gs = GridSearch(model = SVC(kernel="rbf", max_iter=mi), param_grid = param_grid_svmk, parallelize=parallelization, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="rbf", C = gs.best_params['C'], gamma = gs.best_params['gamma'], max_iter=mi)
    elif meth == 'SVMP':
        gs = GridSearch(model = SVC(kernel="poly", degree =2, max_iter=mi), param_grid = param_grid_svmk, parallelize=parallelization, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="poly", C = gs.best_params['C'], gamma = gs.best_params['gamma'], coef0 = gs.best_params['coef0'], max_iter=mi)
    elif meth == 'SVMS':
        gs = GridSearch(model = SVC(kernel="sigmoid", max_iter=mi), param_grid = param_grid_svmk, parallelize=parallelization, num_threads = nt)
        gs.fit(x_tr, y_tr, x_va, y_va)
        clf = SVC(kernel="sigmoid", C = gs.best_params['C'], gamma = gs.best_params['gamma'], coef0 = gs.best_params['coef0'], max_iter=mi)
    else:
        print('Check Valid Classifier Names')

    if trd == 'tr': # correct; use this only
        clf.fit(x_tr, y_tr)
    elif trd == 'tr_val':
        clf.fit(np.concatenate((x_tr, x_va)), np.concatenate((y_tr, y_va)))
    else:
        print('Choose trd as tr or tr_val')

    scr = clf.score(x_te, y_te)

    return scr




def generate_average_sensivity_map(model, df, num_classses, scorename, modelpath,
                                   transform='zscore', cuda=True,
                                   imgpathcol='imgpath', taskmode='clx',
                                   postprocess='abs', target_class=None):
    """Helper function to average sensitivity analysis across multiple images

    Args:
        model (string): model name to use
        df (pandas df): df with image path in a column (imgpathcol)
        num_classses (int): Number of classes
        scorename (str): score column in the df
        modelpath (path): where is the saved model
        transform (str, optional): Transform to apply to data. Defaults to 'zscore'.
        cuda (bool, optional): Defaults to True.
        imgpathcol (str, optional): column with paths for images. Defaults to 'imgpath'.
        taskmode (str, optional): clx or reg. Defaults to 'clx'.
        postprocess (str, optional): Postprocessing to use. Defaults to 'abs'.
        target_class (int, optional): target class. Defaults to None.

    Returns:
        [type]: [description]
    """

    trained_net = returnModel(model, modelpath, num_classses)
    temp = image.load_img(list(df[imgpathcol])[0])
    r_maps = []
    for img in range(len(df)):
        vol, target = read_X_y_5D_idx(df, img, scorename=scorename,
                                    regression=False, transform=transform)
        r_maps.append(sensitivity_analysis(trained_net, vol, cuda=cuda,
                                           target_class=None, taskmode=taskmode,
                                           postprocess=postprocess))


    r_map = np.average(np.vstack(r_maps), axis=0)
    r_map_nifti = image.new_img_like(temp, data=np.squeeze(r_map))

    return r_map, r_map_nifti



def fixed_weight_decoder(df, weight_map, scorename,
                         measure='dotproduct', classify=True,
                         resample_to_signature=False,
                         transform=None):

    y_te = np.asarray(df[scorename])
    signature = image.load_img(weight_map)
    signature_dat = np.asanyarray(signature.dataobj).flatten()
    p_y_te = []
    for img in tqdm(df['imgpath']):
        img = image.load_img(img)
        if resample_to_signature:
            img = image.resample_to_img(img, signature)
        img_dat = np.asanyarray(img.dataobj).flatten()
        if transform == 'zscore':
            img_dat = zscore(img_dat)
        p_y_te.append(np.dot(signature_dat, img_dat))

    p_y_te = np.asarray(p_y_te)
    df['Y_pred'] = p_y_te

    y_te_z = zscore(y_te)
    p_y_te_z = zscore(p_y_te)
    mae_te = mean_absolute_error(y_te, p_y_te)
    rmse_te = mean_squared_error(y_te, p_y_te, squared=False)
    ev_te = explained_variance_score(y_te_z, p_y_te_z)
    mse_te = mean_squared_error(y_te,p_y_te)
    r2_te = r2_score(y_te_z,p_y_te_z)
    r_te = pearsonr(y_te_z,p_y_te_z)[0]


    if classify:
        # Compare all classes one vs one
        comb = list(combinations(list(np.unique(df[scorename])), r=2))

        acc_all = []
        for a, b in comb:

            Y = df[scorename][df[scorename].isin([a, b])]
            Y_pred = np.asarray(p_y_te[df[scorename].isin([a, b])])

            Y_true = np.asarray(np.where(Y == b, 1, 0).astype(bool))
            roc = Roc(Y_pred, Y_true)
            roc.calculate()
            # clf = LinearSVC()
            # clf.fit(np.asarray(pred).reshape(-1, 1), Y)
            # Y_pred = clf.predict(np.asarray(pred).reshape(-1, 1))
            # acc = balanced_accuracy_score(Y, Y_pred)
            df[str(a) + '_vs_' + str(b)] = roc.auc
            acc_all.append(roc.auc)


            # Y = df[scorename][df[scorename].isin([a, b])]
            # pred = p_y_te[df[scorename].isin([a, b])]
            # Y = np.where(Y == a, 1, 0)
            # clf = LinearSVC()
            # clf.fit(np.asarray(pred).reshape(-1, 1), Y)
            # Y_pred = clf.predict(np.asarray(pred).reshape(-1, 1))
            # acc = balanced_accuracy_score(Y, Y_pred)
            # df[str(a) + '_vs_' + str(b)] = acc
            # acc_all.append(acc)

        bal_acc = np.mean(acc_all)
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                     'r_te', 'rmse_te', 'bal_acc'])

        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te,  rmse_te, bal_acc]
    else:
        outs = pd.DataFrame(columns=['mae_te','expvar_te','mse_te','r2_te',
                                'r_te', 'rmse_te'])
        outs.loc[0] = [mae_te, ev_te, mse_te, r2_te, r_te, rmse_te]

    return outs, df





def WeightSampler(target):
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler





def train_regression_tune(config):

    # Create model path if not exists
    if not os.path.exists(config['constant']['outpath']):
        os.makedirs(config['constant']['outpath'])

    if config['model'] == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=1)
    elif config['model'] == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=1)
    elif config['model'] == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=1)
    elif config['model'] == 'ResNet_152':
        net = resnet152(num_classes=1, linear_size=config['constant']['linear_size'])
    elif config['model'] == 'ResNet_50':
        net = resnet50(num_classes=1, linear_size=config['constant']['linear_size'])
    elif config['model'] == 'ResNet_34':
        net = resnet34(num_classes=1, linear_size=config['constant']['linear_size'])
    elif config['model'] == 'DenseNet_121_dropout':
        net = densenet121_3D_DropOut(num_classes=1)
    elif config['model'] == 'DenseNet_161_dropout':
        net = densenet161_3D_DropOut(num_classes=1)
    else:
        print('Check model')

    criterion = nn.MSELoss()

    # Get the training and validation data
    trainset = MRIDataset(config['constant']['df_tr'], config['constant']['scorename'], regression=True, transform=config['x_transform'])

    if config['constant']['imbalance_sampler']:
        sampler = WeightSampler(config['constant']['df_tr'][config['constant']['scorename']])
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=False,
                                sampler=sampler,
                                num_workers=config['constant']['num_workers'], drop_last=True)
    else:
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['constant']['num_workers'], drop_last=True)

    validset = MRIDataset(config['constant']['df_val'], config['constant']['scorename'], regression=True, transform=config['x_transform'])
    validloader = DataLoader(validset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['constant']['num_workers'], drop_last=True)

    # Use GPU is cuda available

    if config['constant']['cuda_avl']:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net,
                                    device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Load optimizer (use ADAM for now)
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=3, verbose=True)

    # Early stopping details
    min_val_mae = 999
    epochs_no_improve = 0
    valid_mae = 999


    history = pd.DataFrame(columns=['scorename','epoch',
                                    'val_mae','val_ev','val_r2', 'val_r',
                                    'tr_loss (mse)', 'val_loss (mse)'])

    # Baseline
    for epoch in range(config['n_epochs']):
        # Train
        print('Training epoch ' + str(epoch))
        t0 = time.time()
        loss = train(trainloader, net, optimizer, criterion,
                     config['constant']['cuda_avl'])
        loss = loss.data.cpu().numpy()
        # Do not get training acc (not useful?)
        # y_true, y_pred, _ = test(trainloader, net, cuda_avl, criterion)
        # train_acc = accuracy_score(y_true, y_pred)
        # bal_train_acc = balanced_accuracy_score(y_true, y_pred)

        # Validate
        # print('Validating')

        y_true, y_pred = test_reg(validloader, net, config['constant']['cuda_avl'])
        valid_mae = mean_absolute_error(y_true, y_pred)
        valid_ev = explained_variance_score(y_true, y_pred)
        val_loss = mean_squared_error(y_true, y_pred)
        valid_r2 = r2_score(y_true, y_pred)
        valid_r = pearsonr(y_true, y_pred)[0]

        history.loc[epoch] = [config['constant']['scorename'], epoch, valid_mae, valid_ev,
                              valid_r2, valid_r, loss, val_loss]
        history.to_csv(opj(config['constant']['outpath'], 'train_history.csv'), index=False)

        print ('scorename: ' + config['constant']['scorename'] +' Epoch '+str(epoch)
                +' Tr. MSE.: '+ str(np.round(loss, 2))
                +' Tr. RMSE: ' + str(np.round(np.sqrt(loss), 2))
                + ' Val. MSE: ' + str(np.round(val_loss, 2))
                + ' Val. RMSE: ' + str(np.round(np.sqrt(val_loss), 2))
                + ' Val. MAE ' + str(np.round(valid_mae, 2))
                + ' Val. R2 ' + str(np.round(valid_r2, 2))
                + ' Val. r '  + str(np.round(valid_r, 2)))
        print('{} seconds'.format(time.time() - t0))

        if config['constant']['early_stopping']:
            # If the validation accuracy is at a maximum
            if valid_mae < min_val_mae:
                # Save the model if there is improvement
                torch.save(net.state_dict(), open(opj(config['constant']['outpath'],
                                                      'model_state_dict.pt'),
                                                  'wb'))
                epochs_no_improve = 0
                min_val_mae = valid_mae
                print('Best score, saving model.')
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == config['constant']['early_plateau']:
                    print('Early stopping!')
                    fig = plt.figure()
                    plt.plot(history['tr_loss (mse)'], label='Training loss')
                    plt.plot(history['val_loss (mse)'], label='Validation loss')
                    plt.xlabel('Epochs')
                    plt.legend()
                    plt.ylabel('Mean squared error')
                    plt.title('Running loss at early stop '
                              + str(epoch) + ' epochs')
                    fig.savefig(opj(config['constant']['outpath'], 'train_history_early.png'))
                    return history, min_val_mae
        else:
            if valid_mae < min_val_mae:
                min_val_mae = valid_mae
                # Save the model if there is improvement
                torch.save(net.state_dict(), open(opj(config['constant']['outpath'],
                                                      'model_state_dict.pt'),
                                                  'wb'))
                epochs_no_improve = 0
                min_val_mae = valid_mae
                print('Best score, saving model.')

        # Decay Learning Rate
        scheduler.step(valid_mae)

    # Add best score to tune
    tune.report(min_val_mae=min_val_mae)

    fig = plt.figure()
    plt.plot(history['tr_loss (mse)'], label='Training loss')
    plt.plot(history['val_loss (mse)'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Mean squared error')
    plt.title('Running loss no early stop ' + str(epoch) + ' epochs')
    fig.savefig(opj(config['constant']['outpath'], 'train_history_full.png'))

    return min_val_mae





def train_tune(config):

    # Create model path if not exists
    if not os.path.exists(config['constant']['outpath']):
        os.makedirs(config['constant']['outpath'])

    if config['model'] == 'AlexNet3D_Dropout_Regression':
        net = AlexNet3D_Dropout_Regression(num_classes=config['constant']['num_classes'])
    elif config['model'] == 'AlexNet3D_Deeper_Dropout':
        net = AlexNet3D_Deeper_Dropout(num_classes=config['constant']['num_classes'])
    elif config['model'] == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=config['constant']['num_classes'])
    elif config['model'] == 'ResNet_152':
        net = resnet152(num_classes=config['constant']['num_classes'], linear_size=config['constant']['linear_size'])
    elif config['model'] == 'ResNet_50':
        net = resnet50(num_classes=config['constant']['num_classes'], linear_size=config['constant']['linear_size'])
    elif config['model'] == 'ResNet_34':
        net = resnet34(num_classes=config['constant']['num_classes'], linear_size=config['constant']['linear_size'])
    elif config['model'] == 'DenseNet_121_dropout':
        net = densenet121_3D_DropOut(num_classes=config['constant']['num_classes'])
    elif config['model'] == 'DenseNet_161_dropout':
        net = densenet161_3D_DropOut(num_classes=config['constant']['num_classes'])
    else:
        print('Check model')

    criterion = nn.CrossEntropyLoss()

    # Get the training and validation data
    trainset = MRIDataset(config['constant']['df_tr'], config['constant']['scorename'], regression=False, transform=config['x_transform'])

    if config['constant']['imbalance_sampler']:
        sampler = WeightSampler(config['constant']['df_tr'][config['constant']['scorename']])
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=False,
                                sampler=sampler,
                                num_workers=config['constant']['num_workers'], drop_last=True)
    else:
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['constant']['num_workers'], drop_last=True)

    validset = MRIDataset(config['constant']['df_val'], config['constant']['scorename'], regression=False, transform=config['x_transform'])
    validloader = DataLoader(validset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['constant']['num_workers'], drop_last=True)

    # Use GPU is cuda available

    if config['constant']['cuda_avl']:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(net,
                                    device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Load optimizer (use ADAM for now)
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=3, verbose=True)

    # Early stopping details
    min_val_mae = 999
    epochs_no_improve = 0
    valid_mae = 999


    history = pd.DataFrame(columns=['scorename','epoch',
                                    'val_mae','val_ev','val_r2', 'val_r',
                                    'tr_loss (mse)', 'val_loss (mse)'])

    # Baseline
    for epoch in range(config['n_epochs']):
        # Train
        print('Training epoch ' + str(epoch))
        t0 = time.time()
        loss = train(trainloader, net, optimizer, criterion,
                     config['constant']['cuda_avl'])
        loss = loss.data.cpu().numpy()
        # Do not get training acc (not useful?)
        # y_true, y_pred, _ = test(trainloader, net, cuda_avl, criterion)
        # train_acc = accuracy_score(y_true, y_pred)
        # bal_train_acc = balanced_accuracy_score(y_true, y_pred)

        # Validate
        # print('Validating')

        y_true, y_pred, _ = test(validloader, net, config['constant']['cuda_avl'], criterion)
        valid_mae = mean_absolute_error(y_true, y_pred)
        valid_ev = explained_variance_score(y_true, y_pred)
        val_loss = mean_squared_error(y_true, y_pred)
        valid_r2 = r2_score(y_true, y_pred)
        valid_r = pearsonr(y_true, y_pred)[0]

        history.loc[epoch] = [config['constant']['scorename'], epoch, valid_mae, valid_ev,
                              valid_r2, valid_r, loss, val_loss]
        history.to_csv(opj(config['constant']['outpath'], 'train_history.csv'), index=False)

        print ('scorename: ' + config['constant']['scorename'] +' Epoch '+str(epoch)
                +' Tr. MSE.: '+ str(np.round(loss, 2))
                +' Tr. RMSE: ' + str(np.round(np.sqrt(loss), 2))
                + ' Val. MSE: ' + str(np.round(val_loss, 2))
                + ' Val. RMSE: ' + str(np.round(np.sqrt(val_loss), 2))
                + ' Val. MAE ' + str(np.round(valid_mae, 2))
                + ' Val. R2 ' + str(np.round(valid_r2, 2))
                + ' Val. r '  + str(np.round(valid_r, 2)))
        print('{} seconds'.format(time.time() - t0))

        if config['constant']['early_stopping']:
            # If the validation accuracy is at a maximum
            if valid_mae < min_val_mae:
                # Save the model if there is improvement
                torch.save(net.state_dict(), open(opj(config['constant']['outpath'],
                                                      'model_state_dict.pt'),
                                                  'wb'))
                epochs_no_improve = 0
                min_val_mae = valid_mae
                print('Best score, saving model.')
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == config['constant']['early_plateau']:
                    print('Early stopping!')
                    fig = plt.figure()
                    plt.plot(history['tr_loss (mse)'], label='Training loss')
                    plt.plot(history['val_loss (mse)'], label='Validation loss')
                    plt.xlabel('Epochs')
                    plt.legend()
                    plt.ylabel('Mean squared error')
                    plt.title('Running loss at early stop '
                              + str(epoch) + ' epochs')
                    fig.savefig(opj(config['constant']['outpath'], 'train_history_early.png'))
                    return history, min_val_mae
        else:
            if valid_mae < min_val_mae:
                min_val_mae = valid_mae
                # Save the model if there is improvement
                torch.save(net.state_dict(), open(opj(config['constant']['outpath'],
                                                      'model_state_dict.pt'),
                                                  'wb'))
                epochs_no_improve = 0
                min_val_mae = valid_mae
                print('Best score, saving model.')

        # Decay Learning Rate
        scheduler.step(valid_mae)

    # Add best score to tune
    tune.report(min_val_mae=min_val_mae)

    fig = plt.figure()
    plt.plot(history['tr_loss (mse)'], label='Training loss')
    plt.plot(history['val_loss (mse)'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Mean squared error')
    plt.title('Running loss no early stop ' + str(epoch) + ' epochs')
    fig.savefig(opj(config['constant']['outpath'], 'train_history_full.png'))

    return min_val_mae