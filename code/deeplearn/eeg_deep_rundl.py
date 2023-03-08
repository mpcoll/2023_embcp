# data path
import pandas as pd
import numpy as np
from os.path import join as opj
import sys
from nilearn import image
from nilearn import plotting
import mne


sys.path.append('deeplearn')
from eeg_deep_learn_utils import (generate_validation_model,
                                  train_test_split_group,
                                  generate_average_sensivity_map,
                                  sensitivity_analysis, returnModel,
                                  generate_validation_model_regression,
                                  evaluate_test_accuracy,
                                  evaluate_test_accuracy_regressor,
                                  run_SML_Classifiers,
                                  fixed_weight_decoder)

# dataset = 'pines_avg_data'
# df = pd.read_csv('/data/datasets/' + dataset +  '/metadata.csv')
# df_te = df[df['original_holdout'] == 'Test']
# df_tr = df[df['original_holdout'] == 'Train']
# df_tr, df_val, df_val2 = train_test_split_group(df_tr, test_prop=0.2)
# df_val = pd.concat([df_val, df_val2])
# Dataset


basepath = "/media/mp/lxhdd/2020_embcp"
datapath = opj(basepath, 'deep_derivatives', 'timefreq', 'strials')



df = pd.read_csv(opj(datapath, 'metadata.csv'))
df = df[df['condition'].isin(['thermal_start', 'rest_start',
                              'thermalrate_start'])]
df['ispain'] = np.where(df['condition'] == 'thermal_start', 1,
                           np.where(df['condition'] == 'thermalrate_start', 1,
                           0))

# Add full path to dataset
df['filepath'] = [opj(datapath, f) for f in df['filename']]


df_tr, df_val, df_te = train_test_split_group(df, test_prop=0.3)




model = 'ResNet_34'
scorename = 'ispain'
num_classes = 2
n_epochs = 50
lr = 0.001
outpath=opj(basepath, 'models/reg_test')
history, max_val_acc = generate_validation_model(df_tr=df_tr,
                                                 df_val=df_val,
                                                 scorename=scorename,
                                                 model=model,
                                                 num_classes=num_classes,
                                                 n_epochs=n_epochs,
                                                 lr=lr,
                                                 linear_size=1024,
                                                 early_stopping=True,
                                                 early_plateau=30,
                                                 outpath=outpath,
                                                 x_transform="standardscaler",
                                                 num_workers=0,
                                                 batch_size=8)






# df_te_avg = pd.read_csv('/data/datasets/pines_avg_data/metadata.csv')
# df_te_avg = df_te_avg[df_te_avg['original_holdout'] == 'Test']

# df_tr = df[df['original_holdout'] == 'Train']
# Hig vs low
# df = pd.read_csv('/data/datasets/' + dataset + '/metadata.csv')
# df = df[~df.subject_id.isin(['sub-34', 'sub-43', 'sub-61'])]

# df['imgpath'] = df['img_path']
# df = df[~df['shock_intensity'].isin([5, 0])].reset_index()
# df['highvslow'] = np.where(df['shock_intensity'] < 6, 0, 1)
# df_tr, df_val, df_te = train_test_split_group(df)

# model = 'AlexNet3D_Dropout_Regression'
# scorename = 'rating'
# num_classes = 1
# n_epochs = 50
# lr = 0.0001
# outpath='/data/models/reg_test'
# history, max_val_acc = generate_validation_model_regression(df_tr=df_tr,
#                                                             df_val=df_val,
#                                                             scorename=scorename,
#                                                             model=model,
#                                                             num_classes=num_classes,
#                                                             n_epochs=n_epochs,
#                                                             lr=lr,
#                                                             cuda_avl=True,
#                                                             early_stopping=True,
#                                                             early_plateau=30,
#                                                             outpath=outpath,
#                                                             x_transform="zscore",
#                                                             num_workers=0,
#                                                             batch_size=16)

# dnn_report, dnn_pred, dnn_class = evaluate_test_accuracy_regressor(model,
#                                                                    outpath,
#                                                                    df_te,
#                                                                    scorename=scorename)



# # Put ratings back 1-5
# sig_report, sig_pred, sig_class = fixed_weight_decoder(df_te,
#                                                        '/data/signatures/pines.nii',
#                                                        scorename='rating',
#                                                        normalize=False)

# from sklearn.svm import LinearSVC, SVC
# from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier




# Y_pred = clf.predict(np.asarray(pred).reshape(-1, 1))
# out['bal_acc'] = balanced_accuracy_score(Y, Y_pred)
# out['acc']  = accuracy_score(Y, Y_pred)



model = 'AlexNet3D_Deeper_Dropout'
scorename = 'rating'
num_classes = 1
n_epochs = 50
lr = 0.0001
outpath='/data/models/reg_test'
history, max_val_acc = generate_validation_model_regression(df_tr=df_tr,
                                                            df_val=df_val,
                                                            scorename=scorename,
                                                            model=model,
                                                            num_classes=num_classes,
                                                            n_epochs=n_epochs,
                                                            lr=lr,
                                                            early_stopping=True,
                                                            early_plateau=30,
                                                            outpath=outpath,
                                                            x_transform="zscore",
                                                            num_workers=0,
                                                            batch_size=32)


df_te_avg = pd.read_csv('/data/datasets/pines_avg_data/metadata.csv')
df_te_avg = df_te_avg[df_te_avg['original_holdout'] == 'Test']
dnn_report, df_dnnpred = evaluate_test_accuracy_regressor(model,
                                                        outpath,
                                                        df_te_avg,
                                                        scorename=scorename)


sig_report, df_sigpred  = fixed_weight_decoder(df_te_avg,
                                                       '/data/signatures/pines.nii',
                                                       scorename='rating')





# _, map = generate_average_sensivity_map(model=model, df=df, scorename=scorename,
#                                         modelpath=outpath, num_classses=2,
#                                         postprocess='abs')



# Histogram plot

import seaborn as sns
import matplotlib.pyplot as plt
df = df_sigpred.copy()
scorename='rating'
fontsize=16

col1 = '#003f5c'
col2 = '#ffa600'

# Hist figure
df['Rounded prediction'] = np.round(df['Y_pred'])
df['Actual'] = df[scorename]
df_melt = df.melt(value_vars=['Rounded prediction', 'Actual'])

fig, ax = plt.subplots(figsize=(3, 5))
g = sns.histplot(x='value', hue='variable', data=df_melt, ax=ax,
                 bins=6, binwidth=0.49, palette=[col1, col2], alpha=0.8,
                 legend=False)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=col1, edgecolor='k', alpha=0.8,
                    label='Rounded prediction'),
                   Patch(facecolor=col2, edgecolor='k', alpha=0.8,
                    label='Actual')]

ax.legend(handles=legend_elements, fontsize=fontsize-3, ncol=2, frameon=False,
          loc=(0, 1))
# g.legend_.set_fontsize(fontsize)
ax.set_xlabel('Rating', fontsize=fontsize)
ax.set_ylabel('Count', fontsize=fontsize)
ax.tick_params('both', labelsize=fontsize)
ax.set_xticks([-0.75, 0.25, 1.25, 2.25, 3.25, 4.25, 5.25])
ax.set_xticklabels(['', 1, 2, 3, 4, 5, ''])
ax.set_title('PINES predicting averaged maps', y=1.1, fontsize=fontsize)
fig.savefig(opj(outpath, 'hist_pines_avg.png'), dpi=600)


# Line plot figure #TODO add both maps
df_avg = df.groupby([scorename, 'subject_id']).mean().reset_index()
fig, ax = plt.subplots(figsize=(5, 5))
sns.pointplot(x='rating', y='Y_pred', data=df_avg, markers=['o'])
ax.set_xlabel('Actual rating', fontsize=fontsize)
ax.set_ylabel('Predictedrating', fontsize=fontsize)
ax.tick_params('both', labelsize=fontsize)


# Classification accuracy figures
acc_cols = [c for c in list(df.columns) if '_vs_' in c]
acc_scores = list(df[acc_cols].loc[0])

acc_labels = [a.replace('_', ' ') for a in acc_cols]
fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(x=acc_labels, y=acc_scores, palette='plasma')
ax.axhline(0.5, linewidth=3, color='k', linestyle='--')
ax.set_xlabel('Binary classification', fontsize=fontsize)
ax.set_ylabel('Balanced accuracy', fontsize=fontsize)
ax.tick_params('both', labelsize=fontsize-2)
ax.set_xticklabels(labels=acc_labels, rotation = 45)

# Accuracy scores summary

sns.barplot(x=sig_report.columns, y=list(sig_report.loc[0]), palette='plasma')



# TODO model 1st level emotions
# TODO use nsynth map


# # Standard machine learning


# # Load all data in a brain mask using nltools
# from nltools import Brain_Data
# data = Brain_Data(list(df['imgpath']), metadata=df)
# data.X = df

# x_tr, y_tr = data[data.X.subject_id.isin(df_tr.subject_id)].data, data[data.X.subject_id.isin(df_tr.subject_id)].X[scorename]
# x_va, y_va = data[data.X.subject_id.isin(df_val.subject_id)].data,  data[data.X.subject_id.isin(df_val.subject_id)].X[scorename]
# x_te, y_te = data[data.X.subject_id.isin(df_te.subject_id)].data, data[data.X.subject_id.isin(df_te.subject_id)].X[scorename]

# import mkl
# mkl.set_num_threads(2)
# out = run_SML_Classifiers('LDA', x_tr, y_tr, x_va, y_va, x_te, y_te,
#                           trd='tr', pp=1, mi=1000, parallelization=False,
#                           n_cpu=2)

# TRY NPS classification
# TRY SIIPS classification
# TRY no smoothing
# TRY in T1s
# TRY in raw bold