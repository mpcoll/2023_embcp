import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
import scipy
from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
from mne.report import Report
from mne.preprocessing import ICA, create_eog_epochs
import matplotlib.pyplot as plt
from functools import partial
from oct2py import octave


###############################
# Parameters
###############################
layout = BIDSLayout('/media/mp/lxhdd/2020_embcp/source')
part = sorted(['sub-' + s for s in layout.get_subject()])
basepath = '/media/mp/lxhdd/2020_embcp/'
bidsout = '/media/mp/lxhdd/2020_embcp/source'
derivpath = '/media/mp/lxhdd/2020_embcp/derivatives2'

# ['PO3', 'C1', 'F3', 'TP7', 'F7']
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
                    #  ' sub-012': [], # Data looks weird (Bad ref?), exclude
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
                    #  ' sub-027': [], # Data looks weird (Bad ref?), exclude
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
                         },
         # Visually identified EOG ICA
         'icatoremove': {'sub-003': [0, 1, 5],
                         'sub-004': [0, 1, 2, 3],
                         'sub-005': [0, 1, 2],
                         'sub-006': [0, 5],
                         'sub-007': [0, 1, 2, 3, 5],
                         'sub-008': [0, 1, 3],
                         'sub-009': [0, 1, 3], 
                         'sub-011': [0, 1],
                        #  'sub-012': [], # CHECK ISSUES
                         'sub-013': [0, 1],
                         'sub-014': [0, 1],
                         'sub-016': [0, 1, 2],
                         'sub-017': [0, 2],
                         'sub-018': [0, 6],
                         'sub-019': [0, 1, 2],
                         'sub-020': [0, 3, 4],
                         'sub-021': [0, 1, 2],
                         'sub-022': [0, 2],
                         'sub-023': [0, 2, 3],
                         'sub-024': [0, 7],
                         'sub-025': [0, 3],
                         'sub-026': [0, 1],
                        #  'sub-027': [], # CHECK ISSUES
                         'sub-028': [0,  5],
                         'sub-029': [0, 1, 2],
                         'sub-030': [0, 2],
                         'sub-031': [0, 1],
                         'sub-032': [0, 1, 2],
                         'sub-033': [0, 1],
                         'sub-034': [0, 1, 2],
                         'sub-036': [0, 1],
                         'sub-037': [0, 3],
                         'sub-038': [0, 1],
                         'sub-039': [0, 1, 3],
                         'sub-040': [0, 1, 2, 3],
                         'sub-041': [0, 2],
                         'sub-042': [0, 3],
                         'sub-043': [0, 1],
                         'sub-044': [0, 1],
                         'sub-045': [0, 1, 2],
                         'sub-046': [0, 2],
                         'sub-047': [0, 1, 2]
                         },
         'jumpthreshold': 60,
         'absthreshold': 150
        }

# Loop participants

part = [p for p in part if p not in  ['sub-012', 'sub-015', 'sub-027', 'sub-023']]

reject_dict = pd.DataFrame(columns=['rest_start', 'thermal_start',
                                    'auditory_start', 'thermalrate_start',
                                    'auditoryrate_start', 'ica_removed'],
                           index=part)



for p in part:
    print(p)
    pdir = opj(derivpath, p, 'eeg')
    if not os.path.exists(pdir):
        os.makedirs(pdir)


    # Initialize report
    report = Report(verbose=False, subject=p,
                    title='EEG report for part ' + p)

    # Load raw file
    raw = mne.io.read_raw_brainvision(opj(bidsout, p, 'eeg',
                                          p + '_task-painaudio_eeg.vhdr'),
                                      eog=['VEOG', 'HEOG'])

    # Set channel positions
    raw = raw.set_montage('standard_1005')

    # _________________________________________________________________
    # Filter and zapline

    # Plot raw spectrum
    plt_psd = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psd, captions='Raw spectrum', section='Preprocessing')

    # Filter 0.05
    raw.load_data()

    raw = raw.filter(0.5, None)


    # Remove line noise with zapline
    octave.addpath(opj(basepath, 'external/NoiseTools'))
    x = raw.get_data()
    x = np.swapaxes(x, 1, 0)
    octave.push('x', x)
    y, yy = octave.eval("[y, yy] = nt_zapline(x, 60/500, 6)", nout=2)

    # Put cleaned data back into structure
    raw = mne.io.RawArray(np.swapaxes(y, 0, 1), raw.info)

    # Plot cleaned spectrum
    plt_psd = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psd, captions='Filtered/zapped spectrum', section='Preprocessing')

    # _________________________________________________________________
    # Bad channels

    # Plot channel variance to help with bad channe identification
    # Get z scored variance
    chan_var = scipy.stats.zscore(np.var(raw.get_data()[:64], axis=1))
    # Plot
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(1, 65), chan_var)
    for idx, c in enumerate(raw.ch_names[:64]):
        plt.text(idx+1, chan_var[idx], c)
    plt.title('Zscored channel variance for ' + p, size=25)
    plt.xlabel('Channel', fontdict=dict(size=25))
    plt.ylabel('Z-score', fontdict=dict(size=25))
    plt.axhline(y=3, linestyle='--')
    plt.axhline(y=-3, linestyle='--')
    plt.yticks(fontsize=15)
    report.add_figs_to_section(
        fig, captions='Channel variance', section='Preprocessing')

    # Plot raw data and stop if visual inspection is on

    if param['visualinspect']:
        raw.plot(
            n_channels=raw.info['nchan'],
            scalings=dict(eeg=0.000020),
            block=False)

    # Flag manually identified bad channels
    raw.info['bads'] = param['badchannels'][p]

    # PLot sensors
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figs_to_section(
        plt_sens,
        captions='Sensor positions (bad in red)',
        section='Preprocessing')

    # _________________________________________________________________
    # Epoch according to condition to remove breaks and pre/post

    events = pd.read_csv(opj(bidsout, p, 'eeg',
                              p + '_task-painaudio_events.tsv'),
                          sep='\t')

    # Epochs are 500 s
    events['offset'] = events['onset'] + raw.info['sfreq']*500

    # Keep only start of trial
    events = events[events.condition.isin(['rest_start',
                                           'thermal_start',
                                           'auditory_start',
                                           'thermalrate_start',
                                           'auditoryrate_start'])]
    events_dict = {'rest_start':  1,
                   'thermal_start': 2,
                   'auditory_start': 3,
                   'thermalrate_start': 4,
                   'auditoryrate_start': 5}
    events_c = pd.DataFrame(columns=['cue_num', 'empty', 'sample'])
    events_c['cue_num'] = [events_dict[s] for s in events.condition]
    events_c['sample'] = list(events['onset'])
    events_c['empty'] = 0
    events_epochs = np.asarray(events_c[['sample', 'empty', 'cue_num']])

    # Create epochs
    epochs_good = mne.Epochs(raw,
                        events=events_epochs,
                        event_id=events_dict,
                        tmin=0,
                        tmax=500,
                        reject=None,
                        baseline=None,
                        preload=True)

    # Make a new raw with only times of interest for ICA
    raw = mne.io.RawArray(np.hstack(epochs_good.get_data()),
                          raw.info)


    # _________________________________________________________________
    # ICA
    # Create regular epochs for ICA

    ica_raw = raw.copy().filter(0.5, 150)
    epochs = mne.make_fixed_length_epochs(raw, duration=2)

    # Drop very bad segments
    epochs.drop_bad(reject=dict(eeg=500e-6), verbose=False)

    ica = ICA(n_components=None,
              method='fastica',
              random_state=23)
    # Fit ICA
    ica.fit(epochs)

    # Detect artifacts
    if p == 'sub-022':
        eogchan = 'Fp1'
    else:
        eogchan = 'VEOG'
    ica.detect_artifacts(epochs, eog_criterion=range(2),
                         skew_criterion=2.0, kurt_criterion=15.0,
                         eog_ch='Fp1')

    # removed_ica = ica.exclude.copy()
    removed_ica = param['icatoremove'][p]
    # Add topo figures to report
    plt_icacomp = ica.plot_components(show=False, res=25)
    for l in range(len(plt_icacomp)):
        report.add_figs_to_section(
            plt_icacomp[l], captions='ICA', section='Artifacts')

    # Create eog epochs to find ICA corrleating with blinks
    eog_epochsv = create_eog_epochs(
        raw, ch_name='VEOG', verbose=False)  # get single EOG trials
    eog_indsv, scoresr = ica.find_bads_eog(
        raw, ch_name='VEOG', verbose=False)  # find correlation

    fig = ica.plot_scores(scoresr, exclude=eog_indsv, show=False)
    report.add_figs_to_section(fig, captions='Correlation with EOG',
                               section='Artifact')

    # Get ICA identified in visual inspection
    figs = list()

    # Plot removed ICA and add to report
    figs.append(ica.plot_sources(eog_epochsv.average(),
                                 show=False,
                                 title='ICA removed on eog epochs'))

    report.add_figs_to_section(figs, section='ICA',
                               captions='Removed components '
                               + 'highlighted')

    report.add_htmls_to_section(
        htmls="IDX of removed ICA: " + str(removed_ica),
        captions='ICA-Removed',
        section='Artifacts')

    report.add_htmls_to_section(htmls="Number of removed ICA: "
                                + str(len(removed_ica)), captions="""ICA-
                                Removed""", section='Artifacts')

    figs = list()
    capts = list()

    for ical in range(len(ica._ica_names)):

        ica.exclude = [ical]
        figs.append(ica.plot_sources(eog_epochsv.average(),
                                     show=False))
        plt.close("all")

    f = None
    report.add_slider_to_section(figs, captions=ica._ica_names,
                                 section='ICA-FULL')


    # # Remove components manually identified
    ica.exclude = removed_ica

    # Apply ICA
    epochs_clean = ica.apply(epochs_good)

    # Interpolate bad channels
    if epochs_clean.info['bads']:
        epochs_clean = epochs_clean.interpolate_bads(reset_bads=True)


    # _________________________________________________________________
    # Add the behav to the data and save as raw
    rej_stats = pd.DataFrame(index=raw.ch_names,
                             columns=['rest_start',
                                           'thermal_start',
                                           'auditory_start',
                                           'thermalrate_start',
                                           'auditoryrate_start',
                                           'total_chan',
                                           'total_dropped'],
                             data=0)

    removed_points = pd.DataFrame(columns=list(events_dict.keys()), index=[p])
    for epo_idx in range(len(epochs_clean)):
        epo = epochs_clean[epo_idx]

        # As in nickel, reference before artifact
        epo, _ = mne.set_eeg_reference(epo, 'average', projection=True)

        epo_type = list(events_dict.keys())[list(events_dict.values()).index(epo.events[0][2])]



        raw = mne.io.RawArray(epo.get_data()[0, ::][:, :-1],
                              info=epo.info)

        if epo_type != 'rest_start':

            # Load behavioural
            behav = pd.read_csv(opj(bidsout, p, 'eeg',
                                    p + '_task-painaudio_beh.tsv'), sep='\t')

            # Create channels for behavioural data
            behav_info = mne.create_info(['pain_rate'], raw.info['sfreq'], ['misc'])
            prate_chan = mne.io.RawArray(np.expand_dims(np.asarray(behav['pain_rating']), 0), info=behav_info)
            prate_chan.info['highpass'] = raw.info['highpass']
            prate_chan.info['lowpass'] = raw.info['lowpass']

            behav_info = mne.create_info(['audio_rate'], raw.info['sfreq'], ['misc'])
            audio_chan = mne.io.RawArray(np.expand_dims(np.asarray(behav['audio_rating']), 0), info=behav_info)
            audio_chan.info['highpass'] = raw.info['highpass']
            audio_chan.info['lowpass'] = raw.info['lowpass']

            behav_info = mne.create_info(['stim_int'], raw.info['sfreq'], ['misc'])
            stim_chan = mne.io.RawArray(np.expand_dims(np.asarray(behav['stim_intensity']), 0), info=behav_info)
            stim_chan.info['highpass'] = raw.info['highpass']
            stim_chan.info['lowpass'] = raw.info['lowpass']

            # Add these channels to the array
            raw.add_channels([prate_chan])
            raw.add_channels([audio_chan])
            raw.add_channels([stim_chan])
        else:
            # Rest is only 5 mins
            raw.crop(tmax=5*60)

        raw = raw.drop_channels(['VEOG', 'HEOG'])



        epo = mne.make_fixed_length_epochs(raw.copy(), 0.2, preload=True)
        epo.drop_bad(reject=dict(eeg=100e-6))

        all_elec = []
        for e in epo.drop_log:
            if e:
                for c in e:
                    rej_stats.loc[c, epo_type] = rej_stats.loc[c, epo_type] + 1
                    rej_stats.loc[c, 'total_chan'] = rej_stats.loc[c, 'total_chan'] + 1

        rej_stats['total_dropped'] += len(np.where(epo.drop_log)[0])

        # epo_ar.drop_bad(reject=dict(eeg=200e-6))
        # epo_ar.drop_log
        # # epo_ar_c = ar.fit_transform(epo_ar)
        # epo_ar.drop_log = epo_c.drop_log
        # # Save epochs to remove
        # reject_dict.loc[p, epo_type] = (len(epo_ar) - len(epo_ar_c))/len(epo_ar)
        # print(reject_dict)

        # epo_c = epo_ar.copy().drop_bad(reject=dict(eeg=200e-6))
        # reject_dict200.loc[p, epo_type] = (len(epo_ar) - len(epo_c))/len(epo_ar)
        # print(reject_dict200)
        # epo_c.drop_log()

        # # Clean the data
        # sfreq = raw.info['sfreq']

        # # Indices of  points above 80 uV
        # absabove = np.unique(np.where(np.abs(raw.get_data()) > 80e-6)[1])


        # # Indices of points with diff > 30 uV
        # diffabove = np.unique(np.where(np.abs(np.diff(raw.get_data()))
        #                                  > 50e-6)[1]
        #                         )

        # # Check how many coming from each channel to additionally spot
        # # bad channels responsible for most bad points
        # chanabove = pd.Series(np.where(np.abs(raw.get_data()) > 120e-6)[0]).value_counts()
        # chandiffabove = pd.Series(np.where(np.abs(np.diff(raw.get_data()))
        #                                  > 30e-6)[0]).value_counts()

        # # Put this in a nice readable data frame
        # rej_by_chans = pd.DataFrame(data=dict(absabove=0, diffabove=0,
        #                                       chan=raw.ch_names),
        #                             index=range(0, len(raw.ch_names)))

        # for c in list(rej_by_chans.index):
        #     if c in list(chanabove.index):
        #         rej_by_chans.loc[c, 'absthresh'] = chanabove[c]
        #     if c in list(chandiffabove.index):
        #         rej_by_chans.loc[c, 'diffthresh'] = chandiffabove[c]

        # rej_by_chans['total'] = rej_by_chans['above80'] + rej_by_chans['diff30']
        # rej_by_chans = rej_by_chans.sort_values(['total'], ascending=False)
        # report.add_htmls_to_section(rej_by_chans.to_html(),
        #                             captions='Rejection by channel ' + epo_type,
        #                             section='Artifacts'
        #                              )
        # # Concatenate
        # bad_idx = np.unique(np.hstack([absabove, diffabove]))
        # bad_idx_s = bad_idx/sfreq
        # # Get % flagged

        # # Flag bad segments in raw
        # # Annotate 400 ms around bad point
        # annot = mne.Annotations(bad_idx_s-0.2,
        #                         duration=0.4,
        #                         description='bad')
        # # Add annot to data
        # raw.set_annotations(annot)

        # # Calculate % of flagged points
        # # Fake raw
        # fraw = np.zeros(raw.get_data().shape[1])
        # # Get anottated points
        # out  = _annotations_starts_stops(raw, 'bad')

        # # Mark all annotated points in fake raw
        # for idx, point in enumerate(out[0]):
        #     fraw[out[0][idx]:out[1][idx]] = 1
        # # Get number of annotated points
        # flagged = np.sum(fraw)

        # # Calculate percentage of points removed
        # removed_points[epo_type] = flagged/raw.get_data().shape[1]*100

        # ______________________________________________________________________
        # Save cleaned data
        raw.save(opj(pdir, p + '_task-painaudio_' + epo_type + '_cleanedeeg_raw.fif'),
                overwrite=True)
    # reject_dict.loc[p, 'ica_removed'] = len(removed_ica)
    # reject_dict.to_csv(opj(derivpath, 'autoreject_dict.csv'))

    # report.add_htmls_to_section(removed_points.to_html(),
    #                             captions='Percent of data points flagged',
    #                             section='Artifacts'
    #                             )
    # removed_points.to_csv(opj(pdir, p + '_task-painaudio_removed_perc.csv'))
    #  _____________________________________________________________________
    # Save report
    rej_stats.sort_values(['total_chan']).to_csv(opj(derivpath, p + '_drop100.csv'))

    report.save(opj(pdir, p + '_task-painaudio_importclean_report.html'),
                open_browser=False, overwrite=True)


