
import numpy as np
import pandas as pd
import mne
from os.path import join as opj
import os
from tqdm import tqdm
from bids import BIDSLayout
from mne.preprocessing import ICA
from oct2py import octave

# %matplotlib inline
# %matplotlib qt5

###############################
# Parameters
###############################
layout = BIDSLayout("/media/mp/lxhdd/2020_embcp/source")
part = sorted(["sub-" + s for s in layout.get_subject()])
bidsout = "/media/mp/lxhdd/2020_embcp/source"
outpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives/visual_clean_epoched_ok"
outpath_filt = "/media/mp/lxhdd/2020_embcp/deep_derivatives/filt_ica"
if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outpath_filt):
    os.mkdir(outpath_filt)

param = {"visualinspect": False}


part_present = list(set([s.split('_')[0] for s in os.listdir(outpath) if 'sub-' in s]))
part = [p for p in part if p not in part_present]
if len(part_present) == 0:
    participants = pd.read_csv(opj(bidsout, 'participants.tsv'), sep='\t')
elif not os.path.exists(opj(outpath, 'participants.csv')):
    participants = pd.read_csv(opj(outpath_filt, 'participants.csv'))
else:
    participants = pd.read_csv(opj(outpath, 'participants.csv'))

participants = participants.set_index('participant_id')

# Loop participants
for p in tqdm(part_present):
    all_epochs = []
    print(p)

    if not os.path.exists(opj(outpath_filt, p + "_task-painaudio_eeg_filtica-raw.fif")):
        # Initialize report
        # Load raw file
        raw = mne.io.read_raw_brainvision(opj(bidsout, p, "eeg",
                                            p + "_task-painaudio_eeg.vhdr"),
                                        eog=["VEOG", "HEOG"])

        # Set channel positions
        raw = raw.set_montage("standard_1005")

        # _________________________________________________________________
        # Filter and zapline

        # Filter 0.05
        raw.load_data()
        raw = raw.filter(0.1, 125)

        # Remove line noise with zapline
        octave.addpath("/media/mp/lxhdd/2020_embcp/external/NoiseTools")
        x = raw.get_data()
        x = np.swapaxes(x, 1, 0)
        octave.push("x", x)
        y, yy = octave.eval("[y, yy] = nt_zapline(x, 60/500, 6)", nout=2)

        # Put cleaned data back into structure
        raw = mne.io.RawArray(np.swapaxes(y, 0, 1), raw.info)


        # _________________________________________________________________
        # Blinks with ICA

        ica = ICA(n_components=None,
                  method='fastica',
                  random_state=23)

        ica.fit(raw)
        eog_indsv, scoresr = ica.find_bads_eog(
            raw, ch_name='VEOG', verbose=False, measure='zscore')  # find correlation
        eog_indsh, scoresr = ica.find_bads_eog(
            raw, ch_name='HEOG', verbose=False, measure='zscore')  # find correlation

        ica.exclude = eog_indsv + eog_indsh

        ica.apply(raw)

        participants.loc[p, 'ica_exclude'] = len(eog_indsv) + len(eog_indsh)

        raw.save(opj(outpath_filt, p + "_task-painaudio_eeg_filtica-raw.fif"))

        participants.to_csv(opj(outpath_filt, 'participants.csv'))
    else:
        raw = mne.io.read_raw_fif(opj(outpath_filt, p + "_task-painaudio_eeg_filtica-raw.fif"))

    # _________________________________________________________________
    # Epoch according to condition to remove breaks and pre/post

    events = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_events.tsv"), sep="\t")

    ratings = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_beh.tsv"), sep="\t")

    for cond in [
        "rest_start",
        "thermal_start",
        "auditory_start",
        "thermalrate_start",
        "auditoryrate_start",]:
        # Keep only start of trial
        events_cond = events[events.condition.isin([cond])]

        if cond != "rest_start":
            raw_cond = raw.copy().crop(
                tmin=np.float(events_cond["onsets_s"]),
                tmax=np.float(events_cond["onsets_s"]) + np.ceil(8.3333333 * 60))
        else:
            raw_cond = raw.copy().crop(
                tmin=np.float(events_cond["onsets_s"]),
                tmax=np.float(events_cond["onsets_s"]) + 5 * 60)
        file_out = opj(outpath, p + '_task-painaudio_'
                       + cond + '_visualclean-raw.fif')


        # Visual inspection
        # Plot with some filtering to remove drift/high freq
        # raw_cond.plot(block=True, n_channels=66, highpass=0.5, lowpass=30,
        #             duration=30, butterfly=False, scalings=dict(eeg=20e-6),
        #             title=p + ' ' + cond)

        # Save after marking. To re-mark, delete file
        # raw_cond.save(file_out, overwrite=True)
        raw_cond = mne.io.read_raw_fif(file_out)

        if cond != "rest_start":
            epochs_cond = []
            for idx, intensity in enumerate([50, 100, 150]):

                # Find where specific intensity starts
                stim_start = np.where(
                    (ratings["stim_intensity"] == intensity)
                    & (np.diff(ratings["stim_intensity"], prepend=[0]) != 0)
                    & (np.ediff1d(ratings["stim_intensity"], to_end=[0]) == 0)
                )[0]

                # Remove two first seconds to make sure it is stabilised
                stim_start = stim_start + raw.info["sfreq"] * 2

                # Find where specific intensity ends
                stim_end = np.where(
                    (ratings["stim_intensity"] == intensity)
                    & (np.ediff1d(ratings["stim_intensity"], to_end=[0]) != 0)
                    & (np.diff(ratings["stim_intensity"], prepend=[0]) == 0)
                )[0]
                # Make sure all ok
                assert len(stim_start) == len(stim_end)
                for s_start, s_end in zip(stim_start, stim_end):
                    # Loop start stop with indexes
                    raw_epo = raw_cond.copy().crop(
                        tmin=s_start / raw.info["sfreq"], tmax=s_end / raw.info["sfreq"]
                    )

                    if cond == "thermalrate_start":
                        pain_rating = np.mean(
                            np.asarray(ratings["pain_rating"])[
                                int(s_start) : int(s_end)
                            ]
                        )
                    else:
                        pain_rating = 999

                    if cond == "auditoryrate_start":
                        audio_rating = np.mean(
                            np.asarray(ratings["audio_rating"])[
                                int(s_start) : int(s_end)
                            ]
                        )
                    else:
                        audio_rating = 999

                    epochs = mne.make_fixed_length_epochs(
                        raw_epo, duration=3.5
                    ).drop_bad()
                    epochs.metadata = pd.DataFrame(
                        dict(
                            subject_id=[p] * len(epochs),
                            condition=[cond] * len(epochs),
                            stim_intensity=[intensity] * len(epochs),
                            stim_start_s=[s_start / raw.info["sfreq"]] * len(epochs),
                            stim_end_s=[s_end / raw.info["sfreq"]] * len(epochs),
                            pain_rating=[pain_rating] * len(epochs),
                            audio_rating=[audio_rating] * len(epochs),
                            order_in_stim=range(len(epochs)),
                        )
                    )

                    epochs_cond.append(epochs)
            epochs_cond = mne.concatenate_epochs(epochs_cond)
            epochs_cond.save(file_out.replace('-raw', '-epo'), overwrite=True)
            participants.loc[p, 'n_chanbads_' + cond] = len(epochs_cond.info['bads'])
            participants.loc[p, 'n_epochs_' + cond] = len(epochs_cond)
        else:
            epochs = mne.make_fixed_length_epochs(raw_cond, duration=2).drop_bad()
            epochs.metadata = pd.DataFrame(
                dict(
                    subject_id=[p] * len(epochs),
                    condition=[cond] * len(epochs),
                    stim_intensity=[0] * len(epochs),
                    stim_start_s=["NA"] * len(epochs),
                    stim_end_s=[["NA"]] * len(epochs),
                    pain_rating=[["NA"]] * len(epochs),
                    audio_rating=[["NA"]] * len(epochs),
                    order_in_stim=range(len(epochs)),
                )
            )
            epochs.save(file_out.replace('-raw', '-epo'), overwrite=True)
            participants.loc[p, 'n_chanbads_' + cond] = len(epochs.info['bads'])
            participants.loc[p, 'n_epochs_' + cond] = len(epochs)
        participants.to_csv(opj(outpath, 'participants.csv'))

