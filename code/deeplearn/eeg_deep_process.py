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

# %matplotlib inline

###############################
# Parameters
###############################
layout = BIDSLayout("/media/mp/lxhdd/2020_embcp/source")
part = sorted(["sub-" + s for s in layout.get_subject()])
bidsout = "/media/mp/lxhdd/2020_embcp/source"
outpath = "/media/mp/lxhdd/2020_embcp/deep_derivatives"
if not os.path.exists(outpath):
    os.mkdir(outpath)

# ['PO3', 'C1', 'F3', 'TP7', 'F7']
param = {
    "visualinspect": False,
    # Visually identified bad channels
}

# Loop participants
for p in tqdm(part):
    all_epochs = []
    print(p)

    # Initialize report
    report = Report(verbose=False, subject=p, title="EEG report for part " + p)

    # Load raw file
    raw = mne.io.read_raw_brainvision(
        opj(bidsout, p, "eeg", p + "_task-painaudio_eeg.vhdr"), eog=["VEOG", "HEOG"]
    )

    # Set channel positions
    raw = raw.set_montage("standard_1005")

    # _________________________________________________________________
    # Filter and zapline

    # Plot raw spectrum
    plt_psd = raw.plot_psd(area_mode="range", tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psd, captions="Raw spectrum", section="Preprocessing"
    )
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

    # Plot cleaned spectrum
    plt_psd = raw.plot_psd(area_mode="range", tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psd, captions="Filtered/zapped spectrum", section="Preprocessing"
    )

    # _________________________________________________________________
    # Bad channels

    # Plot channel variance to help with bad channe identification
    # Get z scored variance
    chan_var = scipy.stats.zscore(np.var(raw.get_data()[:64], axis=1))
    # Plot
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(1, 65), chan_var)
    for idx, c in enumerate(raw.ch_names[:64]):
        plt.text(idx + 1, chan_var[idx], c)
    plt.title("Zscored channel variance for " + p, size=25)
    plt.xlabel("Channel", fontdict=dict(size=25))
    plt.ylabel("Z-score", fontdict=dict(size=25))
    plt.axhline(y=3, linestyle="--")
    plt.axhline(y=-3, linestyle="--")
    plt.yticks(fontsize=15)
    report.add_figs_to_section(
        fig, captions="Channel variance", section="Preprocessing"
    )

    # Plot raw data and stop if visual inspection is on

    if param["visualinspect"]:
        raw.plot(n_channels=raw.info["nchan"], scalings=dict(eeg=0.000020), block=False)

    # PLot sensors
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figs_to_section(
        plt_sens, captions="Sensor positions (bad in red)", section="Preprocessing"
    )

    # _________________________________________________________________
    # Epoch according to condition to remove breaks and pre/post

    events = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_events.tsv"), sep="\t"
    )

    ratings = pd.read_csv(
        opj(bidsout, p, "eeg", p + "_task-painaudio_beh.tsv"), sep="\t"
    )

    for cond in [
        "rest_start",
        "thermal_start",
        "auditory_start",
        "thermalrate_start",
        "auditoryrate_start",
    ]:
        # Keep only start of trial
        events_cond = events[events.condition.isin([cond])]

        if cond != "rest_start":
            raw_cond = raw.copy().crop(
                tmin=np.float(events_cond["onsets_s"]),
                tmax=(np.float(events_cond["onsets_s"]) + 8.33 * 60),
            )

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
                        raw_epo, duration=2
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

                    all_epochs.append(epochs)
        else:
            raw_cond = raw.copy().crop(
                tmin=np.float(events_cond["onsets_s"]),
                tmax=np.float(events_cond["onsets_s"]) + 5 * 60,
            )
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
            all_epochs.append(epochs)

    # Save epochs for this part
    all_epochs = mne.epochs.concatenate_epochs(all_epochs)
    all_epochs = all_epochs.resample(250)
    all_epochs.save(opj(outpath, p + "-epo.fif.gz"), overwrite=True)
