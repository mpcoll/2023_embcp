"""author: mpcoll."""

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import scipy
from mne.report import Report
from mne.preprocessing import ICA, create_eog_epochs
import matplotlib.pyplot as plt
from autoreject import Ransac
from mne_icalabel import label_components
from meegkit import dss

###############################
# Parameters
###############################
bidsroot = "/media/mp/mpx6/2023_embcp/"
part = [s for s in os.listdir(bidsroot) if "sub-" in s]
derivpath = opj(bidsroot, "derivatives")

param = {
    "visualinspect": False,
}

# Loop participants

# part = [p for p in part if p not in [
#     'sub-012', 'sub-015', 'sub-027', 'sub-023']]


part = [p for p in part if p not in ['sub-015']]


# Create a data frame to collect stats
stats_frame = pd.DataFrame(index=part)


for p in part:
    print(p)
    pdir = opj(derivpath, p, "eeg")
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    # Initialize report
    report = Report(verbose=False, subject=p, title="EEG report for part " + p)

    # Load raw file
    raw = mne.io.read_raw_brainvision(
        opj(bidsroot, p, "eeg", p + "_task-painaudio_eeg.vhdr"),
        eog=["VEOG", "HEOG"]
    )

    # Set channel positions
    raw = raw.set_montage("easycap-M1")

    # _________________________________________________________________
    # Filter and zapline
    # Plot raw spectrum
    plt_psd = raw.copy().plot_psd(fmax=150, show=False)
    report.add_figure(plt_psd, "Raw spectrum")

    # High pass filter
    raw = raw.load_data().filter(0.1, None)

    # remove line noise using mne spectrum fit (not efficient)
    # raw = raw.notch_filter(np.arange(60, 200, 60), method="spectrum_fit",
    #                        filter_length='10s')

    # remove line noise using iterative DSS
    eeg_dat = raw.copy().pick_types(eeg=True).get_data().swapaxes(1, 0)
    out, iterations = dss.dss_line_iter(
        eeg_dat, 60, raw.info["sfreq"], nfft=400)

    out = np.vstack(
        [out.swapaxes(1, 0), raw.copy().pick_types(eog=True).get_data()])
    raw = mne.io.RawArray(out, raw.info)
    raw.set_montage("easycap-M1")

    # Plot spectrum after line noise removal/highpass
    plt_psd = raw.copy().plot_psd(fmax=150, show=False)
    report.add_figure(plt_psd, "Spectrum, line noise removed + highpass")

    # Plot raw data and stop if visual inspection is on
    if param["visualinspect"]:
        raw.copy().load_data().filter(1, 30).plot(
            n_channels=raw.info["nchan"], scalings=dict(eeg=0.000020), block=True
        )

    # _________________________________________________________________
    # Epoch according to condition to remove breaks and pre/post

    events = pd.read_csv(
        opj(bidsroot, p, "eeg", p + "_task-painaudio_events.tsv"), sep="\t"
    )

    # Epochs are 500 s
    events["offset"] = events["onset"] + raw.info["sfreq"] * 500

    # Keep only start of trial
    events = events[
        events.condition.isin(
            [
                "rest_start",
                "thermal_start",
                "auditory_start",
                "thermalrate_start",
                "auditoryrate_start",
            ]
        )
    ]

    events_dict = {
        "rest_start": 1,
        "thermal_start": 2,
        "auditory_start": 3,
        "thermalrate_start": 4,
        "auditoryrate_start": 5,
    }
    events_c = pd.DataFrame(columns=["cue_num", "empty", "sample"])
    events_c["cue_num"] = [events_dict[s] for s in events.condition]
    events_c["sample"] = list(events["onset"])
    events_c["empty"] = 0
    events_epochs = np.asarray(events_c[["sample", "empty", "cue_num"]])

    # Create epochs
    epochs_good = mne.Epochs(
        raw,
        events=events_epochs,
        event_id=events_dict,
        tmin=0,
        tmax=500,
        reject=None,
        baseline=None,
        preload=True,
    )

    # Make a new raw with only times of interest for ICA
    raw_cont = mne.io.RawArray(np.hstack(epochs_good.get_data()), raw.info)

    # _________________________________________________________________
    # Identify bad channels using ransac

    # Epochs
    raw_f = raw_cont.copy().load_data().filter(1, 100).set_eeg_reference("average")
    epochs = mne.make_fixed_length_epochs(raw_f, duration=2).load_data()
    raw_f = None

    ransac = Ransac(verbose=True, picks="eeg", n_jobs=-1)
    ransac.fit(epochs.copy())

    epochs.info["bads"] += ransac.bad_chs_
    raw_cont.info["bads"] += ransac.bad_chs_
    epochs_good.info["bads"] += ransac.bad_chs_

    # Remove very bad epochs for ICA
    epochs.drop_bad(reject=dict(eeg=500e-6))

    # Plot channel variance get z scored variance
    chan_var = scipy.stats.zscore(np.var(raw_cont.get_data()[:64], axis=1))

    fig = plt.figure(figsize=(7, 5))
    plt.title("Zscored channel variance for " + p, size=18)
    plt.xlabel("Channel", fontdict=dict(size=16))
    plt.ylabel("Z-score", fontdict=dict(size=16))
    plt.axhline(y=3, linestyle="--")
    plt.axhline(y=-3, linestyle="--")
    plt.yticks(fontsize=15)
    for i, txt in enumerate(chan_var):
        if raw_cont.info["ch_names"][i] in raw_cont.info["bads"]:
            color = "red"
        else:
            color = "blue"
        plt.annotate(raw_cont.info["ch_names"][i],
                     (np.arange(1, 65)[i], chan_var[i]))
        plt.scatter(np.arange(1, 66)[i], chan_var[i], color=color)
    report.add_figure(fig, "Channel variance")

    # plot channels with bads in red
    fig = raw_cont.plot_sensors(show_names=True)
    report.add_figure(fig, "Sensor positions (bad in red)")

    # Average reference
    epochs_good = epochs_good.set_eeg_reference("average", projection=False)

    # Collect in frame
    stats_frame.loc[p, "n_bad_chans"] = len(epochs_good.info["bads"])

    # _________________________________________________________________
    # ICA
    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=23,
    )
    # Fit ICA
    ica.fit(epochs, decim=4)

    # Run ICA labels
    ica_labels = label_components(epochs, ica, method="iclabel")

    # Remove components labveled as bad with > X robability
    remove = [
        1
        if ic
        in [
            "channel noise",
            "eye blink",
            "muscle artifact",
            "line noise",
            "heart beat",
            "eye movement",
        ]
        and prob > 0.70
        else 0
        for ic, prob in zip(ica_labels["labels"], ica_labels["y_pred_proba"])
    ]
    ica.exclude = list(np.argwhere(remove).flatten())

    # Collect number in frame and report
    stats_frame.loc[p, "n_bads_ica"] = len(ica.exclude)

    report.add_html(pd.DataFrame(ica_labels).to_html(), "ICA labels")

    # Add topo figures to report
    fig = ica.plot_components(show=False, res=25)
    report.add_figure(fig, title="ICA components")
    plt.close("all")

    # Get ICA identified in visual inspection
    ica.apply(epochs_good)

    # Interpolate bad channels after ICA
    epochs_good = epochs_good.interpolate_bads(reset_bads=True)

    # _________________________________________________________________
    # Add the behav to the data and save as raw

    for epo_idx in range(len(epochs_good)):
        epo = epochs_good[epo_idx]

        epo_type = list(events_dict.keys())[
            list(events_dict.values()).index(epo.events[0][2])
        ]

        raw = mne.io.RawArray(epo.get_data()[0, ::][:, :-1], info=epo.info)

        if epo_type != "rest_start":

            # Load behavioural
            behav = pd.read_csv(
                opj(bidsroot, p, "eeg", p + "_task-painaudio_beh.tsv"), sep="\t"
            )

            # Create channels for behavioural data
            rate_info = mne.create_info(
                ["remove", "pain_rating", "audio_rating", "stim"],
                raw.info["sfreq"],
                ["eeg", "misc", "misc", "misc"],
            )

            prate_chan = mne.io.RawArray(
                np.vstack(
                    [
                        behav["pain_rating"],
                        behav["pain_rating"],
                        behav["audio_rating"],
                        behav["stim_intensity"],
                    ]
                ),
                info=rate_info,
            )
            prate_chan = prate_chan.filter(
                raw.info["highpass"], None
            ).set_eeg_reference(ref_channels="average", projection=False)

            # Add to raw
            raw.add_channels([prate_chan])

            # Drop the mock eeg channel
            raw.drop_channels(["remove"])

        else:
            # Rest is only 5 mins
            raw.crop(tmax=5 * 60)
            # Add empty rating channels for consistency
            rate_info = mne.create_info(
                ["remove", "pain_rating", "audio_rating", "stim"],
                raw.info["sfreq"],
                ["eeg", "misc", "misc", "misc"],
            )

            prate_chan = mne.io.RawArray(
                np.zeros((4, len(raw.times))), info=rate_info)
            prate_chan = prate_chan.filter(
                raw.info["highpass"], None
            ).set_eeg_reference(ref_channels="average", projection=False)

            # Add to raw
            raw.add_channels([prate_chan])

            # Drop the mock eeg channel
            raw.drop_channels(["remove"])

        # ______________________________________________________________________
        # Save cleaned data
        raw.save(
            opj(pdir, p + "_task-painaudio_" +
                epo_type + "_cleanedeeg-raw.fif"),
            overwrite=True,
        )

    #  _____________________________________________________________________
    # Save report

    report.save(
        opj(pdir, p + "_task-painaudio_preprocess_report.html"),
        open_browser=False,
        overwrite=True,
    )

# Save statistics
stats_frame.to_csv(opj(derivpath, "preprocess_stats.csv"))
