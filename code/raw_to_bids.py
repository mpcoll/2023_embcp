# -*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2023-02-28 11:04:40
Project: EMBCP EEG
Description: Convert raw data to BIDS format
"""
import mne
import os
import numpy as np
import pandas as pd
import bioread
from scipy.signal import decimate
from os.path import join as opj
import shutil
import matplotlib.pyplot as plt
import json
from mne_bids.copyfiles import copyfile_brainvision


###################################################################
# Create dataset level json
###################################################################

# Load delivered stimulation intensity

basepath = "/Volumes/mpx6/2023_embcp"
stim_int = pd.read_csv(opj(basepath, "sourcedata/orig_stim.csv"))

# Get all participants
part = [f for f in os.listdir(opj(basepath, "sourcedata")) if "0" in f]

bidsout = opj(basepath)
derivout = opj(basepath, "derivatives")


# Helperunction to write to json files
def writetojson(outfile, path, content):
    """
    Helper to write a dictionnary to json
    """
    data = os.path.join(path, outfile)
    if not os.path.isfile(data):
        with open(data, "w") as outfile:
            json.dump(content, outfile)
    else:
        print("File " + outfile + " already exists.")


# _________________________________________________________________
# Dataset description JSON
dataset_description = {
    "Name": "Pain-Audio EEG",
    "BIDSVersion": "1.3.0",
    "Authors": ["MP Coll", "Z Walden", "L Yamani", "M Roy"],
    "EthicsApprovals": ["McGill ethics committee"],
}
writetojson("dataset_description.json", bidsout, dataset_description)


# _________________________________________________________________
# Participants csv/json

# Load participants file
partdf = pd.read_csv(opj(basepath, "sourcedata/participants.csv"))
partdf["participant_id"] = [
    "sub-" + str(p).zfill(3) for p in partdf["participant_id"]]

# Read testing notes for each participant and add to csv
partdf["testcomments"] = 9999
for p in part:
    # Get text file
    txtfile = [
        f for f in os.listdir(opj(basepath, "sourcedata/", str(p))) if f.endswith(".txt")
    ][0]

    # Read text file
    with open(opj(basepath, "sourcedata", str(p), txtfile), "r") as file:
        comment = file.read().replace("\n", " ")
    partdf["testcomments"][
        partdf["participant_id"] == "sub-" + str(p).zfill(3)
    ] = comment

# Drop pilot and incomplete participants
partdf = partdf[partdf["testcomments"] != 9999]

# Save
partdf.to_csv(opj(basepath, "participants.tsv"), sep="\t", index=False)


# _________________________________________________________________
# Score questionnaires

# PCS
pcsraw = pd.read_csv(opj(basepath, "sourcedata/quest_pcs.csv"))

pcsraw.columns = (
    ["participant_id"] + ["pcs_q" + str(i)
                          for i in range(1, 14)] + ["pcs_total"]
)
pcsraw["participant_id"] = [
    "sub-" + str(i).zfill(3) for i in pcsraw["participant_id"]]
# Calculate subscales
pcsraw["pcs_rum"] = (
    pcsraw["pcs_q8"] + pcsraw["pcs_q9"] + pcsraw["pcs_q10"] + pcsraw["pcs_q11"]
)
pcsraw["pcs_mag"] = pcsraw["pcs_q6"] + pcsraw["pcs_q7"] + pcsraw["pcs_q13"]
pcsraw["pcs_hel"] = (
    pcsraw["pcs_q1"]
    + pcsraw["pcs_q2"]
    + pcsraw["pcs_q3"]
    + pcsraw["pcs_q4"]
    + pcsraw["pcs_q5"]
    + pcsraw["pcs_q12"]
)

partdf = pd.merge(partdf, pcsraw, on="participant_id")

# Big5
bigraw = pd.read_csv(opj(basepath, "sourcedata/quest_big5.csv"))
bigraw.columns = (
    ["participant_id"]
    + ["bg5_q" + str(i) for i in range(1, 45)]
    + ["bg5_ext", "bg5_agr", "bg5_con", "bg5_neu", "bg5_ope"]
)
bigraw["participant_id"] = [
    "sub-" + str(i).zfill(3) for i in bigraw["participant_id"]]

partdf = pd.merge(partdf, bigraw, on="participant_id")

# STAI-state
stais = pd.read_csv(opj(basepath, "sourcedata/quest_stais.csv"))
stais.columns = (
    ["participant_id"] + ["stais_q" + str(i)
                          for i in range(1, 21)] + ["stais_total"]
)
stais["participant_id"] = [
    "sub-" + str(i).zfill(3) for i in stais["participant_id"]]

partdf = pd.merge(partdf, stais, on="participant_id")


# STAI-trait
stait = pd.read_csv(opj(basepath, "sourcedata/quest_stait.csv"))
stait.columns = (
    ["participant_id"] + ["stait_q" + str(i)
                          for i in range(1, 21)] + ["stait_total"]
)
stait["participant_id"] = [
    "sub-" + str(i).zfill(3) for i in stait["participant_id"]]

partdf = pd.merge(partdf, stait, on="participant_id")
partdf = partdf.replace(np.nan, "n/a")
partdf.to_csv(opj(basepath, "participants.tsv"), sep="\t", index=False)


# Json
partvar = {
    "age": {"Description": "age in years nan if not reported)"},
    "sex": {"Description": "M if male, F female, nan not reported)"},
    "testdate": {"Description": "Test date"},
    "birthdate": {"Description": "Birth date"},
    "testcomments": {"Description": "Experimenter notes"},
    "pcs_total": {"Description": "Pain Catastrophizing Scale total score"},
    "pcs_rum": {"Description": "Pain Catastrophizing Scale rumination score"},
    "pcs_mag": {"Description": "Pain Catastrophizing Scale magnification score"},
    "pcs_hel": {"Description": "Pain Catastrophizing Scale helplessness score"},
    "pcs_qX": {"Description": "Pain Catastrophizing Scale question X"},
    "pcs_q2": {"Description": "Pain Catastrophizing Scale question 2"},
    "pcs_q3": {"Description": "Pain Catastrophizing Scale question 3"},
    "pcs_q4": {"Description": "Pain Catastrophizing Scale question 4"},
    "pcs_q5": {"Description": "Pain Catastrophizing Scale question 5"},
    "pcs_q6": {"Description": "Pain Catastrophizing Scale question 6"},
    "pcs_q7": {"Description": "Pain Catastrophizing Scale question 7"},
    "pcs_q8": {"Description": "Pain Catastrophizing Scale question 8"},
    "pcs_q9": {"Description": "Pain Catastrophizing Scale question 9"},
    "pcs_q10": {"Description": "Pain Catastrophizing Scale question 10"},
    "pcs_q11": {"Description": "Pain Catastrophizing Scale question 11"},
    "pcs_q12": {"Description": "Pain Catastrophizing Scale question 12"},
    "bg5_qX": {"Description": "Big5 question X"},
    "bg5_ext": {"Description": "Big5 extraversion score"},
    "bg5_agr": {"Description": "Big5 agreeableness score"},
    "bg5_con": {"Description": "Big5 conscientiousness score"},
    "bg5_neu": {"Description": "Big5 neuroticism score"},
    "bg5_ope": {"Description": "Big5 openness score"},
    "stais_total": {"Description": "State-Trait Anxiety Inventory state total score"},
    "stais_qX": {"Description": "State-Trait Anxiety Inventory state question X"},
    "stait_total": {"Description": "State-Trait Anxiety Inventory trait total score"},
    "stait_qX": {"Description": "State-Trait Anxiety Inventory trait question X"},

}
writetojson("participants.json", bidsout, partvar)


# _________________________________________________________________
# EEG Json

# Task description

fcond = dict()

fcond['rest'] = {
    "TaskName": "rest",
    "TaskDescription": """
                             Starts with 5 mins resting state
                             """,
    "Instructions": """
         restingstate: We will now begin recording your brain activity during resting-state.
         Please make sure that you are comfortably seated facing the computer monitor with your eyes fixated on the cross that will appear in the centre of the screen.
         Please remain relaxed, being careful not to clench your jaw or tap your feet, and please blink as you normally would.
         This trial will last 5 minutes. Let us know when you are ready to begin
          """,
    "InstitutionName": "McGill",
    "EEGChannelCount": 64,
    "EOGChannelCount": 2,
    "ECGChannelCount": 0,
    "TriggerChannelCount": 1,
    "EEGPlacementScheme": "10-10",
    "EEGReference": "FCz",
    "EEGGround": "AFz",
    "SamplingFrequency": 500,
    "SoftwareFilters": {"lowapss": {"half-amplitude cutoff (Hz)": 250}},
    "PowerLineFrequency": 60,
    "Manufacturer": "BrainVision",
    "RecordingType": "continuous",
}


fcond['thermalrate'] = {
    "TaskName": "thermalrate",
    "TaskDescription": """
                              8 mins of continous heat stimulation with continous rating
                             """,
    "Instructions": """
        thermalrate: You will now be presented with a continuous heat
          stimulation where the temperature will vary. For this trial,
          we ask you to please continuously rate how PAINFUL the stimulation
          is based on the scale below. 0 = No pain at all 100 = Very faint pain
          200 = Exteremely painful This trial will last 8 minutes. Let us know when you are ready to begin.
          """,
    "InstitutionName": "McGill",
    "EEGChannelCount": 64,
    "EOGChannelCount": 2,
    "ECGChannelCount": 0,
    "TriggerChannelCount": 1,
    "EEGPlacementScheme": "10-10",
    "EEGReference": "FCz",
    "EEGGround": "AFz",
    "SamplingFrequency": 500,
    "SoftwareFilters": {"lowapss": {"half-amplitude cutoff (Hz)": 250}},
    "PowerLineFrequency": 60,
    "Manufacturer": "BrainVision",
    "RecordingType": "continuous",
}


fcond['thermal'] = {
    "TaskName": "thermal",
    "TaskDescription": """
                              8 mins of continous heat stimulation
                             """,
    "Instructions": """
         thermal: You will now be presented with a continuous heat stimulation where the temperature will vary.
          This trial will last 8 minutes. Let us know when you are ready to begin
         audionorate: You will now be presented with a continuous sound
          stimulation where the loudness will vary. This trial will last 8 minutes.
          Let us know when you are ready to begin
          """,
    "InstitutionName": "McGill",
    "EEGChannelCount": 64,
    "EOGChannelCount": 2,
    "ECGChannelCount": 0,
    "TriggerChannelCount": 1,
    "EEGPlacementScheme": "10-10",
    "EEGReference": "FCz",
    "EEGGround": "AFz",
    "SamplingFrequency": 500,
    "SoftwareFilters": {"lowapss": {"half-amplitude cutoff (Hz)": 250}},
    "PowerLineFrequency": 60,
    "Manufacturer": "BrainVision",
    "RecordingType": "continuous",
}


fcond['auditoryrate'] = {
    "TaskName": "auditoryrate",
    "TaskDescription": """
                              8 mins of continous audio stimulation with continous ratings
                             """,
    "Instructions": """
            audiorate: You will now be presented with a continuous sound
          stimulation where the loudness will vary. For this trial, we ask
          you to please continuously rate how UNPLEASANT the stimulation is
          based on the scale below. 0 = Not unpleasant at all 100 = Very faint
          unpleasantness 200 = Exteremely unpleasant This trial will last 8 minutes.
          Let us know when you are ready to begin
          """,
    "InstitutionName": "McGill",
    "EEGChannelCount": 64,
    "EOGChannelCount": 2,
    "ECGChannelCount": 0,
    "TriggerChannelCount": 1,
    "EEGPlacementScheme": "10-10",
    "EEGReference": "FCz",
    "EEGGround": "AFz",
    "SamplingFrequency": 500,
    "SoftwareFilters": {"lowapss": {"half-amplitude cutoff (Hz)": 250}},
    "PowerLineFrequency": 60,
    "Manufacturer": "BrainVision",
    "RecordingType": "continuous",
}


fcond['auditory'] = {
    "TaskName": "auditory",
    "TaskDescription": """
                              8 mins of continous audio stimulation
                             """,
    "Instructions": """
         audionorate: You will now be presented with a continuous sound
          stimulation where the loudness will vary. This trial will last 8 minutes.
          Let us know when you are ready to begin
          """,
    "InstitutionName": "McGill",
    "EEGChannelCount": 64,
    "EOGChannelCount": 2,
    "ECGChannelCount": 0,
    "TriggerChannelCount": 1,
    "EEGPlacementScheme": "10-10",
    "EEGReference": "FCz",
    "EEGGround": "AFz",
    "SamplingFrequency": 500,
    "SoftwareFilters": {"lowapss": {"half-amplitude cutoff (Hz)": 250}},
    "PowerLineFrequency": 60,
    "Manufacturer": "BrainVision",
    "RecordingType": "continuous",
}


# Event dict with code : [name, duration]
events_dict = {
    "Description": {'description': """
                                Markers indicate the start and end of the long
                                5-8 minutes segments. For segments with ratings,
                                intermediary markers indicate the switches in
                                intensity. Segements were always presented in
                                the same order: rest/heat/audio/heatrate/audiorate.
                                See pain-audio_events.tsv for details.
                                """},
    "trig_label": {"Description": "Trigger label"},
    "condition": {"Description": "Condition"},
    "onsets_s": {"Description": "Onset in seconds"},
    14: {"trial_type": "Start of segment /intensity swtich"},
    12: {"trial_type":  "End of segment"},
    15: {"trial_type": "Intensity switch"},
}


stim_dict = {
    "SamplingFrequency": 500,
    "StartTime": 0,
    "Columns": ["pain_rating", "audio_rating", "stim"]
}


def make_chan_file(raw):
    """
    Helper function to create a channel description file from raw eeg data
    """
    types = []
    notes = []
    units = []
    for c in raw.info["ch_names"]:

        if "EOG" in c:
            types.append("EOG")
            notes.append("bipolar")
            units.append("microV")
        elif c in ['pain_rating', 'audio_rating', 'stim']:
            types.append("MISC")
            if "pain" in c:
                notes.append("Pain rating")
            elif "audio" in c:
                notes.append('Audio rating')
            elif "stim" in c:
                notes.append('Stimulation intensity')
            units.append("0-100")
        else:
            types.append("EEG")
            notes.append("10-10")
            units.append("microV")

    chan_desc = pd.DataFrame(
        data={
            "name": raw.info["ch_names"],
            "type": types,
            "units": units,
            "description": notes,
        }
    )
    return chan_desc


###################################################################
# Import files, rename and copy to BIDS, create event frame for EEG
# and behav frame for ratings and audiostimulation.
###################################################################


# Collect figures of ratings/triggers to make sure all ok
fig_pain, axes_pain = plt.subplots(7, 7, figsize=(30, 30))
axes_pain = axes_pain.flatten()

fig_audio, axes_audio = plt.subplots(7, 7, figsize=(30, 30))
axes_audio = axes_audio.flatten()

if not os.path.exists(opj(derivout)):
    os.makedirs(opj(derivout))

# Load and rename files
for pidx, p in enumerate(part):
    # Create bids folders
    bsub = "sub-" + p
    if not os.path.exists(opj(bidsout, bsub)):
        os.makedirs(opj(bidsout, bsub))
        os.makedirs(opj(bidsout, bsub, "eeg"))

    # # Copy raw eeg
    # for ext in [".vhdr"]:
    #     copyfile_brainvision(
    #         opj(basepath, "raw/", p, "EEG", p + ext),
    #         opj(bidsout, bsub, "eeg", bsub + "_task-painaudio_eeg" + ext),
    #     )

    # Load EEG to create events frame
    raw = mne.io.read_raw_brainvision(
        opj(basepath, "sourcedata/", p, "EEG", p + ".vhdr"),
        eog=["VEOG", "HEOG"])

    events = mne.events_from_annotations(raw)[0]
    events = [e for e in events if e[2] < 20]
    endcode = 12
    # Remove some extra events due to false starts
    if p == "021":
        events = events[1:]
    if p == "028":
        events = events[2:]
    if p == "011":
        events = events[1:]
    if p == "005":
        events[-2][2] = 14
        events[-1][2] = 12
        del events[-3]
    if p == "022":
        events = events[2:]
    if p == "018":
        del events[-3]
        del events[-3]

    # Find start/end
    lastfound = -1
    byepochs = []
    count = 0
    for idx, e in enumerate(list(events)):
        evcode = e[2]
        # IF end of epoch, find next end of epoch
        if evcode == endcode:
            byepochs.append(events[lastfound + 1: idx + 1])
            lastfound = idx

    # Check all epochs are present
    assert len(byepochs) == 5

    # Check durations of epochs make sense
    for i in range(len(byepochs)):
        dur = float(byepochs[i][-1][0] - byepochs[i]
                    [0][0]) / raw.info["sfreq"] / 60
        assert (dur < 8.4) & (dur > 4.9)

    # Create events df
    events_frame = pd.DataFrame(
        data={
            "onset": [e[0] for e in events],
            "duration": 0,
            "trig_label": [e[2] for e in events],
        }
    )

    # Add labels to  data frame
    epochs_labels = [
        ["rest_start", "rest_end"],
        ["thermal_start", "thermal_end"],
        ["auditory_start", "auditory_end"],
        ["thermalrate_start", "thermalrate_end"],
        ["auditoryrate_start", "auditoryrate_end"],
    ]

    label = np.asarray(["Na"] * len(events)).astype("<U30")
    for i in range(len(byepochs)):
        for idx, e in enumerate(events):
            end = byepochs[i][-1][0]
            start = byepochs[i][0][0]
            if i == 0 & e[2] == 99999:
                start = byepochs[i][idx + 1][0]
                print(start)
            if e[0] == start:
                label[idx] = epochs_labels[i][0]
                idx_start = idx
            elif e[0] == end:
                label[idx] = epochs_labels[i][1]
                idx_end = idx

        label[idx_start + 1: idx_end] = epochs_labels[i][0].split("_")[0]

    events_frame["condition"] = label
    events_frame["onsets_s"] = events_frame["onset"] / raw.info["sfreq"]

    # Rating data
    if p == "002":  # Files are combined for this sub
        audio_pain = bioread.read(
            opj(basepath, "sourcedata/002/BIOPAC/" +
                "002_thermal_audio_trial.acq")
        )
        audiodata = audio_pain.channels[2].data[1345791:]
        paindata = audio_pain.channels[2].data[0:1345790]
        paintrig = audio_pain.channels[0].data[0:1345790]
        audiotrig = audio_pain.channels[0].data[1345791:]
        srate = audio_pain.samples_per_second

    else:
        audio = bioread.read(
            opj(basepath, "sourcedata/", p, "BIOPAC", p + "_audio_trial.acq"))
        pain = bioread.read(
            opj(basepath, "sourcedata/", p, "BIOPAC", p + "_thermal_trial.acq")
        )

        audiodata = audio.channels[1].data * 4
        audiotrig = audio.channels[0].data
        paindata = pain.channels[1].data * 4
        srate = audio.samples_per_second
        paintrig = pain.channels[0].data

    if p == "024":
        audiotrig = np.concatenate([paintrig, audiotrig])
        audiotrig[:1140496] = 0
        audiodata = np.concatenate([paindata, audiodata])
        audiodata[:1140496] = 0
        audiodata = np.pad(audiodata, (1000, 0))

    # Recording was started late for this part, pad start with 0
    if p == "011":
        paindata = np.pad(paindata, (int(500 * srate - len(paindata) + 10), 0))
        paintrig = np.pad(paintrig, (int(500 * srate - len(paintrig) + 10), 0))
        paintrig[1] = 50

    # Find start/end trigger
    start = np.where(np.abs(np.diff(audiotrig)) > np.max(audiotrig) / 2)[0][0]
    startend = [start, int(start + 500 * srate)]
    assert len(startend) == 2

    # Keep only trial data
    aud_rating = audiodata[startend[0]: startend[1]]

    # Make sure length is ok
    assert round(len(aud_rating) / srate) == 500

    # Remove extra samples
    aud_rating = aud_rating[: int(srate * 500)]

    # Downsample to 500 Hz to match EEG and save space
    aud_rating = decimate(aud_rating, int(srate / 500))

    # Same for pain rating
    start = np.where(np.abs(np.diff(paintrig)) > np.max(paintrig) / 2)[0][0]
    startend = [start, int(start + 500 * srate)]
    assert len(startend) == 2

    pain_rating = paindata[startend[0]: startend[1]]
    assert round(len(pain_rating) / srate) == 500

    pain_rating = pain_rating[: int(srate * 500)]
    pain_rating = decimate(pain_rating, int(srate / 500))

    assert len(pain_rating) == len(aud_rating)

    # Add the eeg trigger to make sure everything is aligned
    start = np.where(events_frame["condition"] == "thermalrate_start")[0][0]
    end = np.where(events_frame["condition"] == "thermalrate_end")[0][0]
    triggers = np.asarray(events_frame["onset"])[start:end]
    triggers_pain = triggers - triggers[0]

    triggers_array = np.zeros((250000,))
    triggers_array[triggers_pain] = 1

    # Plot to eyeball
    axes_audio[pidx].plot(aud_rating, label="audio intensity rating")
    axes_audio[pidx].plot(stim_int, label="sound intensity")
    axes_audio[pidx].set_xlabel("samples")
    axes_audio[pidx].set_ylabel("rating")
    axes_audio[pidx].set_title(bsub)
    if pidx == 0:
        axes_audio[pidx].legend()

    axes_pain[pidx].plot(pain_rating, label="pain intensity rating")
    axes_pain[pidx].plot(stim_int, label="heat intensity")
    for t in triggers_pain:
        axes_pain[pidx].axvline(t, linestyle="--", color="k")
    axes_pain[pidx].set_xlabel("samples")
    axes_pain[pidx].set_ylabel("rating")
    axes_pain[pidx].set_title(bsub)
    if pidx == 0:
        axes_pain[pidx].legend()

    # Create a behavioural frame to save
    behav_out = pd.DataFrame(
        data={
            "pain_rating": pain_rating,
            "audio_rating": aud_rating,
            "stim_intensity": stim_int["orig_stim"],
            "eeg_triggers": triggers_array,
        }
    )

    for t in ['rest_start', 'thermal_start', 'auditory_start',
              'thermalrate_start', 'auditoryrate_start']:
        start_s = events_frame[events_frame['condition']
                               == t]['onsets_s'].values[0]
        end_s = events_frame[events_frame['condition'] ==
                             t.replace('start', 'end')]['onsets_s'].values[0]

        if t != 'rest_start':
            task_raw = raw.copy().crop(tmin=start_s, tmax=start_s+499.998)
        else:
            task_raw = raw.copy().crop(tmin=start_s, tmax=start_s+299.9998)

        events_task = events_frame.loc[(events_frame['onsets_s'] >= start_s) & (
            events_frame['onsets_s'] <= end_s)]

        # Save to bids data
        events_frame.to_csv(
            opj(bidsout, bsub, "eeg", bsub + "_task-"
                + t.replace("_start", "") + "_events.tsv"),
            sep="\t",
            index=False,
        )
        writetojson(bsub + "_task-" + t.replace("_start", "") + "_eeg.json",
                    opj(bidsout, bsub, "eeg"), fcond[t.replace("_start", "")])
        writetojson(
            bsub +
            "_task-" + t.replace("_start", "") + "_events.json",
            opj(bidsout, bsub, "eeg"), events_dict
        )

        # Add stimulation and ratings as misc channels

        if t != "rest_start":

            # Create channels for behavioural data
            rate_info = mne.create_info(
                ["remove", "pain_rating", "audio_rating", "stim"],
                raw.info["sfreq"],
                ["eeg", "misc", "misc", "misc"],
            )

            if "rate" in t:
                prate_chan = mne.io.RawArray(
                    np.vstack(
                        [
                            behav_out["pain_rating"],
                            behav_out["pain_rating"],
                            behav_out["audio_rating"],
                            behav_out["stim_intensity"],
                        ]
                    ),
                    info=rate_info,
                )
            else:
                prate_chan = mne.io.RawArray(
                    np.vstack(
                        [
                            np.zeros((1, len(task_raw.times))),
                            np.zeros((1, len(task_raw.times))),
                            np.zeros((1, len(task_raw.times))),
                            behav_out["stim_intensity"],
                        ]
                    ),
                    info=rate_info,
                )
            # Add to raw
            task_raw.load_data().add_channels([prate_chan])

            # Drop the mock eeg channel
            task_raw.drop_channels(["remove"])

        else:
            # Add empty rating channels for consistency
            rate_info = mne.create_info(
                ["remove", "pain_rating", "audio_rating", "stim"],
                raw.info["sfreq"],
                ["eeg", "misc", "misc", "misc"],
            )

            prate_chan = mne.io.RawArray(
                np.zeros((4, len(task_raw.times))), info=rate_info)

            task_raw.load_data().add_channels([prate_chan])

            # Drop the mock eeg channel
            task_raw.drop_channels(["remove"])

        mne.export.export_raw(opj(bidsout, bsub, "eeg", bsub + "_task-"
                                  + t.replace("_start", "") + "_eeg.vhdr"), task_raw,
                              add_ch_type=True, overwrite=True)

        behav_out.to_csv(
            opj(bidsout, bsub, "eeg", bsub + "_task-"
                + t.replace("_start", "") + "_stim.tsv.gz"), sep="\t",
            index=False
        )

        writetojson(bsub + "_task-"
                    + t.replace("_start", "")
                    + "_stim.json", opj(bidsout, bsub, "eeg"),
                    stim_dict
                    )

    # Create channel file
    chanfile = make_chan_file(task_raw)
    chanfile.to_csv(
        opj(bidsout, bsub, "eeg", bsub + "_channels.tsv"), sep="\t",
        index=False)

# Save verification plots
fig_pain.tight_layout()
fig_pain.savefig(opj(basepath, "derivatives/import_pain_ratings_check.png"))
fig_audio.tight_layout()
fig_audio.savefig(opj(basepath, "derivatives/import_audio_ratings_check.png"))

# Write readme
with open(opj(basepath, 'README.md'), 'w') as f:
    f.write('Experiment performed at McGILL University, Montreal, Canada')
    f.write('readme')
