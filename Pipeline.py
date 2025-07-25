
import os 
import mne
import numpy as np
import pandas as pd
import pickle
import gc
from config import data_dir
from functions import preprocess, get_noisyICs, interpolate_HANoise

def pre_ica_denoise(subject_id, lowPassFilter = None, prestim_duration = 0):


    with open(os.path.join( data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    
    bads_channel= all_bads[subject_id]['channel_names']
    bad_trials= all_bads[subject_id]['trial_numbers']

    raw = load_data()
    
    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    events = mne.find_events(raw)
    all_events = sub.get_all_events_times(events).dropna()
    alltrials = sub.segment_stimRt(raw, all_events, bad_trials, prestim_duration)
    pre_ica_data = mne.concatenate_raws(alltrials)
    # interpolate bridged channels
    pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(pre_ica_data, bridged_channels['bridged_idx'], bad_limit=4) 
    return pre_ica_data

def pre_HA_denoise(id, lowPassFilter = None):

    with open(os.path.join( data_dir, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join( data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    path_ic = os.path.join(data_dir, f'S{id}_ica_infomax.fif')
    bads_channel= all_bads[id]['channel_names']
    bad_trials= all_bads[id]['trial_numbers']
    noisy_components = all_bads[id]['noisy_components']
    bridged_channels= all_bridged_channels[id] 
    sub = preprocess(id)
    raw = sub.load_data()

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    events = mne.find_events(raw)
    all_events = sub.get_all_events_times( events).dropna()
    all_trials = sub.segment_stimRt(raw, all_events, bad_trials)
    pre_ica_data= mne.concatenate_raws(all_trials)
    # 4. ICA cleaning
    ica = mne.preprocessing.read_ica(path_ic)
    ica.exclude = noisy_components
    ica.apply(pre_ica_data)
    # interpolate bridged channels
    pre_HA_denoise = mne.preprocessing.interpolate_bridged_electrodes(pre_ica_data, bridged_channels['bridged_idx'], bad_limit=4) 
    # 5. interpolate bad channels
    pre_HA_denoise.interpolate_bads()
    # 6. re-reference to average
    pre_HA_denoise.set_eeg_reference(ref_channels='average')
    # 7. z-score
    data = pre_HA_denoise.get_data()
    means = data.mean(axis=1, keepdims=True)
    stds  = data.std(axis=1, keepdims=True)
    pre_HA_denoise._data = (data - means) / stds

    return pre_HA_denoise

 

def ICA_denoise(id, lowPassFilter = None, n_components=None, decim=2, ica_name = 'ica_infomax', overwrite = False):
    ICA_path = os.path.join(data_dir, f'S{id}_{ica_name}.fif')
    if os.path.exists(ICA_path) and overwrite == False:
        print(f'ICA already exists for subject {id}, skipping ICA computation.')
        return 
    pre_ica_data = pre_ica_denoise(id, lowPassFilter = lowPassFilter)
    ica = mne.preprocessing.ICA(n_components = n_components, method= 'infomax', fit_params=dict(extended=True))
    ica.fit(pre_ica_data, decim=decim)
    ica.save(ICA_path, overwrite=True)
    del ica, pre_ica_data
    gc.collect()
    return

def detect_HA_outliers(subject, threshold=6):

    datanorm = pre_HA_denoise(subject , 30)
    annotation = datanorm._annotations
    df = annotation.to_data_frame()
    df = df[df['duration'] != 0]
    df= df.reset_index(drop=True)
    # ----- Extract epochs and data -----
    all_raws = []
    all_data = []
    all_description = []
    for i, duration in enumerate(df['duration']):
        start = np.sum(df['duration'][:i]) if i > 0 else 0
        end = start + duration
        # Crop raw for this trial
        raw_epoch = datanorm.copy().crop(tmin=start, tmax=end)
        all_raws.append(raw_epoch)
        data = raw_epoch.get_data(picks='eeg')
        all_data.append(data)
        all_description.append(df['description'][i])

    # ----- Compute thresholds -----
    concat = datanorm.copy().get_data(picks='eeg')
    mean = np.mean(concat, axis=1)
    std = np.std(concat, axis=1)

    channel_thresholds = mean + threshold * std  # shape: (n_channels,)

    # ----- Detect threshold crossings -----
    n_trials = len(all_data)
    n_channels = concat.shape[0]
    # Vectorized detection for speed
    detection = np.zeros((n_channels, n_trials), dtype=bool)
    for i, trial in enumerate(all_data):
        # trial shape: (n_channels, n_times)
        detection[:, i] = np.any(np.abs(trial) > channel_thresholds[:, None], axis=1)

    return detection


def concat_data(ids,
            ica_name='ica_infomax',
            lowPassFilter_pregICA=30,
            file_name='groupData'):

    # Load once
    with open(os.path.join(data_dir, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join(data_dir, 'new.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    pre_concatenated_data = []

    for subject_id in ids:
        # Per‐subject params
        bads_channel = all_bads[subject_id]['channel_names']
        bad_trials   = all_bads[subject_id]['trial_numbers']
        bridged      = all_bridged_channels[subject_id]
        bad_components = all_bads[subject_id]['noisy_components']
        detected_noise = all_bads[subject_id]['detection']

        # 1. load & preprocess
        sub = preprocess(subject_id)
        raw = sub.load_data()
        raw.info['bads'] = bads_channel

        # 2. filter
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=lowPassFilter_pregICA)

        # 3. epoch & drop noisy trials and wrong trials, 
        events    = mne.find_events(raw)
        all_events = sub.get_all_events_times( events).dropna()
        all_trials    = sub.segment_stimRt(raw, all_events, bad_trials)

        # 4. ICA cleaning
        ica_path = os.path.join(data_dir, f'S{subject_id}_{ica_name}.fif')
        if not os.path.exists(ica_path):
            print(f'ICA missing for subject {subject_id}; skipping.')

        ica = mne.preprocessing.read_ica(ica_path)

        noisy = bad_components
        ica.exclude = noisy
        all_trials_clean = []
        for trial in all_trials:
            trialclean = ica.apply(trial)
            all_trials_clean.append(trialclean)
        
        

        all_trials_interp = interpolate_HANoise(all_trials_clean, detected_noise)
                
        new_raw = mne.concatenate_raws(all_trials_interp)
        # 5–7. interpolate & re‐ref
        new_raw = mne.preprocessing.interpolate_bridged_electrodes(
            new_raw, bridged['bridged_idx'], bad_limit=4
        )

        new_raw.set_eeg_reference(ref_channels='average')

        # 8. z‐score
        data = new_raw.get_data()
        means = data.mean(axis=1, keepdims=True)
        stds  = data.std(axis=1, keepdims=True)
        new_raw._data = (data - means) / stds
        # store and then clear
        pre_concatenated_data.append(new_raw)

        # --- clear out all the big locals ---
        del (sub, raw, events, all_events, ica,
                data, means, stds, bridged,
                bads_channel, bad_trials)
        gc.collect()

    # concatenate and save
    concat = mne.concatenate_raws(pre_concatenated_data)
    concat.save(os.path.join(data_dir, f'{file_name}.fif'), overwrite=True)

    return concat


