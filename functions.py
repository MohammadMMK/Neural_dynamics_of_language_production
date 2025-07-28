from sklearn.utils import resample
from config import data_dir # this is the directory where the all needed data is stored
import os 
import mne
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from mne.preprocessing import compute_bridged_electrodes
from unidecode import unidecode
import gc
import pickle

def load_data(id=None, path_input=None, add_montage=True):
    """
    Loads EEG data from a BDF file, optionally adding a rotated montage.
    Parameters
    ----------
    id : str or int, optional
        Identifier used to select the EEG data file from `data_dir`. If provided, the function searches for a file containing `id` in its name and ending with '.bdf'.
    path_input : str, optional
        Direct path to the EEG data file. If provided, this path is used to load the data.
    add_montage : bool, default=True
        If True, adds a rotated montage to the loaded EEG data using channel locations from 'Head128elec.xyz'.
    Returns
    -------
    raw : mne.io.Raw
        The loaded EEG data as an MNE Raw object, with montage applied if `add_montage` is True.
    Raises
    ------
    ValueError
        If neither `id` nor `path_input` is provided.
    FileNotFoundError
        If the specified EEG data file does not exist.
    Notes
    -----
    - Check that the `data_dir` variable is set correctly in config file.
    - Channel locations are loaded from 'Head128elec.xyz' in `data_dir`.
    """

    if path_input is None:
        path = os.path.join(data_dir, [f for f in os.listdir(data_dir) if str(id) in f and f.endswith('.bdf')][0])
    if id is None:
        path = path_input
    if id is None and path_input is None:
        raise ValueError("Either id or path_input must be provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    raw = mne.io.read_raw(path, preload=True)
    if add_montage:
        # add montage to EEG data ( the original should be rotated 90 degrees to fit correctly)
        ch_location_file = 'Head128elec.xyz'
        ch_location_path = os.path.join(data_dir, ch_location_file)
        # Load channel locations and create montage
        xyz_data = pd.read_csv(ch_location_path, delim_whitespace=True, skiprows=1, header=None)
        rotated_montage = mne.channels.make_dig_montage(
            ch_pos={name: R.from_euler('z', 90, degrees=True).apply(coord)  # Rotate 90 degrees
                    for name, coord in zip(xyz_data[3].values, xyz_data[[0, 1, 2]].values)},
            coord_frame="head"
        )
        # Set the rotated montage to the Raw object
        raw.set_montage(rotated_montage)
    
    return raw

    

def epoching(subject_id, raw, stim="unicity_point", tmin=-0.2, tmax=0.8, baseline=(None, 0), metadata=None):

    events_unicity = mne.find_events(raw)
    if stim == "unicity_point":
        epochs = mne.Epochs(raw, events_unicity, event_id={'High': 1, 'Low': 2}, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, metadata = metadata)
    if stim == "onset_definition":
        all_events = get_all_events_times(subject_id, events_unicity)
        def_onset = all_events['defOnset']
        events_defOnset = []
        for i, onset in enumerate(def_onset):
            onset_sample = int(onset * raw.info['sfreq'])  # Co`nvert to sample index
            id = events_unicity[i, 2] 
            events_defOnset.append([onset_sample, 0, id])  # Use the existing event ID
        events_defOnset = np.array(events_defOnset, dtype=int)
        epochs = mne.Epochs(raw, events_defOnset, event_id={'High': 1, 'Low': 2}, tmin=tmin, tmax=tmax, baseline=baseline, preload=True,metadata = metadata)

    return epochs


def bridged_channels(epochs,   lm_cutoff = 5, epoch_threshold=0.5):

    bridged_idx, ed_matrix  = compute_bridged_electrodes( epochs, lm_cutoff = lm_cutoff, epoch_threshold= epoch_threshold)

    bridged_channels = list(set([channel for pair in bridged_idx for channel in pair]))
    bridged_ch_names = [epochs.ch_names[i] for i  in bridged_channels]
    epochs.info['bads'] += bridged_ch_names

    return bridged_idx, ed_matrix, bridged_ch_names





def get_all_events_times( subject_id, events, path_beh_task=None, path_beh_subject=None):


    # load data from exel file 
    if path_beh_task is None or path_beh_subject is None:
        path_beh_task = os.path.join(data_dir, 'BehavioralResultsDefinition24ss_11_11_2017_RF_2023.xlsx')
        path_beh_subject =  os.path.join(data_dir, 'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx')
    
    # Load Excel sheets
    behavior_tasks = pd.read_excel(path_beh_task, sheet_name='ItemsBalancés')
    behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
    # Filter data for current subject
    subject_key = f'S{subject_id}'
    subject_data = behavior_subjets[behavior_subjets['Sujet'] == subject_key]
    subject_data = subject_data.dropna(subset=['Cible'])
    subject_data = subject_data[subject_data['Ordre'] <= 108]

    # Clean and normalize stimulus duration
    behavior_tasks = behavior_tasks.dropna(subset=['ortho'])
    behavior_tasks['ortho'] = behavior_tasks['ortho'].apply(unidecode)

    sfreq = 512 
    results = {
        'Trial': [],
        'defOnset': [],
        'SecWordOnset': [],
        'LWOnset': [],
        'Respons': [],
        'freqs': [],
        'word': []
    }

    for trial in range(1, 109):
        # Get stimulus word
        word = subject_data[subject_data['Ordre'] == trial]['Cible'].values[0]

        # Lookup in duration data
        stim_info = behavior_tasks[behavior_tasks['ortho'] == word]
        
        total_duration_ms = stim_info['DureeTot_second'].values[0] * 1000
        pu_ms = stim_info['PU_second'].values[0] * 1000
        def2_onset_ms = stim_info['Def2_Audio_Onset'].values[0] * 1000
        lw_onset_ms = stim_info['LW_Onset'].values[0] * 1000
        # Get frequency
        freq = stim_info['Freq_Manulex'].values[0]


        # Get response time
        rt_corrPU_ms = subject_data[subject_data['Ordre'] == trial]['RT_Correct_CorrPU'].values[0]

        # Calculate onsets
        event_time_ms = (events[trial - 1][0] / sfreq) * 1000
        onset_def = event_time_ms - pu_ms
        onset_sec_word = onset_def + def2_onset_ms
        onset_lw = onset_def + lw_onset_ms
        response_time = event_time_ms + rt_corrPU_ms
        # return onset_def, onset_sec_word, onset_lw, response_time, event_time_ms, 
        # Append to results
        results['Trial'].append(trial)
        results['defOnset'].append(onset_def / 1000.0)
        results['SecWordOnset'].append(onset_sec_word / 1000.0)
        results['LWOnset'].append(onset_lw / 1000.0)
        results['Respons'].append(response_time / 1000.0)
        results['freqs'].append(freq)
        results['word'].append(word)

    return pd.DataFrame(results)




def segment_stimRt(subject_id, raw, bad_trials, prestim_duration = 0):


    events = mne.find_events(raw)
    all_events = get_all_events_times(subject_id, events).dropna() # since the wrong answer or no answer trials have na values, at this point we can drop them


    all_trials = []
    for idx, row in all_events.iterrows():

        Tnum = row['Trial']
        if Tnum in bad_trials:
            continue
        start = row['defOnset'] - prestim_duration
        end = row['Respons'] - 0.1 # 0.1 seconds before response to avoid including the response peak
        duration = end - start

        # Copy and crop raw data
        data = raw.copy().crop(start, end)
        
        # Create annotation
        onset_in_cropped = 0  # onset relative to start of cropped data
        annotation = mne.Annotations(onset=[onset_in_cropped],
                                    duration=[duration],
                                    description=[f'S{subject_id}_Trial_{Tnum}'])
        
        # Set annotation to this segment
        data.set_annotations(annotation)

        all_trials.append(data)
    all_trials_return = all_trials
    stimRT_concat = mne.concatenate_raws(all_trials)
    return all_trials_return, stimRT_concat








def pre_ica_denoise(subject_id, lowPassFilter = None, prestim_duration = 0):

    with open(os.path.join( data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    
    bads_channel= all_bads[subject_id]['channel_names']
    bad_trials= all_bads[subject_id]['trial_numbers']

    raw = load_data(subject_id)
    # Get bridged channels
    epochs_unicity = epoching(subject_id, raw, stim  = 'unicity_point' , tmin = -0.5, tmax = 0.2) # a random epoch just for computing the bridged channels
    bridged_idx, ed_matrix, bridged_ch_names = bridged_channels(epochs_unicity,   lm_cutoff = 5, epoch_threshold=0.5)

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    all_trials_list, stimRT_concat = segment_stimRt(subject_id, raw,bad_trials )

    # interpolate bridged channels
    pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(stimRT_concat, bridged_idx, bad_limit=4) 
    return pre_ica_data


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


def load_trials_metadata(id):
    path_beh_subject = os.path.join(data_dir,'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx' ) 

    behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
    # Filter data for current subject
    subject_key = f'S{id}'
    trial_metadata = behavior_subjets[behavior_subjets['Sujet'] == subject_key]
    # remove the rows that Ordre is more than 108
    trial_metadata = trial_metadata[trial_metadata['Ordre'] <= 108]
    trial_metadata = trial_metadata.sort_values(by = 'Ordre')
    trial_metadata.reset_index(drop=True, inplace=True)
    return trial_metadata





def pre_hmp(subject_id, resample=True, n_jobs=1):

    # load the selected bad channels, trials and components
    with open(os.path.join( data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    bads_channel= all_bads[subject_id]['channel_names']
    bad_trials= all_bads[subject_id]['trial_numbers']
    noisy_components = all_bads[subject_id]['noisy_components']

    # load raw data
    raw = load_data(subject_id)
    # Get bridged channels
    epochs_unicity = epoching(subject_id, raw, stim  = 'unicity_point' , tmin = -0.5, tmax = 0.2) # a random epoch just for computing the bridged channels
    bridged_idx, ed_matrix, bridged_ch_names = bridged_channels(epochs_unicity,   lm_cutoff = 5, epoch_threshold=0.5)

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= 30)

    # 3. epoch the data and remove bad trials
    meta_data = load_trials_metadata(subject_id)
    epochs_unicity = epoching(subject_id, raw, stim  = 'unicity_point' , tmin = -0.5, tmax = 2, baseline=None,  metadata= meta_data) 

    # we get new metadata from the epochs because at epoching step there might be some trials that are automatically dropped because of length for example
    # so we need to reset the index of the metadata
    meta_data = epochs_unicity.metadata
    meta_data= meta_data.reset_index(drop=True)

    # bad indices
    bad_indices = meta_data[meta_data['Ordre'].isin(bad_trials)].index
    na_indices = meta_data[meta_data['RT_Correct_CorrPU'].isna()].index
    # Combine bad indices
    remove_indices = bad_indices.union(na_indices)
    # Remove trials (manual selected noisy trials and trials with NaN RT_Correct_CorrPU)
    epochs_unicity.drop(remove_indices)




    # 4. remove noisy components
    path_ic = os.path.join(data_dir, f'S{subject_id}_ica_infomax.fif')
    ica = mne.preprocessing.read_ica(path_ic)
    ica.exclude = noisy_components
    ica.apply(epochs_unicity, exclude=noisy_components)

    # 5. autoreject 
    import autoreject #version 0.3.1 https://autoreject.github.io/
    ar = autoreject.AutoReject(consensus=np.linspace(0, .4, 11), n_jobs=n_jobs, picks = 'eeg') 
    ar.fit(epochs_unicity)  # fit on the first 20 epochs to save time
    epochs_ar, reject_log = ar.transform(epochs_unicity, return_log=True)
    mat = np.delete(reject_log.labels, reject_log.bad_epochs, axis=0)
    percentage_interp = (mat == 2).sum() / mat.size * 100


    # interpolate bridged channels
    epochs_ar = mne.preprocessing.interpolate_bridged_electrodes(epochs_ar, bridged_idx, bad_limit=4) 

    # 5. interpolate bad channels
    epochs_ar.interpolate_bads()
    # 6. re-reference to average
    epochs_ar.set_eeg_reference(ref_channels='average')

    # 7. add metadata to epochs
    epochs_ar.metadata['AoA'] = epochs_ar.events[:, 2]  # Add AoA from events to metadata

    # save the logs 
    logs = {'n_manually_removed_trials': len(bad_trials),
            'n_wrong_answers': len(na_indices),
            'n_autoreject_removed_trials': int(np.sum(reject_log.bad_epochs)),
            'n_manually_removed_channels': len(bads_channel),
            'n_bridged_channels': len(bridged_ch_names),
            'autoreject_interp_percentage': percentage_interp,}
    
    # save the epochs
    path_epochs = os.path.join(data_dir,'Dnl', f'S{subject_id}_epochs_preHMP.fif')
    path_logs = os.path.join(data_dir,'Dnl', f'S{subject_id}_logs_preHMP.pkl')

    # resample epochs to 256 Hz
    if resample:
        epochs_ar = epochs_ar.resample(256)

    epochs_ar.save(path_epochs, overwrite=True)
    with open(path_logs, 'wb') as f:
        pickle.dump(logs, f)







