from sklearn.utils import resample
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))  # Add the parent
import mne
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from mne.preprocessing import compute_bridged_electrodes
from unidecode import unidecode
import gc
import pickle
from config import data_dir # this is the directory where the all needed data is stored


class preprocess:
    def __init__(self, data_dir, id=None):
        self.data_dir = data_dir
        self.id = id

    def load_data(self, path_input=None, add_montage=True):
        if path_input is None:
            path = os.path.join(self.data_dir, [f for f in os.listdir(self.data_dir) if str(self.id) in f and f.endswith('.bdf')][0])
        else:
            path = path_input
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        raw = mne.io.read_raw(path, preload=True)
        if add_montage:
            ch_location_file = 'Head128elec.xyz'
            ch_location_path = os.path.join(self.data_dir, ch_location_file)
            xyz_data = pd.read_csv(ch_location_path, delim_whitespace=True, skiprows=1, header=None)
            rotated_montage = mne.channels.make_dig_montage(
                ch_pos={name: R.from_euler('z', 90, degrees=True).apply(coord)
                        for name, coord in zip(xyz_data[3].values, xyz_data[[0, 1, 2]].values)},
                coord_frame="head"
            )
            raw.set_montage(rotated_montage)
        return raw

    def epoching(self, raw, stim="unicity_point", tmin=-0.2, tmax=0.8, baseline=(None, 0), metadata=None):
        events_unicity = mne.find_events(raw)
        if stim == "unicity_point":
            epochs = mne.Epochs(raw, events_unicity, event_id={'High': 1, 'Low': 2}, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, metadata=metadata)
        elif stim == "onset_definition":
            all_events = self.get_all_events_times(events_unicity)
            def_onset = all_events['defOnset']
            events_defOnset = []
            for i, onset in enumerate(def_onset):
                onset_sample = int(onset * raw.info['sfreq'])
                id = events_unicity[i, 2]
                events_defOnset.append([onset_sample, 0, id])
            events_defOnset = np.array(events_defOnset, dtype=int)
            epochs = mne.Epochs(raw, events_defOnset, event_id={'High': 1, 'Low': 2}, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, metadata=metadata)
        return epochs

    def bridged_channels(self, epochs, lm_cutoff=5, epoch_threshold=0.5):
        bridged_idx, ed_matrix = compute_bridged_electrodes(epochs, lm_cutoff=lm_cutoff, epoch_threshold=epoch_threshold)
        bridged_channels = list(set([channel for pair in bridged_idx for channel in pair]))
        bridged_ch_names = [epochs.ch_names[i] for i in bridged_channels]
        epochs.info['bads'] += bridged_ch_names
        return bridged_idx, ed_matrix, bridged_ch_names

    def get_all_events_times(self, events, path_beh_task=None, path_beh_subject=None):
        if path_beh_task is None or path_beh_subject is None:
            path_beh_task = os.path.join(self.data_dir, 'BehavioralResultsDefinition24ss_11_11_2017_RF_2023.xlsx')
            path_beh_subject = os.path.join(self.data_dir, 'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx')
        behavior_tasks = pd.read_excel(path_beh_task, sheet_name='ItemsBalancés')
        behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
        subject_key = f'S{self.id}'
        subject_data = behavior_subjets[behavior_subjets['Sujet'] == subject_key].dropna(subset=['Cible'])
        subject_data = subject_data[subject_data['Ordre'] <= 108]
        behavior_tasks = behavior_tasks.dropna(subset=['ortho'])
        behavior_tasks['ortho'] = behavior_tasks['ortho'].apply(unidecode)
        sfreq = 512
        results = {'Trial': [], 'defOnset': [], 'SecWordOnset': [], 'LWOnset': [], 'Respons': [], 'freqs': [], 'word': [], 'unicity_point': []}
        for trial in range(1, 109):
            word = subject_data[subject_data['Ordre'] == trial]['Cible'].values[0]
            stim_info = behavior_tasks[behavior_tasks['ortho'] == word]
            pu_ms = stim_info['PU_second'].values[0] * 1000
            def2_onset_ms = stim_info['Def2_Audio_Onset'].values[0] * 1000
            lw_onset_ms = stim_info['LW_Onset'].values[0] * 1000
            freq = stim_info['Freq_Manulex'].values[0]
            rt_corrPU_ms = subject_data[subject_data['Ordre'] == trial]['RT_Correct_CorrPU'].values[0]
            event_time_ms = (events[trial - 1][0] / sfreq) * 1000
            onset_def = event_time_ms - pu_ms
            onset_sec_word = onset_def + def2_onset_ms
            onset_lw = onset_def + lw_onset_ms
            response_time = event_time_ms + rt_corrPU_ms
            results['Trial'].append(trial)
            results['defOnset'].append(onset_def / 1000.0)
            results['SecWordOnset'].append(onset_sec_word / 1000.0)
            results['LWOnset'].append(onset_lw / 1000.0)
            results['Respons'].append(response_time / 1000.0)
            results['freqs'].append(freq)
            results['word'].append(word)
            results['unicity_point'].append(event_time_ms / 1000.0) 
        return pd.DataFrame(results)

    def segment_stimRt(self, raw, bad_trials, prestim_duration=0, stim = 'defOnset'):
        events = mne.find_events(raw)
        all_events = self.get_all_events_times(events).dropna()
        all_trials = []
        for idx, row in all_events.iterrows():
            Tnum = row['Trial']
            if Tnum in bad_trials:
                continue
            if stim == 'defOnset':
                start = row['defOnset'] - prestim_duration
            if stim == 'unicity_point':
                start = row['unicity_point'] - prestim_duration
            end = row['Respons'] - 0.1
            duration = end - start
            data = raw.copy().crop(start, end)
            annotation = mne.Annotations(onset=[0], duration=[duration], description=[f'S{self.id}_Trial_{Tnum}'])
            data.set_annotations(annotation)
            all_trials.append(data)
        stimRT_concat = mne.concatenate_raws(all_trials)
        return all_trials, stimRT_concat

    def pre_ica_denoise(self, lowPassFilter=None, prestim_duration=0):
        with open(os.path.join(self.data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
            all_bads = pickle.load(f)
        bads_channel = all_bads[self.id]['channel_names']
        bad_trials = all_bads[self.id]['trial_numbers']
        raw = self.load_data()
        epochs_unicity = self.epoching(raw, stim='unicity_point', tmin=-0.5, tmax=0.2)
        bridged_idx, _, _ = self.bridged_channels(epochs_unicity)
        raw.info['bads'] = bads_channel
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=lowPassFilter)
        _, stimRT_concat = self.segment_stimRt(raw, bad_trials)
        pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(stimRT_concat, bridged_idx, bad_limit=4)
        return pre_ica_data

    def ICA_denoise(self, lowPassFilter=None, n_components=None, decim=2, ica_name='ica_infomax', overwrite=False):
        ICA_path = os.path.join(self.data_dir, f'S{self.id}_{ica_name}.fif')
        if os.path.exists(ICA_path) and not overwrite:
            print(f'ICA already exists for subject {self.id}, skipping ICA computation.')
            return
        pre_ica_data = self.pre_ica_denoise(lowPassFilter=lowPassFilter)
        ica = mne.preprocessing.ICA(n_components=n_components, method='infomax', fit_params=dict(extended=True))
        ica.fit(pre_ica_data, decim=decim)
        ica.save(ICA_path, overwrite=True)
        del ica, pre_ica_data
        gc.collect()

    def load_trials_metadata(self):
        path_beh_subject = os.path.join(self.data_dir, 'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx')
        behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
        subject_key = f'S{self.id}'
        trial_metadata = behavior_subjets[behavior_subjets['Sujet'] == subject_key]
        trial_metadata = trial_metadata[trial_metadata['Ordre'] <= 108]
        trial_metadata = trial_metadata.sort_values(by='Ordre')
        trial_metadata.reset_index(drop=True, inplace=True)
        return trial_metadata

    @classmethod
    def pre_hmp(cls, subject_id, stim='unicity_point', resample=True, n_jobs=1, save=True):
        from config import data_dir  # Import data_dir from config

        # load the selected bad channels, trials and components
        with open(os.path.join(data_dir, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
            all_bads = pickle.load(f)
        bads_channel = all_bads[subject_id]['channel_names']
        bad_trials = all_bads[subject_id]['trial_numbers']
        noisy_components = all_bads[subject_id]['noisy_components']

        preprocess = cls(data_dir, subject_id)
        # load raw data
        raw = preprocess.load_data()
        # Get bridged channels
        epochs = preprocess.epoching(raw, stim=stim, tmin=-0.5, tmax=0.2)  # a random epoch just for computing the bridged channels
        bridged_idx, ed_matrix, bridged_ch_names = preprocess.bridged_channels(epochs, lm_cutoff=5, epoch_threshold=0.5)

        # 1. remove noisy channels
        raw.info['bads'] = bads_channel

        # 2. Filter the data
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=30)

        # 3. epoch the data and remove bad trials
        meta_data = preprocess.load_trials_metadata()
        epochs = preprocess.epoching(raw, stim=stim, tmin=-0.5, tmax=2, baseline=None, metadata=meta_data)

        # we get new metadata from the epochs because at epoching step there might be some trials that are automatically dropped because of length for example
        # so we need to reset the index of the metadata
        meta_data = epochs.metadata
        meta_data = meta_data.reset_index(drop=True)

        # bad indices
        bad_indices = meta_data[meta_data['Ordre'].isin(bad_trials)].index
        na_indices = meta_data[meta_data['RT_Correct_CorrPU'].isna()].index
        # Combine bad indices
        remove_indices = bad_indices.union(na_indices)
        # Remove trials (manual selected noisy trials and trials with NaN RT_Correct_CorrPU)
        epochs.drop(remove_indices)

        # 4. remove noisy components
        path_ic = os.path.join(data_dir, f'S{subject_id}_ica_infomax.fif')
        ica = mne.preprocessing.read_ica(path_ic)
        ica.exclude = noisy_components
        ica.apply(epochs, exclude=noisy_components)

        # 5. autoreject
        import autoreject  # version 0.3.1 https://autoreject.github.io/
        ar = autoreject.AutoReject(consensus=np.linspace(0, .4, 11), n_jobs=n_jobs, picks='eeg')
        ar.fit(epochs)  
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
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
                'autoreject_interp_percentage': percentage_interp, }

        # save the epochs
        path_epochs = os.path.join(data_dir, 'Dnl', f'S{subject_id}_epochs_preHMP.fif')
        path_logs = os.path.join(data_dir, 'Dnl', f'S{subject_id}_logs_preHMP.pkl')

        # resample epochs to 256 Hz
        if resample:
            epochs_ar = epochs_ar.resample(256)

        epochs_ar.save(path_epochs, overwrite=True)
        with open(path_logs, 'wb') as f:
            pickle.dump(logs, f)
        
        return epochs_ar, logs