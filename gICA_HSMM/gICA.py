import mne
import numpy as np
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))# Add the parent directory to the path
from config import data_dir
from Preprocessing import preprocess
from scipy.stats import zscore

class gICA:
    def __init__(self, ids):
        self.ids = ids
        self.paths = [os.path.join(data_dir, 'prehmp', f'S{id}_epochs_preHMP.fif') for id in ids]

    @classmethod
    def concatenate_stimRt(cls, paths, zscore_norm=True):
        all_data = []
        for path in paths:
            epochs = mne.read_epochs(path, preload=True)
            sfreq = epochs.info['sfreq']
            rt_sample = [int((rt * sfreq) + (-epochs.tmin * sfreq)) for rt in epochs.metadata['RT_Correct_CorrPU'].values]
            data = epochs.get_data()
            all_data.append(np.concatenate([trial[:, :sample] for trial, sample in zip(data, rt_sample)], axis=1))

        concat_all_data = np.concatenate(all_data, axis=1)
        if zscore_norm:
            concat_all_data = zscore(concat_all_data)
        info = mne.io.read_info(paths[0])
        concat_all = mne.io.RawArray(concat_all_data, info)
        return concat_all

    def run(self, zscore_norm=True, n_components=0.99, max_iter=1000, method='fastica', icaName ='gICA.fif'):
        concat_all = self.concatenate_stimRt(self.paths, zscore_norm=zscore_norm)
        ica = mne.preprocessing.ICA(n_components=n_components, max_iter=max_iter, method=method)
        ica.fit(concat_all, picks='eeg', verbose=True)
        ica_path = os.path.join(data_dir, icaName)
        ica.save(ica_path, overwrite=True)
        return ica
