import numpy as np
from enum import Enum
from typing import List, Tuple, Union, Optional
from itertools import combinations
import pandas as pd

class Electrodes(Enum):
    Fp1 = 1
    Fp2 = 2
    F7 = 3
    F3 = 4
    Fz = 5
    F4 = 6
    F8 = 7
    T3 = 8
    T4 = 9
    T5 = 10
    T6 = 11
    O1 = 12
    O2 = 13
    C3 = 14
    Cz = 15
    C4 = 16
    P3 = 17
    Pz = 18
    P4 = 19

class Bands(Enum):
    delta = 1
    theta = 2
    alpha1 = 3
    alpha2 = 4
    beta1 = 5
    beta2 = 6
    gamma = 7

    @staticmethod
    def get_name_by_id(id):
        return Bands(id).name

    @staticmethod
    def get_values():
        return [el.value for el in Bands]
    

class PairsElectrodes1020:
    def __init__(self, electrodes: Electrodes):
        self.electrodes = electrodes
        self.nearest = [('Fp1', 'Fp2'),
                        ('Fp1', 'Fz'),
                        ('Fp2', 'Fz'),
                        ('Fp1', 'F3'),
                        ('Fp1', 'F7'),
                        ('Fp2', 'F4'),
                        ('Fp2', 'F8'),
                        ('F7', 'T3'),
                        ('F7', 'C3'),
                        ('F7', 'F3'),
                        ('F3', 'C3'),
                        ('F3', 'Cz'),
                        ('F3', 'Fz'),
                        ('Fz', 'C3'),
                        ('Fz', 'C4'),
                        ('Fz', 'Cz'),
                        ('Fz', 'F4'),
                        ('F4', 'Cz'),
                        ('F4', 'C4'),
                        ('F4', 'T4'),
                        ('F4', 'F8'),
                        ('F8', 'T4'),
                        ('F8', 'C4'),
                        ('T3', 'T5'),
                        ('T3', 'C3'),
                        ('T3', 'P3'),
                        ('C3', 'P3'),
                        ('C3', 'Cz'),
                        ('C3', 'Pz'),
                        ('C3', 'T5'),
                        ]

    @property
    def electrode_pairs(self):
        els = list(map(lambda x: x.name, self.electrodes))
        return list(combinations(els, 2))

    def create_pairs_dict(self, pairs_list, filter_by=None):
        pairs_dict = dict()
        p_list = pairs_list.copy()
        els = list(map(lambda x: x.name, self.electrodes))
        if filter_by:
            for opt in filter_by:
                p_list = [pair for pair in p_list if opt in pair]
        for i, el1 in enumerate(els):
            el1_p_list = [pair for pair in p_list if el1 in pair]
            for el2 in els[i + 1:]:
                pairs_dict[(el1, el2)] = [pair for pair in el1_p_list if el2 in pair]
        return pairs_dict


class EEGData:
    '''
    Class function for reading EEG data via constructors
        data : [np.ndarray] EEG data of shape (n_subjects, n_channels, num_freqs)
        subj_list : List[str] List of subject identifiers corresponding to the first axis of 'data'
        electrodes : Enum, Enum represents electrodes
        el_pairs_list : List[Tuple[str, str]] List of electrode pairs
        bands : Enum or List[str] List of EEG frequency bands
    '''

    def __init__(self, data, subj_list, electrodes, el_pairs_list, bands):
        self.data = data
        self.subj_list = subj_list
        self.electrodes = electrodes
        self.el_pairs_list = el_pairs_list
        self.bands = bands


def read_from_eeg_dataframe(path_to_df,
                            cond_prefix='fo',
                            band_list=None):
    
    ''' 
    Function to read EEG data from (.csv) format & load into (n_subjects, n_channels, num_freqs) format

    Parameters: 
        path_to_df: str, path to csv file
        cond_prefix: Condition string for 
        band_list : List[int], optional, frequency bands

    Returns: 

    '''
    if band_list is None:
        band_list = [1, 2, 3, 4, 5, 6, 7]        
        bands = Bands
    df = pd.read_csv(path_to_df, index_col=0)
    subj_list = list(df.index)
    pairs = PairsElectrodes1020(Electrodes)
    pairs_list = list(df.columns)
    data = []
    for b in band_list:
        pairs_dict = pairs.create_pairs_dict(pairs_list, filter_by=[cond_prefix, f'_{b}_'])
        columns = [col[0] for col in list(pairs_dict.values())]
        data.append(df[columns].values)
    data = np.array(data).swapaxes(0, 1).swapaxes(1, 2)
    return EEGData(data, subj_list, Electrodes, (pairs_dict.keys()), bands)


def reshape_eeg_data(data: np.ndarray,
                     reshape_bands: bool = True
                    ) -> np.ndarray:
    """
    Reshape EEG data from (n_subjects, chan_pairs, num_freqs) or (chan_pairs, chan_pairs) to
    (n_subjects, n_chans, n_chans, n_freq) or to (n_subjects, n_chans*n_freq, n_chans*n_freq,) if reshape_bands is True,
    where each chansxchans block corresponds to a specific frequency. The number of electrode pairs is considered as 19, 
    for this instance of implementation.

    Parameters:
        data: [np.ndarray] of shape (n_subjects,  chan_pairs, num_freqs)
        reshape_bands (bool): default True, returns a block diagonal matrix where each block is w.r.t individual frequency bands

    Returns:
        reshaped_data: [np.ndarray] of shape (n_subjects,  chan_pairs, num_freqs) or  (n_subjects, n_chans*n_freq, n_chans*n_freq,)
        For single subjects, reshaped_data: [np.ndarray] of shape (chan_pairs, chan_pairs, num_freqs) or (chan_pairs*num_freqs, chan_pairs*num_freqs)        
    """

    num_els = len(Electrodes) # Default 19 electrodes 
    el_pairs_list = PairsElectrodes1020(Electrodes).electrode_pairs 

    # Input with single subject: 
    dtype = data.ndim == 2
    if dtype == True:
        data = data[np.newaxis,...]
        
    n_subjects, _, n_frequencies = data.shape
    reshaped_data = np.zeros((n_subjects, num_els, num_els, n_frequencies))

    # Fill in the 19x19 matrices for each frequency
    for pair_idx, (el1, el2) in enumerate(el_pairs_list):
        i, j = Electrodes[el1].value - 1, Electrodes[el2].value - 1
        reshaped_data[:, i, j, :] = data[:, pair_idx, :]
        reshaped_data[:, j, i, :] = data[:, pair_idx, :]  

    if reshape_bands:
        #Create block-diagonal form
        to_reshape = reshaped_data.copy()
        reshaped_data = np.zeros((n_subjects, num_els * n_frequencies, num_els * n_frequencies))
        for k in range(n_frequencies):
            reshaped_data[:, k * num_els:(k + 1) * num_els, k * num_els:(k + 1) * num_els] = to_reshape[..., k] 

    return  reshaped_data[0] if dtype else reshaped_data



def inverse_reshape_eeg_data(
            reshaped_data: np.ndarray,
            reshape_bands: bool = True
    ) -> np.ndarray:
        """
        Inversely reshape EEG data back to (n_subjects, chan_pairs, num_freqs).

        Parameters:
            reshaped_data: [np.ndarray] EEG data of shape (n_subjects, num_els, num_els, num_freqs)
            or (n_subjects, num_els*n_freqs, num_els*n_freqs) if reshape_bands=True.

            reshape_bands (bool): Whether input is in band-flattened form.

        Returns:
            np.ndarray: Original EEG data of shape (n_subjects, chan_pairs, num_freqs) or (chan_pairs, num_freq)
        """

        num_els = len(Electrodes) # Default 19 electrodes 
        el_pairs_list = PairsElectrodes1020(Electrodes).electrode_pairs 

        # Input with single subject
        dtype = reshaped_data.ndim == 2
        if dtype == True:
            reshaped_data = reshaped_data[np.newaxis,...]

        n_subjects = reshaped_data.shape[0]

        if reshape_bands:
            # Extract frequency-specific blocks
            n_frequencies = reshaped_data.shape[1] // num_els
            extracted_data = np.zeros((n_subjects, num_els, num_els, n_frequencies))
            for k in range(n_frequencies):
                extracted_data[..., k] = reshaped_data[:, k * num_els:(k + 1) * num_els, k * num_els:(k + 1) * num_els]

            reshaped_data = extracted_data  # Convert back to (n_subjects, num_els, num_els, num_freqs)

        # Reconstruct the (n_subjects, chan_pairs, num_freqs) array
        n_frequencies = reshaped_data.shape[-1]
        original_data = np.zeros((n_subjects, len(el_pairs_list), n_frequencies))

        for pair_idx, (el1, el2) in enumerate(el_pairs_list):
            i, j = Electrodes[el1].value - 1, Electrodes[el2].value - 1
            original_data[:, pair_idx, :] = reshaped_data[:, i, j, :]

             
        return original_data[0] if dtype else original_data
