from unittest import TestCase
import numpy as np
import tfnbs.eeg_utils as eeg_utils
from tfnbs.eeg_utils import Electrodes, PairsElectrodes1020, Bands
import pickle

class TesteegUtils(TestCase):

    def setUp(self) -> None:
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        pairs = PairsElectrodes1020(Electrodes)
        self.path_to_df = 'tests/eeg_dataframe_nansfilled.csv'

    def test_read_from_eeg_dataframe(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')
        self.assertTrue(stable_fo.data.shape == (177, 171, 7))
        self.assertTrue(stable_fz.data.shape == (177, 171, 7))
        self.assertEqual(len(stable_fo.subj_list), len(stable_fz.subj_list))

    def test_reshape_eeg_data(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')
        reshaped_data = eeg_utils.reshape_eeg_data(stable_fo.data - stable_fz.data, reshape_bands=False)
        self.assertTrue(reshaped_data.shape, (177, 19, 19, 7))
        reshaped_data = eeg_utils.reshape_eeg_data(stable_fo.data - stable_fz.data, reshape_bands=True)
        self.assertTrue(reshaped_data.shape, (177, 19 * 7, 19 * 7))

    
    def test_inverse_reshape_eeg_data(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')

        orginal_data = stable_fo.data - stable_fz.data
        reshaped_data = eeg_utils.reshape_eeg_data(orginal_data, reshape_bands=True)
        inversed_data = eeg_utils.inverse_reshape_eeg_data(reshaped_data, reshape_bands=True)
        np.testing.assert_almost_equal(orginal_data, inversed_data)

