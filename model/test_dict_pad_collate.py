import unittest

from torch.utils.data import DataLoader

from model.dataset import EnergyLabelData
from model.dict_pad_collate import dict_pad_collate


class TestDictPadCollate(unittest.TestCase):
    def test_dataset(self):
        dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
        data_loader = DataLoader(dataset,
                                 batch_size=50,
                                 num_workers=0,
                                 collate_fn=dict_pad_collate)

        for batch in data_loader:
            self.assertEqual(len(batch[1]), 50)
