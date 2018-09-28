import unittest

from torch import nn

from model.base_model import val_acc
from model.dataset import EnergyLabelData


class TestBaseModel(unittest.TestCase):
    def test_val_acc(self):
        model = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 9))
        model.cuda()

        dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
        va = val_acc(model, dataset)
        print('Accuracy:', va)
        self.assertEqual(type(va).__name__, 'float64')