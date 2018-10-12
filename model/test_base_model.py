import torch
import unittest

import yaml
from torch import nn
from torch.utils.data import DataLoader

from model.M3EP import M3EP
from model.base_model import val_acc
from model.dataset import EnergyLabelData
from model.dict_pad_collate import dict_pad_collate

with open('../config.yml') as f:
    config = yaml.load(f)


class TestBaseModel(unittest.TestCase):
    def test_val_acc(self):
        model = M3EP(config)

        dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
        val_loader = DataLoader(dataset,
                                batch_size=100,
                                num_workers=config['hp']['data_loader']['num_workers'],
                                collate_fn=dict_pad_collate)
        va = val_acc(model, val_loader, torch.device('cpu'))
        print('Accuracy:', va)
        self.assertEqual(type(va).__name__, 'float64')