import torch
import unittest

import yaml

from model.M3EP import M3EP, batch_to_device
from model.dataset import EnergyLabelData

from torch.utils.data import DataLoader

from model.dict_pad_collate import dict_pad_collate

with open('../config.yml') as f:
    config = yaml.load(f)

dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
data_loader = DataLoader(dataset)


class MyTestCase(unittest.TestCase):
    global dataset, data_loader

    def test_variable_length_geometries(self):
        model = M3EP(config)

        batch_size = 12
        number_of_points = range(1, 100)
        for n in number_of_points:
            batch = torch.rand(batch_size, n, 5)
            geom_model = model.geometry_submodule(batch.shape[1], torch.device('cpu'))
            output = geom_model(batch)
            self.assertEqual(output.shape, torch.Size([12, config['hp']['submodules']['geometry_cnn']['output_size'], 1]))

    def test_model_cpu(self):
        model = M3EP(config)
        for step, batch in enumerate(data_loader):
            batch = batch_to_device(batch, torch.device('cpu'))
            pred = model(batch)
            output_size = config['hp']['output_size']
            self.assertEqual(pred.shape, torch.Size([1, output_size]))
            self.assertEqual(pred.device, torch.device('cpu'))

    def test_correct_device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = M3EP(config)
        model = model.to(device)

        for step, batch in enumerate(data_loader):
            batch = batch_to_device(batch, device)
            pred = model(batch)
            self.assertEqual(pred.device.type, device.type)

    def test_correct_device_cuda_call(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = M3EP(config)
        if torch.cuda.is_available():
            model.cuda()

        for step, batch in enumerate(data_loader):
            batch = batch_to_device(batch, device)
            pred = model(batch)
            # output_size = config['hp']['output_size']
            self.assertEqual(pred.device.type, device.type)

    def test_parallel_data_loader_batch(self):
        data_loader = DataLoader(dataset,
                                 batch_size=50,
                                 collate_fn=dict_pad_collate)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for step, batch in enumerate(data_loader):
            model = M3EP(config)
            if torch.cuda.is_available():
                model.cuda()
            batch = batch_to_device(batch, device)
            pred = model(batch)
            self.assertEqual(pred.device.type, device.type)


if __name__ == '__main__':
    unittest.main()