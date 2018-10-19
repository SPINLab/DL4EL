import torch
import unittest

import yaml

from model.M3EP import M3EP, batch_to_device
from model.base_model import val_acc
from model.dataset import EnergyLabelData

from torch.utils.data import DataLoader

from model.dict_pad_collate import dict_pad_collate

with open('../config.yml') as f:
    config = yaml.load(f)

dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz', config)
data_loader = DataLoader(dataset)


class MyTestCase(unittest.TestCase):
    global dataset, data_loader

    def test_variable_length_geometries(self):
        submodule_name = 'geometry'

        if submodule_name in config['submodules']:
            model = M3EP(config)
            batch_size = 12
            number_of_points = range(1, 100)
            channels = 5

            for n in number_of_points:
                batch = torch.rand(batch_size, channels, n)
                output = model.geometry(batch)
                fusion_input_size = config['late_fusion']['input_size']
                size = torch.Size([12, config['submodules']['geometry']['output_size'], fusion_input_size])
                self.assertEqual(output.shape, size)
        else:
            pass

    def test_model_cpu(self):
        model = M3EP(config)
        for step, batch in enumerate(data_loader):
            batch = batch_to_device(batch, torch.device('cpu'))
            pred = model(batch)
            output_size = config['late_fusion']['output_size']
            self.assertEqual(pred.shape[1], output_size)
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

    def test_val_acc(self):
        model = M3EP(config)

        val_loader = DataLoader(dataset,
                                batch_size=100,
                                num_workers=config['data_loader']['num_workers'],
                                collate_fn=dict_pad_collate)
        va = val_acc(model, val_loader, torch.device('cpu'))
        print('Accuracy:', va)
        self.assertEqual(type(va).__name__, 'float64')

    def test_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        for step, batch in enumerate(data_loader):
            model = M3EP(config)
            pred = model(batch_to_device(batch, torch.device('cpu')))
            targets = batch[1]
            loss = loss_fn(pred, targets)
            self.assertEqual(type(loss).__name__, 'Tensor')


if __name__ == '__main__':
    unittest.main()
