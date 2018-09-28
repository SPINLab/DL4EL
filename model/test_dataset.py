import torch
import unittest

from torch.utils.data import DataLoader
import numpy as np

from model.dataset import EnergyLabelData


class TestDataLoader(unittest.TestCase):
    def test_dataset_loads(self):
        global dataset, data_loader
        dataset = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
        data_loader = DataLoader(dataset)
        self.assertEqual(len(dataset), 100)

    def test_geometry_valid(self):
        for (data, label) in data_loader:
            for (key, val) in data.items():
                self.assertTrue(type(val).__name__ in ['Tensor', 'list'])
            geom_as_numpy = data['geometry_vec'].numpy()

            mean = np.mean(geom_as_numpy[0, :, :2], axis=0)
            self.assertAlmostEqual(mean[0], 0, places=2)
            self.assertAlmostEqual(mean[1], 0, places=2)

            render_one_hot_sum = np.sum(geom_as_numpy[0, :, 2:], axis=1)
            # since a one-hot vector sums to one, each element should sum to 1.
            for s in render_one_hot_sum:
                self.assertEqual(s, 1.)

    def test_dataset_std(self):
        geoms = []
        for (data, label) in data_loader:
            geoms.append(data['geometry_vec'].numpy()[0, :, :2].tolist())
        # flatten to single list
        geoms = sum(geoms, [])
        std = np.std(geoms)
        self.assertAlmostEqual(std, 0.76, 1)

    def test_normalized_recorded_date(self):
        recorded_dates = []
        for (data, label) in data_loader:
            recorded_dates.append(data['recorded_date_vec'])

        recorded_dates = np.array(recorded_dates)
        year_mean = np.mean(recorded_dates[:, 0])
        year_std = np.std(recorded_dates[:, 0])
        month_mean = np.mean(recorded_dates[:, 1])
        month_std = np.std(recorded_dates[:, 1])
        day_mean = np.mean(recorded_dates[:, 2])
        day_std = np.std(recorded_dates[:, 2])
        weekday_mean = np.mean(recorded_dates[:, 3])
        weekday_std = np.std(recorded_dates[:, 3])

        self.assertAlmostEqual(year_mean, 0, places=1)
        self.assertAlmostEqual(year_std, 1, places=1)
        self.assertAlmostEqual(month_mean, 0, places=1)
        self.assertAlmostEqual(month_std, 1, places=1)
        self.assertAlmostEqual(day_mean, 0, places=1)
        self.assertAlmostEqual(day_std, 1, places=1)
        self.assertAlmostEqual(weekday_mean, 0, places=2)
        self.assertAlmostEqual(weekday_std, 1, places=2)

    def test_normalized_registration_date(self):
        registration_dates = []
        for (data, label) in data_loader:
            registration_dates.append(data['registration_date_vec'])

        registration_dates = np.array(registration_dates)
        year_mean = np.mean(registration_dates[:, 0])
        year_std = np.std(registration_dates[:, 0])
        month_mean = np.mean(registration_dates[:, 1])
        month_std = np.std(registration_dates[:, 1])
        day_mean = np.mean(registration_dates[:, 2])
        day_std = np.std(registration_dates[:, 2])
        reg_weekday_mean = np.mean(registration_dates[:, 3])
        reg_weekday_std = np.std(registration_dates[:, 3])

        self.assertAlmostEqual(year_mean, 0, places=2)
        self.assertAlmostEqual(year_std, 1, places=2)
        self.assertAlmostEqual(month_mean, 0, places=2)
        self.assertAlmostEqual(month_std, 1, places=2)
        self.assertAlmostEqual(day_mean, 0, places=2)
        self.assertAlmostEqual(day_std, 1, places=2)
        self.assertAlmostEqual(reg_weekday_mean, 0, places=2)
        self.assertAlmostEqual(reg_weekday_std, 1, places=2)

    def test_normalized_year_of_construction(self):
        construction_years = []
        for (data, label) in data_loader:
            construction_years.append(data['year_of_construction_vec'])

        mean = np.mean(construction_years)
        std = np.std(construction_years)

        self.assertAlmostEqual(mean, 0, places=2)
        self.assertAlmostEqual(std, 1, places=2)

    def test_normalized_house_number(self):
        house_numbers = []
        for (data, label) in data_loader:
            house_numbers.append(data['house_number_vec'])

        mean = np.mean(house_numbers)
        std = np.std(house_numbers)

        self.assertAlmostEqual(mean, 0, places=2)
        self.assertAlmostEqual(std, 1, places=2)

    def test_label_shape(self):
        for (data, label) in data_loader:
            self.assertEqual(label.shape, torch.Size([1, 9]))
