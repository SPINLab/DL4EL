from copy import deepcopy

from torch.utils.data import Dataset
import numpy as np
import deep_geometry.geom_scaler as gs
from tqdm import tqdm


class EnergyLabelData(Dataset):
    def __init__(self, numpy_zip_path, normalization=None):
        """
        Loads data from numpy_zip_path, applies a quick integrity check and sets normalization settings
        :param numpy_zip_path: path as string to a Energy Data numpy zip file
        :param normalization: A dictionary with normalization settings use as


        Use:
            train_dataset = EnergyLabelData('path/training/npz', normalization=None)  # 'None is not necessary, tho'
            val_dataset   = EnergyLabelData('path/validation/npz', normalization=other_dataset.normalization)
        """
        npz = np.load(numpy_zip_path)
        self.data = []
        self.labels = [label['energy_performance_vec'] for label in npz['labels']]

        print('Loading data from', numpy_zip_path)
        for record in tqdm(npz['data']):
            # final sanity check
            allowed_classes = ['ndarray', 'list', 'int']
            inputs = {}

            for (key, val) in record.items():
                if key.endswith('_vec'):
                    if type(val).__name__ not in allowed_classes:
                        raise ValueError('Unknown data type ' + type(val).__name__ + ' in ' + str(record))
                    inputs[key] = val

            self.data.append(inputs)

        if normalization:  # re-use normalization settings from a different data loader
            self.normalization = normalization
            print('Re-used normalization settings, stored in .normalization dictionary')

        else:  # create new normalization settings
            self.normalization = {}

            print('Getting normalization parameters...')

            # scale geometry
            geoms = [sample['geometry_vec'] for sample in self.data]
            self.normalization['geom_scale'] = gs.fit(geoms)

            # recorded dates
            recorded_dates = [sample['recorded_date_vec'] for sample in self.data]
            recorded_dates = np.array(recorded_dates)
            self.normalization['rec_year_mean'] = np.mean(recorded_dates[:, 0])
            self.normalization['rec_year_std'] = np.std(recorded_dates[:, 0])
            self.normalization['rec_month_mean'] = np.mean(recorded_dates[:, 1])
            self.normalization['rec_month_std'] = np.std(recorded_dates[:, 1])
            self.normalization['rec_day_mean'] = np.mean(recorded_dates[:, 2])
            self.normalization['rec_day_std'] = np.std(recorded_dates[:, 2])
            self.normalization['rec_weekday_mean'] = np.mean(recorded_dates[:, 3])
            self.normalization['rec_weekday_std'] = np.std(recorded_dates[:, 3])

            # registration dates
            registration_dates = [sample['registration_date_vec'] for sample in self.data]
            registration_dates = np.array(registration_dates)
            self.normalization['reg_year_mean'] = np.mean(registration_dates[:, 0])
            self.normalization['reg_year_std'] = np.std(registration_dates[:, 0])
            self.normalization['reg_month_mean'] = np.mean(registration_dates[:, 1])
            self.normalization['reg_month_std'] = np.std(registration_dates[:, 1])
            self.normalization['reg_day_mean'] = np.mean(registration_dates[:, 2])
            self.normalization['reg_day_std'] = np.std(registration_dates[:, 2])
            self.normalization['reg_weekday_mean'] = np.mean(registration_dates[:, 3])
            self.normalization['reg_weekday_std'] = np.std(registration_dates[:, 3])

            # house numbers
            house_numbers = [sample['house_number_vec'] for sample in self.data]
            self.normalization['house_number_mean'] = np.mean(house_numbers)
            self.normalization['house_number_std'] = np.std(house_numbers)

            # year of construction
            construction_years = [sample['year_of_construction_vec'] for sample in self.data]
            self.normalization['construction_years_mean'] = np.mean(construction_years)
            self.normalization['construction_years_std'] = np.std(construction_years)

            print('Normalization settings stored in .normalization dictionary')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = deepcopy(self.data[index])
        sample['geometry_vec'][:, :2] /= self.normalization['geom_scale']
        sample['recorded_date_vec'][0] -= self.normalization['rec_year_mean']
        sample['recorded_date_vec'][0] /= self.normalization['rec_year_std']
        sample['recorded_date_vec'][1] -= self.normalization['rec_month_mean']
        sample['recorded_date_vec'][1] /= self.normalization['rec_month_std']
        sample['recorded_date_vec'][2] -= self.normalization['rec_day_mean']
        sample['recorded_date_vec'][2] /= self.normalization['rec_day_std']
        sample['recorded_date_vec'][3] -= self.normalization['rec_weekday_mean']
        sample['recorded_date_vec'][3] /= self.normalization['rec_weekday_std']
        sample['registration_date_vec'][0] -= self.normalization['reg_year_mean']
        sample['registration_date_vec'][0] /= self.normalization['reg_year_std']
        sample['registration_date_vec'][1] -= self.normalization['reg_month_mean']
        sample['registration_date_vec'][1] /= self.normalization['reg_month_std']
        sample['registration_date_vec'][2] -= self.normalization['reg_day_mean']
        sample['registration_date_vec'][2] /= self.normalization['reg_day_std']
        sample['registration_date_vec'][3] -= self.normalization['reg_weekday_mean']
        sample['registration_date_vec'][3] /= self.normalization['reg_weekday_std']
        sample['year_of_construction_vec'] -= self.normalization['construction_years_mean']
        sample['year_of_construction_vec'] /= self.normalization['construction_years_std']
        sample['house_number_vec'] -= self.normalization['house_number_mean']
        sample['house_number_vec'] /= self.normalization['house_number_std']

        label = deepcopy(self.labels[index])
        label_one_hot = np.zeros((9,))
        label_one_hot[label] = 1

        return sample, label
