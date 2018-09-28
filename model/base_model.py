import torch
import yaml
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import EnergyLabelData
from model.dict_pad_collate import dict_pad_collate

with open('../config.yml') as f:
    config = yaml.load(f)

if torch.cuda.is_available():
    print('Using cuda:0')
    device = torch.device("cuda:0")
else:
    print('Using CPU implementation')
    device = torch.device("cpu")

# train_data = EnergyLabelData('../data/building_energy_train_v1.2_part_1.npz')
train_data = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
train_loader = DataLoader(train_data,
                          batch_size=config['hp']['data_loader']['batch_size'],
                          num_workers=config['hp']['data_loader']['num_workers'],
                          collate_fn=dict_pad_collate)
# val_loader = EnergyLabelData('../data/building_energy_val_v1.2.npz')

hidden_size = 100
# output_shape = train_loader[0][1].shape
output_shape = 9
ref_sample_input = train_data[0][0]
input_shape = np.shape(ref_sample_input['year_of_construction_vec'])

model = nn.Sequential(
    nn.Linear(1, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_shape))
model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    params=model.parameters(),
    lr=float(config['hp']['learning_rate']))

for epoch in range(config['hp']['epochs']):
    print('Epoch {} of {}:'.format(str(epoch + 1), config['hp']['epochs']))

    for batch in tqdm(train_loader):
        inputs = batch[0]['year_of_construction_vec'][0]
        inputs = inputs.unsqueeze(dim=0)
        inputs = inputs.to(device)
        labels = batch[1].to(device)

        pred = model(inputs.float())
        loss = loss_fn(pred, labels)
        # print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
