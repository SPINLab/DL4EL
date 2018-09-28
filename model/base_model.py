import torch
import yaml
from sklearn.metrics import accuracy_score
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import EnergyLabelData
from model.dict_pad_collate import dict_pad_collate

if torch.cuda.is_available():
    print('Using cuda:0')
    device = torch.device("cuda:0")
else:
    print('Using CPU implementation')
    device = torch.device("cpu")

with open('../config.yml') as f:
    config = yaml.load(f)


def val_acc(model, val_data):
    validation_indices = list(range(len(val_data)))
    val_sample_indices = np.random.choice(
        validation_indices, size=config['hp']['data_loader']['validation_size'])
    val_sample_data = [val_data[idx] for idx in val_sample_indices]
    sample_tensor = torch.Tensor([sample[0]['year_of_construction_vec'] for sample in val_sample_data])
    sample_tensor = sample_tensor.float().to(device)
    val_pred = model(sample_tensor)
    y_pred = val_pred.cpu().detach().numpy()
    y_pred = [np.argmax(p) for p in y_pred]
    val_sample_labels = [sample[1] for sample in val_sample_data]
    score = accuracy_score(val_sample_labels, y_pred)
    return score


if __name__ == '__main__':
    train_data = EnergyLabelData('../data/building_energy_train_v1.2_part_1.npz')
    # train_data = EnergyLabelData('../data/building_energy_unit_test_v1.2.npz')
    train_loader = DataLoader(train_data,
                              batch_size=config['hp']['data_loader']['batch_size'],
                              num_workers=config['hp']['data_loader']['num_workers'],
                              collate_fn=dict_pad_collate)
    val_data = EnergyLabelData('../data/building_energy_val_v1.2.npz')

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
        loss_msg = {}

        with tqdm(total=len(train_loader)) as pbar:
            for idx, batch in enumerate(train_loader):
                inputs = batch[0]['year_of_construction_vec'][0]
                inputs = inputs.unsqueeze(dim=1)
                inputs = inputs.float()
                inputs = inputs.to(device)

                labels = batch[1].to(device)

                pred = model(inputs)
                loss = loss_fn(pred, labels)
                loss_msg['loss'] = loss.item()
                pbar.postfix = loss_msg
                pbar.update()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Validation accuracy:', val_acc(model, val_data))
