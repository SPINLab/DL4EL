import os
import socket

import numpy as np
import torch
import yaml
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from time import time
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.M3EP import M3EP, batch_to_device
from model.dataset import EnergyLabelData
from model.dict_pad_collate import dict_pad_collate

SCRIPT_VERSION = '0.0.1'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.').replace(' ', '_')
SCRIPT_START = time()

def val_acc(model, val_loader, device):
    val_batch = next(iter(val_loader))
    val_batch = batch_to_device(val_batch, device)
    val_pred = model(val_batch)
    y_pred = val_pred.cpu().detach().numpy()
    y_pred = [np.argmax(p) for p in y_pred]
    score = accuracy_score(val_batch[1], y_pred)
    return score


if __name__ == '__main__':
    with open('../config.yml') as f:
        config = yaml.load(f)

    # Tensorboard log writer
    log_dir = os.path.join('runs', TIMESTAMP)
    print('Writing tensorboard logs every', config['log_frequency'], 'steps to', log_dir)
    writer = SummaryWriter(log_dir)

    model = M3EP(config)

    if torch.cuda.is_available():
        print('Using CUDA implementation')
        device = torch.device('cuda')
        model.cuda()
    else:
        print('Using CPU implementation')
        device = torch.device('cpu')

    train_data = EnergyLabelData('../data/building_energy_train_v1.2_part_1.npz')
    train_loader = DataLoader(train_data,
                              batch_size=config['data_loader']['batch_size'],
                              num_workers=config['data_loader']['num_workers'],
                              collate_fn=dict_pad_collate)
    val_data = EnergyLabelData('../data/building_energy_val_v1.2.npz', normalization=train_data.normalization)
    val_loader = DataLoader(val_data,
                            batch_size=config['data_loader']['validation_size'],
                            num_workers=config['data_loader']['num_workers'],
                            collate_fn=dict_pad_collate)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=float(config['learning_rate']))

    for epoch in range(config['epochs']):
        print('Epoch {} of {}:'.format(str(epoch + 1), config['epochs']))
        loss_msg = {}

        with tqdm(total=len(train_loader)) as progress_bar:
            for step, batch in enumerate(train_loader):
                batch = batch_to_device(batch, device)
                labels = batch[1]

                prediction = model(batch)
                loss = loss_fn(prediction, labels)
                loss_msg['loss'] = loss.item()
                progress_bar.postfix = loss_msg
                progress_bar.update()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % config['log_frequency'] == 0:
                    writer.add_scalar('train_loss', loss.item(), global_step=(epoch * len(train_loader) + step))

        acc = val_acc(model, val_loader, device)
        print('Validation accuracy:', acc.item())
        writer.add_scalar('val_acc', acc.item(), global_step=epoch)

    accuracy = val_acc(model, val_loader, device)
    runtime = time() - SCRIPT_START
    message = 'on {} completed with accuracy of \n{:f} \nin {} in {} epochs\n'.format(
        socket.gethostname(), accuracy, timedelta(seconds=runtime), config['epochs'])

    # for key, value in sorted(config['hp']):
    #     message += '{}: {}\t'.format(key, value)

    # notify(SIGNATURE, message)
    print(SCRIPT_NAME, 'finished successfully with', message)
