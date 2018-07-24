import torch

import yaml

with open('../config.yml') as f:
    config = yaml.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    net.to(device)
else:
    device = torch.device("cpu")

for epoch in config.hp.epochs:
    inputs, labels = train_loader.next()

    if torch.cuda.is_available():
        inputs, labels = inputs.to(device), labels.to(device)

