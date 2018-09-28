#!/usr/bin/env bash
export LC_ALL=C.UTF-8 # for amazon

sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-compose

# python 3.6 for ubuntu 16.04 and earlier
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.6 python3-pip

sudo -H pip install --upgrade pip
sudo -H pip install pipenv

# PostGreSQL client only
sudo apt-get install -y postgresql-client
