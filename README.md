# DL4EL
Deep learning for energy performance label prediction

## Installation
If you want to re-create the source data, please install the required software packages. A ubuntu 16.04 script is [available](script/install.sh)

Once you have the basic software requirements (Docker community edition, Python 3.6+, pip3, pipenv, csvkit) you:
- start a PostGreSQL/PostGIS container using [geodata-docker](https://github.com/SPINlab/geodata-docker) with the BAG database loaded
- with a spinning "geodata" container, execute the scripts in the [data](data) directory:
  - [get_data](data/get_data.sh)
  - [insert_into_database](data/insert_into_database.sh) 
  
