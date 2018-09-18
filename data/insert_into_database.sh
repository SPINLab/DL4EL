#!/usr/bin/env bash
csvsql --db postgresql:///geodata \
  --create-if-not-exists \
  --db-schema energy \
  --tables building_energy_performance \
  --insert woningen_07.2017-1.csv

csvsql --db postgresql:///geodata \
  --create-if-not-exists \
  --db-schema energy \
  --tables building_energy_performance \
  --insert woningen_07.2017-2.csv

csvsql --db postgresql:///geodata \
  --create-if-not-exists \
  --db-schema energy \
  --tables building_energy_performance \
  --insert woningen_07.2017-3.csv

csvsql --db postgresql:///geodata \
  --create-if-not-exists \
  --db-schema energy \
  --tables building_energy_performance \
  --insert woningen_07.2017-4.csv
