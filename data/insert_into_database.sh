#!/usr/bin/env bash
set -ex
chmod a+rw ./*.csv

psql --host localhost --username postgres --dbname geodata --file create_building_energy_performance_table.sql
psql --host localhost --username postgres --dbname geodata --command "\COPY energy.building_energy_performance FROM '$(pwd)/woningen_07.2017-1.csv' DELIMITER ',' CSV HEADER;"
psql --host localhost --username postgres --dbname geodata --command "\COPY energy.building_energy_performance FROM '$(pwd)/woningen_07.2017-2.csv' DELIMITER ',' CSV HEADER;"
psql --host localhost --username postgres --dbname geodata --command "\COPY energy.building_energy_performance FROM '$(pwd)/woningen_07.2017-3.csv' DELIMITER ',' CSV HEADER;"
psql --host localhost --username postgres --dbname geodata --command "\COPY energy.building_energy_performance FROM '$(pwd)/woningen_07.2017-4.csv' DELIMITER ',' CSV HEADER;"

psql --host localhost --username postgres --dbname geodata --file create_combined_purpose_table.sql
psql --host localhost --username postgres --dbname geodata --file create_combined_view.sql