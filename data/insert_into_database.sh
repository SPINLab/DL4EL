#!/usr/bin/env bash
psql --host localhost --username postgres --dbname geodata --file create_building_energy_performance_table.sql
psql --host localhost --username postgres --dbname geodata --file insert_building_energy_data.sql
