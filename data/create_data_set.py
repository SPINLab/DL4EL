import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from time import time

import psycopg2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

SCRIPT_VERSION = '1'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
TRAIN_DATA_FILE = 'building_energy_train_v{}'.format(SCRIPT_VERSION)
VAL_DATA_FILE = 'building_energy_val_v{}'.format(SCRIPT_VERSION)
TEST_DATA_FILE = 'building_energy_test_v{}'.format(SCRIPT_VERSION)
SCRIPT_START = time()

parser = argparse.ArgumentParser(description='Create energy label prediction data set')
parser.add_argument('-H', '--host', type=str, help='database host', required=True)
parser.add_argument('-P', '--port', type=str, help='database port')
parser.add_argument('-d', '--database', type=str, help='database name', required=True)
parser.add_argument('-u', '--user', type=str, help='database user', required=True)
parser.add_argument('-p', '--password', type=str, help='database password')
parser.add_argument('-o', '--output_folder', type=str, help='output folder', required=True)
args = parser.parse_args()


def get_data_from_db(cursor):
    """
    Get data from the database given a query-instantiated cursor
    :param cursor: query-instantiated database cursor
    :return: tuple of labels and training data
    """
    training_data, labels = [], []
    cols = [desc[0] for desc in cursor.description]

    for row in cursor:
        sample = {
            'postal_code': row[cols.index('postal_code')],
            'house_number': int(row[cols.index('house_number')]),
            'house_number_addition': row[cols.index('house_number_addition')],
            'purposes': row[cols.index('purposes')],
            'year_of_construction': int(row[cols.index('year_of_construction')]),
            'recorded_date': row[cols.index('recorded_date')],
            'registration_date': row[cols.index('registration_date')],
            'geometry_wgs84': row[cols.index('geometry_wgs84')],
            'centroid_wgs84': row[cols.index('centroid_wgs84')],
        }

        training_data.append(sample)
        labels.append({
            'energy_performance_index': row[cols.index('energy_performance_index')],
            'energy_performance_label': row[cols.index('energy_performance_label')],
        })

    return training_data, labels


def main():
    try:
        connection = psycopg2.connect(
            host=args.host, database=args.database,
            user=args.user, password=args.password if 'password' in args else None,
            connect_timeout=3)
    except psycopg2.OperationalError as e:
        print('Unable to connect:', e)
        sys.exit(1)

    connection.set_client_encoding('utf-8')
    cursor = connection.cursor()

    print('Constructing database query, this will take a few minutes...')
    data_query = open('epl_all_data.sql', mode='r', encoding='utf-8').read()
    cursor.execute(data_query)
    runtime = time() - SCRIPT_START
    print('Query executed in', timedelta(seconds=runtime))
    data, labels = get_data_from_db(cursor)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)
    connection.close()

    print('Saving training data...')
    np.savez_compressed(os.path.join(args.output_folder, TRAIN_DATA_FILE),
                        data=train_data,
                        labels=train_labels)

    print('Saving validation data...')
    np.savez_compressed(os.path.join(args.output_folder, VAL_DATA_FILE),
                        data=val_data,
                        labels=val_labels)

    print('Saving test data...')
    np.savez_compressed(os.path.join(args.output_folder, TEST_DATA_FILE),
                        data=test_data,
                        labels=test_labels)


if __name__ == '__main__':
    main()
    runtime = time() - SCRIPT_START
    print('Done in', timedelta(seconds=runtime))
