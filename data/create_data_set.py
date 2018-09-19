import argparse
import gc
import os
import sys
from datetime import datetime, timedelta
from time import time

import psycopg2
import numpy as np
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt

from deep_geometry.vectorizer import vectorize_wkt
from tqdm import tqdm

SCRIPT_VERSION = '1.2'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
TRAIN_DATA_FILE = 'building_energy_train_v{}'.format(SCRIPT_VERSION)
VAL_DATA_FILE = 'building_energy_val_v{}'.format(SCRIPT_VERSION)
TEST_DATA_FILE = 'building_energy_test_v{}'.format(SCRIPT_VERSION)
UNIT_TEST_DATA_FILE = 'building_energy_unit_test_v{}'.format(SCRIPT_VERSION)
SCRIPT_START = time()

parser = argparse.ArgumentParser(description='Create energy label prediction data set')
parser.add_argument('-H', '--host', type=str, help='database host', required=True)
parser.add_argument('-P', '--port', type=str, help='database port')
parser.add_argument('-d', '--database', type=str, help='database name', required=True)
parser.add_argument('-u', '--user', type=str, help='database user', required=True)
parser.add_argument('-p', '--password', type=str, help='database password')
parser.add_argument('-o', '--output_folder', type=str, help='output folder', required=True)
args = parser.parse_args()

purpose_to_english = {
    "woonfunctie": "residential",
    "bijeenkomstfunctie": "gathering",
    "celfunctie": "cell",
    "gezondheidszorgfunctie": "health",
    "industriefunctie": "industry",
    "kantoorfunctie": "office",
    "logiesfunctie": "lodging",
    "onderwijsfunctie": "education",
    "sportfunctie": "sports",
    "winkelfunctie": "shopping",
    "overige gebruiksfunctie": "other",
}

VOCABULARY = '0123456789abcdefghijklmnopqrstuvwxyz'
PURPOSES = [
    "residential",
    "gathering",
    "cell",
    "health",
    "industry",
    "office",
    "lodging",
    "education",
    "sports",
    "shopping",
    "other",
]
ENERGY_CLASSES = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']


def get_data_from_db(cursor):
    """
    Get data from the database given a query-instantiated cursor
    :param cursor: query-instantiated database cursor
    :return: tuple of labels and training data
    """
    training_data, labels = [], []
    cols = [desc[0] for desc in cursor.description]

    for row in tqdm(cursor, total=cursor.rowcount):
        purposes = [purpose_to_english[p] for p in row[cols.index('purposes')]]
        record = {
            'postal_code': row[cols.index('postal_code')],
            'house_number': int(row[cols.index('house_number')]),
            'house_number_addition': row[cols.index('house_number_addition')],
            'purposes': purposes,
            'year_of_construction': int(row[cols.index('year_of_construction')]),
            'recorded_date': row[cols.index('recorded_date')],
            'registration_date': row[cols.index('registration_date')],
            'geometry_crs84': row[cols.index('geometry_wgs84')],
            'centroid_crs84': row[cols.index('centroid_wgs84')],
        }

        # just duplicate for house_number and year of construction
        record['house_number_vec'] = record['house_number']
        record['year_of_construction_vec'] = record['year_of_construction']

        # one-hot encoding for house number addition
        if record['house_number_addition']:
            hna = np.zeros(shape=(len(record['house_number_addition']), len(VOCABULARY)))
            for idx, char in enumerate(record['house_number_addition']):
                hna[idx, VOCABULARY.index(char.lower())] = 1.
        else:
            hna = np.zeros(shape=(1, len(VOCABULARY)))
        record['house_number_addition_vec'] = hna

        # 'multi-hot' encoding for building purposes
        purposes = np.zeros(shape=(len(PURPOSES,)))
        for purpose in record['purposes']:
            purposes[PURPOSES.index(purpose)] = 1.
        record['purposes_vec'] = purposes

        pc = np.zeros(shape=(len(record['postal_code']), len(VOCABULARY)))
        for idx, char in enumerate(record['postal_code']):
            pc[idx, VOCABULARY.index(char.lower())] = 1.
        record['postal_code_vec'] = pc

        geom = record['geometry_crs84']
        geom = vectorize_wkt(geom)
        # center
        centroid = np.mean(geom[:, :2], axis=0)
        geom[:, :2] = geom[:, :2] - centroid
        record['geometry_vec'] = geom
        record['centroid_vec'] = vectorize_wkt(record['centroid_crs84'])[0]

        rd = record['recorded_date']
        record['recorded_date_vec'] = [rd.year, rd.month, rd.day, rd.weekday()]

        rgd = record['registration_date']
        record['registration_date_vec'] = [rgd.year, rgd.month, rgd.day, rgd.weekday()]

        training_data.append(record)
        labels.append({
            'energy_performance_index': row[cols.index('energy_performance_index')],
            'energy_performance_label': row[cols.index('energy_performance_label')],
            'energy_performance_vec': ENERGY_CLASSES.index(row[cols.index('energy_performance_label')])
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

    print('Constructing database query result, this will take a few minutes...')
    data_query = open('epl_all_data.sql', mode='r', encoding='utf-8').read()
    cursor.execute(data_query)
    runtime = time() - SCRIPT_START
    print('Query executed in', timedelta(seconds=runtime))
    data, labels = get_data_from_db(cursor)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
    # clean up and free some memory
    connection.close()
    del data, labels, cursor, connection
    gc.collect()

    val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)

    print('Saving unit test data')
    np.savez_compressed(os.path.join(args.output_folder, UNIT_TEST_DATA_FILE),
                        data=train_data[:100],
                        labels=train_labels[:100])

    print('Saving test data...')
    np.savez_compressed(os.path.join(args.output_folder, TEST_DATA_FILE),
                        data=test_data,
                        labels=test_labels)

    print('Saving validation data...')
    np.savez_compressed(os.path.join(args.output_folder, VAL_DATA_FILE),
                        data=val_data,
                        labels=val_labels)

    print('Saving training data...')

    parts = 10
    for part in range(parts):
        np.savez_compressed(os.path.join(args.output_folder, TRAIN_DATA_FILE + '_part_' + str(part + 1)),
                            data=train_data[part::parts],
                            labels=train_labels[part::parts])


if __name__ == '__main__':
    main()
    runtime = time() - SCRIPT_START
    print('Done in', timedelta(seconds=runtime))
