# main.py

from preprocessing import preprocess_data
import numpy as np
from model import Model
import logging

def read_data(path):
    data = []

    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    return data

def extract_features(data, f1, f2):
    new_data = []
    data_t = np.transpose(data)

    for column in data_t:
        if column[0] in [f1, f2, 'species']:
            new_data.append(column)

    new_data = np.transpose(new_data)
    return list(new_data)

if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    logging.info('Starting...')
    data = read_data('penguins.csv')
    preprocess_data(data)
    logging.info('Preprocessing Done.')
    #generate_plots(data)

    a = slice(1, 51)
    b = slice(51, 101)
    c = slice(101, 151)

    selected_data = extract_features(data, 'bill_depth_mm', 'flipper_length_mm')
    logging.info(f"Extracted features: {selected_data[0][1]} and {selected_data[0][2]}")

    model = Model(selected_data[a], selected_data[b])
    logging.info('Initialized the model')
    model.train()
    logging.info('Trained the model')
    acc = model.test()
    logging.info(f'Model accuracy = {acc}')
    print(acc)
    logging.info('Finished.')
