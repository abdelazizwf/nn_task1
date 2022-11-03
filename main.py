# main.py

from preprocessing import preprocess_data
from model import Model
import logging
from plots import plot_features
from itertools import combinations
from util import *

def read_data(path):
    data = []

    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    return data

def pre_plots(data):
    for f1, f2 in combinations(data[0][1:], 2):
        selected_data = extract_features(data, f1, f2)
        plot_features(selected_data)

if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    logging.info('Starting...')
    data = read_data('penguins.csv')
    preprocess_data(data)
    logging.info('Preprocessing Done.')

    feature1 = 'bill_depth_mm'
    feature2 = 'flipper_length_mm'

    a = get_class_slice(C1)
    b = get_class_slice(C2)
    c = get_class_slice(C3)

    selected_data = extract_features(data, feature1, feature2)
    logging.info(f"Extracted features: {feature1} and {feature2}")

    pre_plots(data)

    model = Model(selected_data[a], selected_data[b])
    logging.info('Initialized the model')
    model.train()
    logging.info('Trained the model')
    acc = model.test()
    logging.info(f'Model accuracy = {acc}')
    print(acc)
    logging.info('Finished.')
