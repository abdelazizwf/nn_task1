# main.py

from itertools import combinations
import logging
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model


PLOT_PATH = Path('./plots/')


# Draw and save the 10 feature plots
def pre_plots(data):
    features = list(data.columns)
    features.remove('species')
    for f1, f2 in combinations(features, 2):
        plt.xlabel(f1)
        plt.ylabel(f2)

        for name, group in data.groupby('species'):
            plt.scatter(group[f1], group[f2], label=name)

        plt.legend()

        path = PLOT_PATH / f"{f1}-{f2}"
        plt.savefig(path, dpi=150)
        plt.clf()


# Black magick
def partition_data(data):
    training, testing, = [], []

    d = data.groupby('species')
    for _, group in d:
        training += list(group.values)[:30]
        testing += list(group.values)[30:]

    shuffle(training)
    shuffle(testing)

    return [d[:2] for d in training], [d[2] for d in training], [d[:2] for d in testing], [d[2] for d in testing]


def preprocess(data):
    data.fillna('pad', inplace=True) # Replace Na values with the last valid value of their column
    data['gender'] = np.where(data['gender'] == 'male', 1, 0) # Convert `male` to 1, `female` to 0
    data['flipper_length_mm'] = data['flipper_length_mm'] / 10 # Convert `flipper_length_mm` to Cm
    data['body_mass_g'] = data['body_mass_g'] / 1000 # Convert `body_mass_g` to Kg

def run(data):
    features = ['bill_depth_mm', 'bill_length_mm']
    species = ['Gentoo', 'Chinstrap']

    filt = data['species'].isin(species)
    sel_data = data.loc[filt, features + ['species']]

    x_train, y_train, x_test, y_test = partition_data(sel_data)
    logging.info(f"Extracted features: {features} for species {species}")

    model = Model(x_train, y_train, x_test, y_test, species)
    logging.info('Initialized the model')

    model.train()
    logging.info('Trained the model')

    acc = model.test()
    logging.info(f'Model accuracy = {acc}')

    print(acc)

    logging.info('Finished.')


if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    logging.info('Starting...')

    plt.style.use('ggplot')

    data = pd.read_csv('penguins.csv')
    preprocess(data)

    logging.info('Preprocessing Done.')

    run(data)
