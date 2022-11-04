# main.py

from itertools import combinations
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model


PRE_PLOTS_PATH = Path('./pre_plots/')
POST_PLOTS_PATH = Path('./post_plots/')


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

        path = PRE_PLOTS_PATH / f"{f1}-{f2}.png"
        plt.savefig(path, dpi=150)
        plt.clf()


# Black magick
def partition_data(data, y_label):
    training = pd.DataFrame()
    testing = pd.DataFrame()

    for _, group in data.groupby(y_label):
        training = pd.concat([training, group.iloc[:29]], ignore_index=True)
        testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

    training = training.sample(frac=1)
    testing = testing.sample(frac=1)

    return training.loc[:, training.columns != y_label], training[y_label], testing.loc[:, testing.columns != y_label], testing[y_label]



def preprocess(data):
    data.fillna('pad', inplace=True) # Replace Na values with the last valid value of their column
    data['gender'] = np.where(data['gender'] == 'male', 1, 0) # Convert `male` to 1, `female` to 0
    data['flipper_length_mm'] = data['flipper_length_mm'] / 10 # Convert `flipper_length_mm` to Cm
    data['body_mass_g'] = data['body_mass_g'] / 1000 # Convert `body_mass_g` to Kg


def post_plots(data):
    features = list(data.columns.values)
    species = list(data['species'].unique())

    features.remove('species')

    for f in combinations(features, 2):
        for s in combinations(species, 2):
            plt.xlabel(f[0])
            plt.ylabel(f[1])

            model = run(data, f, s)
            x_test, y_test = model.x_test, model.y_test
            x0 = model.x0
            w0, w1, w2 = model.weights

            xs = []
            ys = []
            for x1, _ in x_test.values:
                y = (-(w1 / w2) * x1) - ((x0 * w0) / w2)
                xs.append(x1)
                ys.append(y)
            plt.plot(xs, ys)

            testing_data = x_test.assign(species=y_test)

            for name, group in testing_data.groupby('species'):
                plt.scatter(group[f[0]], group[f[1]], label=name)

            plt.legend()

            path = POST_PLOTS_PATH / f"{f[0]}-{f[1]} for {s[0]}-{s[1]} at {str(model.accuracy)}.png"
            plt.savefig(path, dpi=150)
            plt.clf()


def run(data, features, species):
    #features = ('bill_depth_mm', 'bill_length_mm')
    #species = ('Gentoo', 'Chinstrap')

    filt = data['species'].isin(species)
    sel_data = data.loc[filt, features + ('species',)]

    x_train, y_train, x_test, y_test = partition_data(sel_data, 'species')
    logging.info(f"Extracted features: {features} for species {species}")

    model = Model(x_train, y_train, x_test, y_test, species)
    logging.info('Initialized the model')

    model.train()
    logging.info('Trained the model')

    acc = model.test()
    logging.info(f'Model accuracy = {acc}')

    print(acc)

    return model


if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    logging.info('Starting...')

    plt.style.use('ggplot')

    data = pd.read_csv('penguins.csv')
    preprocess(data)

    logging.info('Preprocessing Done.')

    pre_plots(data)
    post_plots(data)

    logging.info('Finished.')
