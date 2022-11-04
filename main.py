# main.py

from itertools import combinations
import logging
import os
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

    if not os.path.exists(PRE_PLOTS_PATH):
        os.makedirs(PRE_PLOTS_PATH)

    for f1, f2 in combinations(features, 2):
        plt.xlabel(f1)
        plt.ylabel(f2)

        for name, group in data.groupby('species'):
            plt.scatter(group[f1], group[f2], label=name)

        plt.legend()

        path = PRE_PLOTS_PATH / f"{f1}-{f2}.png"
        plt.savefig(path, dpi=150)
        plt.clf()


def partition_data(data, y_label):
    # Create empty DataFrames
    training = pd.DataFrame()
    testing = pd.DataFrame()

    # Group by the label, then add 30 training rows to the training DataFrame
    # and 20 test rows to the testing frame
    for _, group in data.groupby(y_label):
        training = pd.concat([training, group.iloc[:29]], ignore_index=True)
        testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

    # Randomly shuffle the data
    training = training.sample(frac=1)
    testing = testing.sample(frac=1)

    # Return the feature columns and label column separately for each DataFrame, such that the return values
    # are training_feature_columns, training_label_column, testing_feature_columns, testing_label_coloumn
    # Inspired by scikit-learn train_test_split function
    return (
        training.loc[:, training.columns != y_label],
        training[y_label],
        testing.loc[:, testing.columns != y_label],
        testing[y_label]
    )


def preprocess(data):
    # Replace Na values with the last valid value of their column
    data.fillna('pad', inplace=True)
    # Convert `male` to 1, `female` to 0
    data['gender'] = np.where(data['gender'] == 'male', 1, 0)
    # Convert `flipper_length_mm` to Cm 
    data['flipper_length_mm'] = data['flipper_length_mm'] / 10
    # Convert `body_mass_g` to Kg
    data['body_mass_g'] = data['body_mass_g'] / 1000


def post_plots(data):
    features = list(data.columns.values)
    species = list(data['species'].unique())

    features.remove('species')

    if not os.path.exists(POST_PLOTS_PATH):
        os.makedirs(POST_PLOTS_PATH)

    acc_sum = 0

    # Combine each combination of features with every combination of species
    for f1, f2 in combinations(features, 2):
        for s in combinations(species, 2):
            plt.xlabel(f1)
            plt.ylabel(f2)

            # Run the model and retreive its data
            model = run(data, [f1, f2], s)
            x_test, y_test = model.x_test, model.y_test
            x0 = model.x0
            w0, w1, w2 = model.weights
            accuracy = model.accuracy

            acc_sum += accuracy
            
            # Calculate the y values of the line and plot it
            line_eq = lambda x: (-(w1 / w2) * x) - ((x0 * w0) / w2)
            line_ys = list(map(line_eq, x_test[f1].values))
            plt.plot(x_test[f1], line_ys)

            # Add the label column to the feature columns
            testing_data = x_test.assign(species=y_test)

            # Plot the features of each species 
            for name, group in testing_data.groupby('species'):
                plt.scatter(group[f1], group[f2], label=name)

            plt.suptitle(f"Accuracy: {accuracy}")

            plt.legend()

            # Save the plot to disk
            path = POST_PLOTS_PATH / f"{f1}-{2} for {s[0]}-{s[1]}.png"
            plt.savefig(path, dpi=150)
            plt.clf()

    print(f"Average Accuracy: {acc_sum / 30}")


def run(data, features, species):
    # Filter the data by species and extract the features
    filt = data['species'].isin(species)
    sel_data = data.loc[filt, list(features) + ['species']]

    # Partition the data
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
