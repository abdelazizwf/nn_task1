# main.py

from preprocessing import preprocess_data
from plots import generate_plots
import numpy as np
from model import Model

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
    data = read_data('penguins.csv')
    preprocess_data(data)
    #generate_plots(data)

    a = slice(1, 51)
    b = slice(51, 101)
    c = slice(101, 151)

    selected_data = extract_features(data, 'bill_depth_mm', 'flipper_length_mm')

    model = Model(selected_data[a], selected_data[c], 0.4, 150, True)
    model.train()
    print(model.test())
