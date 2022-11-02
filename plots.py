# plots.py

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def plot(x1, x2, x_label, y_label):
    plt.figure('Figure')
    plt.scatter(x1, x2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def generate_plots(data):
    data_t = np.transpose(data)

    for x1, x2 in combinations(data_t[1:], 2):
        plot(x1[1:], x2[1:], x1[0], x2[0])
