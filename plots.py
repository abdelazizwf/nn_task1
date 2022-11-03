# plots.py

import matplotlib.pyplot as plt
import numpy as np
from util import *

def plot_features(data):
    cmap = {C1: 'r', C2: 'g', C3: 'b'} # Color map
    mmap = {C1: 'o', C2: '^', C3: 's'} # Marker map

    # Get the data slice of the given class and plot it
    for cls in CLASSES:
        sl = get_class_slice(cls)
        sl_data = data[sl]
        sl_data = np.transpose(sl_data)
        plt.scatter(sl_data[1], sl_data[2], c=cmap[cls], marker=mmap[cls], label=cls)

    plt.xlabel(data[0][1])
    plt.ylabel(data[0][2])

    plt.legend()
    plt.show()
