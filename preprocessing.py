# preprocessing.py

import random

def preprocess_data(data):
    for i in range(1, 151):

        # Convert `male` to 1, `female` = 0 and set a random gender to `NA`
        if data[i][-2] == 'NA':
            data[i][-2] = random.choice([0, 1])
        elif data[i][-2] == 'male':
            data[i][-2] = 1
        else:
            data[i][-2] = 0

        # Convert `body_mass` from gramms to Kg
        data[i][-1] = float(data[i][-1]) / 1000

        # Convert `flipper_length` from mm to cm
        data[i][-3] = float(data[i][-3]) / 10

        # Convert fields from strings to numbers
        data[i][-4] = float(data[i][-4])
        data[i][-5] = float(data[i][-5])

