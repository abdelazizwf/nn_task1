import logging
import random

class Model:

    def __init__(self, x_train, y_train, x_test, y_test, labels, eta=0.4, epochs=750, bias_flag=True):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.eta = eta # Learning rate
        self.epochs = epochs
        self.x0 = 1 if bias_flag else 0

        self.label_map = {labels[0]: -1, labels[1]: 1} # Representing classes as -1 and 1
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] # Initializing weights

        logging.debug(f'Model initialized with: eta={eta}, epochs={epochs}, bias_flag={bias_flag}')
        logging.debug(f"Label Map: {self.label_map}")
        logging.debug(f'Starting weights: {self.weights}')

    def train(self):
        for i in range(self.epochs):
            for [x1, x2], label in zip(self.x_train, self.y_train):
                t = self.label_map[label] # Set the target to -1 or 1 according to the class

                # Calculate the net value (W^T * X)
                net = (self.x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

                # Activation function
                if net < 0:
                    y = -1
                else:
                    y = 1

                # Calculate error
                e = t - y

                # Update the weights
                self.weights[0] += self.eta * e * self.x0
                self.weights[1] += self.eta * e * x1
                self.weights[2] += self.eta * e * x2

            logging.debug(f'Weights after epoch {i + 1}: {self.weights}')

    def test(self):
        correct = 0 # A counter for the correct prediction made by the model

        for [x1, x2], label in zip(self.x_test, self.y_test):
            t = self.label_map[label] # Set the target to -1 or 1 according to the class

            # Calculate the net value (W^T * X)
            net = (self.x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

            # Activation function
            if net < 0:
                y = -1
            else:
                y = 1

            # Calculate error
            if t == y:
                correct += 1

        # Return the accuracy as a percentage
        return (correct / len(self.y_test)) * 100
