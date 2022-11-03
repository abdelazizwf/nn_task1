import logging
import random

class Model:

    def __init__(self, data1, data2, eta=0.4, epochs=750, bias_flag=1):
        """
        data = [
            ['class', 'f1', 'f2'],
            ['class', 'f1', 'f2'],
            ...
        ]
        """
        # Partitioning data
        self.c1_train, self.c1_test = self.partition(data1)
        self.c2_train, self.c2_test = self.partition(data2)
        self.eta = eta # Learning rate
        self.epochs = epochs
        self.bias_flag = bias_flag

        self.label_map = {data1[0][0]: -1, data2[0][0]: 1} # Representing classes as -1 and 1
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] # Initializing weights

        logging.debug(f'Model initialized with: eta={eta}, epochs={epochs}, bias_flag={bias_flag}')
        logging.debug(f"Label Map: {self.label_map}")
        logging.debug(f'Starting weights: {self.weights}')

    def partition(self, data):
        return data[:30], data[30:]

    def train(self):
        # Combine training data and shuffle it
        training_data = self.c1_train + self.c2_train
        random.shuffle(training_data)
        logging.debug(f"Training Data: {training_data}")

        x0 = 1 if self.bias_flag else 0 # Set the bias input according to the bias flag

        for i in range(self.epochs):
            self._train(training_data, x0)
            logging.debug(f'Weights after epoch {i + 1}: {self.weights}')

    def _train(self, data, x0):
        for label, x1, x2 in data:
            t = self.label_map[label] # Set the target to -1 or 1 according to the class

            # Calculate the net value (W^T * X)
            net = (x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

            # Activation function
            if net < 0:
                y = -1
            else:
                y = 1

            # Calculate error
            e = t - y

            # Update the weights
            self.weights[0] += self.eta * e * x0
            self.weights[1] += self.eta * e * x1
            self.weights[2] += self.eta * e * x2

    def test(self):
        # Combine the test data and shuffle it
        test_data = self.c1_test + self.c2_test
        random.shuffle(test_data)
        logging.debug(f"Testing Data: {test_data}")
        
        x0 = 1 if self.bias_flag else 0 # Set the bias input according to the bias flag
        correct = 0 # A counter for the correct prediction made by the model

        for label, x1, x2 in test_data:
            t = self.label_map[label] # Set the target to -1 or 1 according to the class

            # Calculate the net value (W^T * X)
            net = (x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

            # Activation function
            if net < 0:
                y = -1
            else:
                y = 1

            # Calculate error
            if t == y:
                correct += 1

        # Return the accuracy as a percentage
        return (correct / len(test_data)) * 100
