import random

class Model:

    def __init__(self, data1, data2, eta, epochs, bias_flag):
        """
        data = [
            ['class', 'f1', 'f2'],
            ['class', 'f1', 'f2'],
            ...
        ]
        """
        self.c1_train, self.c1_test = self.partition(data1)
        self.c2_train, self.c2_test = self.partition(data2)
        self.eta = eta
        self.epochs = epochs
        self.bias_flag = bias_flag

        self.label_map = {data1[0][0]: -1, data2[0][0]: 1}
        self.weights = [random.random(), random.random(), random.random()]

    def partition(self, data):
        return data[:30], data[30:]

    def train(self):
        training_data = self.c1_train + self.c2_train
        random.shuffle(training_data)

        x0 = 1 if self.bias_flag else 0

        for _ in range(self.epochs):
            self._train(training_data, x0)

    def _train(self, data, x0):
        for label, x1, x2 in data:
            x1 = float(x1)
            x2 = float(x2)
            t = self.label_map[label]

            net = (x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

            if net < 0:
                y = -1
            else:
                y = 1

            e = t - y

            self.weights[0] += self.eta * e * x0
            self.weights[1] += self.eta * e * x1
            self.weights[2] += self.eta * e * x2

    def test(self):
        test_data = self.c1_test + self.c2_test
        random.shuffle(test_data)
        
        x0 = 1 if self.bias_flag else 0
        correct = 0

        for label, x1, x2 in test_data:
            t = self.label_map[label]

            net = (x0 * self.weights[0]) + (float(x1) * self.weights[1]) + (float(x2) * self.weights[2])

            if net < 0:
                y = -1
            else:
                y = 1

            if t == y:
                correct += 1

        return (correct / len(test_data)) * 100
