import numpy as np

class Perceptron():
    def __init__(self, num_of_inputs, max_weights=0.2, threshold=1000, learning_rate=0.01, bias=True):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.bias = bias

        if self.bias:
            num_of_inputs += 1

        self.weights = np.random.rand(num_of_inputs) * max_weights * 2 - max_weights
        
    def train(self, training_data, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                scaled_error = self.learning_rate * self._error(prediction, label)

                if self.bias:
                    self.weights[1:] -= scaled_error * inputs
                    self.weights[0] -= scaled_error
                else:
                    self.weights -= scaled_error * inputs
            
        print(f'{self.weights}')

    def predict(self, inputs):
        if isinstance(inputs, list):
            inputs = self._fix_inputs(inputs)

        if self.bias:
            sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        else:
            sum = np.dot(inputs, self.weights)

        return self._activation(sum)

    def _activation(self, value):
        return 1 if value > 0 else 0

    def _error(self, label, prediction):
        return label - prediction

    def _fix_inputs(self, inputs):
        return np.array(inputs)

training_data = np.array([
    [1, 1], 
    [1, 0], 
    [0, 1], 
    [0, 0]
])

labels_and = np.array([
    1, 
    0, 
    0, 
    0,
])

labels_or = np.array([
    1,
    1,
    1,
    0
])




#OR
print('[OR]')
perceptron = Perceptron(2)
perceptron.train(training_data, labels_or)
for val in training_data:
    print(f'{val} => {perceptron.predict(val)}')

#AND
print('[AND]')
perceptron = Perceptron(2)
perceptron.train(training_data, labels_and)
for val in training_data:
    print(f'{val} => {perceptron.predict(val)}')