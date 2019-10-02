import numpy as np

class Perceptron():
    def __init__(self, num_of_inputs, max_weights=2, threshold=1000, learning_rate=0.05):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_of_inputs + 1) * max_weights - (max_weights / 2)
        
    def train(self, training_data, labels):
        for i in range(self.threshold):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                scaled_error = self.learning_rate * self._error(prediction, label)

                self.weights[1:] += scaled_error * inputs
                self.weights[0] += scaled_error

        print(f'Epoch {i} finished. Weights: {self.weights}')

    def predict(self, inputs):
        if isinstance(inputs, list):
            inputs = self._fix_inputs(inputs)

        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]

        return 1 if sum > 0 else 0

    def _error(self, label, prediction):
        return prediction - label

    def _fix_inputs(self, inputs):
        return np.array(inputs)

training_data = np.array([
    [1, 1], 
    [1, 0], 
    [0, 1], 
    [0, 0]
])

labels = np.array([
    1, 
    0, 
    0, 
    0,
])

perceptron = Perceptron(2)
perceptron.train(training_data, labels)

inputs = np.array([1, 1])
result = perceptron.predict(inputs) 
print(result)

inputs = np.array([0, 1])
result = perceptron.predict(inputs) 
print(result)

inputs = np.array([0, 0])
result = perceptron.predict(inputs) 
print(result)

inputs = np.array([1, 0])
result = perceptron.predict(inputs) 
print(result)
