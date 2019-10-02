import numpy as np

np.random.seed(1)

class Perceptron():
    def __init__(self, num_of_inputs, max_weights=1, iterations=1000, learning_rate=0.01, bias=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = np.reshape(np.random.rand(num_of_inputs), (2, 1)) * max_weights * 2 - max_weights
        self.threshold = 0.5
        print(f'Init weights: {self.weights.shape}')
        
    def train(self, training_data, labels):
        for _ in range(self.iterations):
            prediction = self.predict(training_data)
            scaled_error = self._error(labels, prediction)
            self.weights += self.learning_rate * (training_data.T @ scaled_error)

        print(f'{self.weights}')

    def predict(self, inputs):
        if isinstance(inputs, list):
            inputs = self._fix_inputs(inputs)
        
        sum = np.dot(inputs, self.weights)

        return self._activation(sum)

    def _activation(self, value):
        value = value - self.threshold
        return np.where(value > 0, 1, 0)

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
perceptron.train(training_data, np.reshape(labels_or, (4, 1)))
for val in training_data:
    print(f'{val} => {perceptron.predict(val)}')

#AND
print('[AND]')
perceptron = Perceptron(2)
perceptron.train(training_data, np.reshape(labels_and, (4, 1)))
for val in training_data:
    print(f'{val} => {perceptron.predict(val)}')