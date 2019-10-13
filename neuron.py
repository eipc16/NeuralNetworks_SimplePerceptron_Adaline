import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    BIPOLAR = 'bipolar'
    UNIPOLAR = 'unipolar'

np.random.seed(1)

class Neuron:
    def __init__(self, activation, weight_bounds, theta):
        self._weight_bounds = weight_bounds
        self._weights = np.array([])
        self._activation_type = activation
        self._theta = theta

    def _feed(self, input_vector):
        return np.round(input_vector) @ self._weights

    def _init(self, training_data):
        self._weights = np.random.uniform(self._weight_bounds[0], self._weight_bounds[1],
                                         size=(training_data.shape[1], 1))

    def _error(self, label, prediction):
        return label - prediction

    def predict(self, inputs):
        pass

    def _activation(self, value):
        pass

    def transform_labels(self, labels):
        if self._activation_type == ActivationFunction.BIPOLAR:
            return 2 * labels - 1
        
        return labels

class SimplePerceptron(Neuron):
    def __init__(self, weight_bounds=[-1, 1], 
                activation=ActivationFunction.UNIPOLAR, 
                theta=0.5):
        super().__init__(activation, weight_bounds, theta)

    def train(self, training_data, labels, 
              iterations=1000, learning_rate=0.05):
        super()._init(training_data)
        for iteration in range(1, iterations + 1):
            y = self.predict(training_data)
            error = self._error(labels, y)
            loss = np.mean(error ** 2)
            
            previous_weights = np.copy(self._weights)
            self._weights += learning_rate * (training_data.T @ error)

            if not self._weights_change(previous_weights):
                break

        return iteration, loss, y

    def _weights_change(self, previous_weights):
        return np.any(previous_weights != self._weights)

    def _activation(self, value):
        if self._activation_type == ActivationFunction.BIPOLAR:
            return np.where(value > self._theta, 1, -1)
        elif self._activation_type == ActivationFunction.UNIPOLAR:
            return np.where(value > self._theta, 1, 0)
    
        return value
    
    def predict(self, inputs):
        prediction = self._feed(inputs)
        return self._activation(prediction)
        
    @staticmethod
    def name():
        return 'Prosty perceptron'

class Adaline(Neuron):
    def __init__(self, weight_bounds=[-1, 1], theta=0.5):
        super().__init__(ActivationFunction.BIPOLAR, 
                        weight_bounds, theta)
        self._bias = np.zeros((1, 1))

    def train(self, training_data, labels, 
              iterations=1000, error_threshold=0.08,
              learning_rate=0.05):
        super()._init(training_data)
        for iteration in range(1, iterations + 1):
            y = self._feed(training_data)
            error = self._error(labels, y)
            loss = np.mean(error ** 2)

            self._weights += learning_rate * 2 * (training_data.T @ error)
            self._bias += learning_rate * np.sum(error, axis=0)

            if loss <= error_threshold:
                break

        return iteration, loss, self._quantization(y)

    def _quantization(self, value):
        return np.where(value > self._theta, 1, -1)
    
    def predict(self, inputs):
        prediction = self._feed(inputs)
        return self._quantization(prediction)
    
    def _feed(self, input_vector):
        return super()._feed(input_vector) + self._bias

    @staticmethod
    def name():
        return 'Adaline'





training_X = np.array([
    [1, 1], 
    [1, 0], 
    [0, 1], 
    [0, 0]
])

labels_and = np.reshape(np.array([
    1, 
    0, 
    0, 
    0,
]), (4,1))

labels_or = np.reshape(np.array([
    1,
    1,
    1,
    0
]), (4,1))
