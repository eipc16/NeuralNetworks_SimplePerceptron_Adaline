from neuron import SimplePerceptron, ActivationFunction, Adaline
import matplotlib.pyplot as plt
import csv
import numpy as np

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

def plot(xx, yy, y_err, xlabel, ylabel, title=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.errorbar(xx, yy, yerr=y_err, fmt='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.canvas.set_window_title(title)
    ax.grid(True)


def test_weights_simple(runs=10):

    csv_columns = [
        'Weights [LOW]', 'Weights [HIGH]', 'RUN',
        'Epochs'
    ]

    weights_tests = np.array([
        [-1, 1],
        [-0.8, 0.8],
        [-0.6, 0.6],
        [-0.4, 0.4],
        [-0.2, 0.2],
        [-0.05, 0.05],
        [-0.0, 0.0]
    ])

    X, y = training_X, np.reshape(labels_and, (4, 1))
    xx, yy, y_err = [], [], []
    with open('./results/results_simple_weights.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        for test in weights_tests:
            epochs = []
            for i in range(1, runs + 1):
                perceptron = SimplePerceptron(test)
                epoch, _, p = perceptron.train(X, y)

                if np.all(p == y):
                    writer.writerow([test[0], test[1], i, epoch])
                    epochs.append(epoch)
                else:
                    print('Wrong prediction')
            
            if epochs:
                xx.append(str(f'({test[0]}, {test[1]})'))
                yy.append(np.mean(epochs))
                y_err.append([np.mean(epochs) - np.min(epochs), np.max(epochs) - np.mean(epochs)])
        
        y_err = np.transpose(y_err)
        print(y_err)
        plot(xx, yy, y_err, 'Zakres wag', 'Liczba epok', 'Wpływ początkowego zakresu wag na szybkość uczenia się')
        plt.show()

def test_weights_adaline(runs=100):
    pass

test_weights_simple()
#OR
# print('[OR]')
# perceptron = Perceptron(2)
# print(training_X.shape, labels_or.shape)
# perceptron.train(training_X, np.reshape(labels_or, (4, 1)))
# for val in training_X:
#     print(f'{val} => {perceptron.predict(val)}')

# #AND
# print('[AND]')
# perceptron = Perceptron(2)
# perceptron.train(training_X, np.reshape(labels_and, (4, 1)))
# for val in training_X:
#     print(f'{val} => {perceptron.predict(val)}')
                                                        
# print(perceptron.predict([0.99, 0]))
# print(perceptron.predict([0.0, 0.4]))
# print(perceptron.predict([0.99, 0.6]))