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

def plot_bar(xx, yy, xlabel, ylabel, title=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.bar(xx, yy,)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.canvas.set_window_title(title)

def test_weights(runs=10, neuron=SimplePerceptron):
    name = neuron.name()

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
    
    with open(f'./results/results_weights_{name}_{runs}.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        for test in weights_tests:
            epochs = []
            for i in range(1, runs + 1):
                perceptron = neuron(test)
                epoch, _, p = perceptron.train(X, y)

                if np.all(p == perceptron.transform_labels(y)):
                    writer.writerow([test[0], test[1], i, epoch])
                    epochs.append(epoch)
                else:
                    print('Wrong prediction')
            
            if epochs:
                xx.append(str(f'({test[0]}, {test[1]})'))
                yy.append(np.mean(epochs))
                y_err.append([np.mean(epochs) - np.min(epochs), np.max(epochs) - np.mean(epochs)])
        
        y_err = np.transpose(y_err)
        plot(xx, yy, y_err, 'Zakres wag', 'Liczba epok', 'Wpływ początkowego zakresu wag na szybkość uczenia się')
        plt.savefig(f'./results/test_weights_{name}_{runs}.png')

def test_learning_rate(runs=100, neuron=SimplePerceptron):
    name = neuron.name()

    csv_columns = [
        'RUN', 'Learning Rate', 'Predicted?'
    ]

    learning_rates = np.array([
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
        0.8,
        1.0
    ])

    X, y = training_X, np.reshape(labels_and, (4, 1))
    xx, yy, yy2, y_err = [], [], [], []
    
    with open(f'./results/results_LR_{name}_{runs}.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        for test in learning_rates:
            epochs, correct_predictions = [], []
            for i in range(1, runs + 1):
                perceptron = neuron([-0.2, 0.2])
                epoch, _, p = perceptron.train(X, y, learning_rate=test)

                if np.all(p == perceptron.transform_labels(y)): 
                    epochs.append(epoch)
                    correct_predictions.append(np.all(p == perceptron.transform_labels(y)))
                else:
                    epochs.append(0)
                    correct_predictions.append(False)

                writer.writerow([i, epoch, test, test, np.all(p == perceptron.transform_labels(y))])
 
            if epochs:
                xx.append(str(f'{test}'))
                yy.append(np.mean(epochs))
                yy2.append(np.sum(correct_predictions) / (runs))
                y_err.append([np.mean(epochs) - np.min(epochs), np.max(epochs) - np.mean(epochs)])
        
        y_err = np.transpose(y_err)
        plot(xx, yy, y_err, 'Współczynnik uczenia', 'Liczba epok', 'Wpływ współczynnika uczenia na szybkość uczenia się')
        plt.savefig(f'./results/test_LR_{name}_{runs}_epochs.png')
        plot_bar(xx, yy2, 'Współczynnik uczenia', 'Poprawne predykcje [%]', 'Wpływ współczynnika uczenia na dokładność predykcji')
        plt.savefig(f'./results/test_LR_{name}_{runs}_correctness.png')

def test_activation_function(runs=10):
    neuron = SimplePerceptron
    name = neuron.name()

    csv_columns = [
        'Activation function', 'RUN', 'Epochs'
    ]

    X, y = training_X, np.reshape(labels_and, (4, 1))
    xx, yy, y_err = [], [], []

    activation_fn = np.array([
        (2 * y - 1, ActivationFunction.BIPOLAR),
        (y, ActivationFunction.UNIPOLAR)
    ])
    
    with open(f'./results/results_activation_{name}_{runs}.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        for labels, test in activation_fn:
            epochs = []
            for i in range(1, runs + 1):
                perceptron = neuron([-.2, .2], activation=test)
                epoch, _, p = perceptron.train(X, labels)

                if np.all(p == labels):
                    writer.writerow([test.value, i, epoch])
                    epochs.append(epoch)
                else:
                    print('Wrong prediction')
            
            if epochs:
                val = test.value
                xx.append(str(f'{val}'))
                yy.append(np.mean(epochs))
                y_err.append([np.mean(epochs) - np.min(epochs), np.max(epochs) - np.mean(epochs)])
        
        y_err = np.transpose(y_err)
        plot_bar(xx, yy, 'Funkcja aktywacji', 'Liczba epok', 'Wpływ funkcji aktywacji na szybkość uczenia się')
        plt.savefig(f'./results/test_activation_{name}_{runs}.png')

def compare_algorithms(runs=100):
    csv_columns = [
        'Algorithm', 'RUN', 'Epochs'
    ]

    X, y = training_X, np.reshape(labels_and, (4, 1))
    xx, yy, y_err = [], [], []

    algorithms = np.array([
        (Adaline, 2 * y - 1, None),
        (SimplePerceptron, 2 * y - 1, ActivationFunction.BIPOLAR),
        (SimplePerceptron, y, ActivationFunction.UNIPOLAR)
    ])
    
    with open(f'./results/results_algorithms_{runs}.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        for neuron, labels, activation in algorithms:
            epochs = []
            for i in range(1, runs + 1):
                if activation is None:
                    perceptron = neuron([-.2, .2])
                    epoch, _, p = perceptron.train(X, y, learning_rate=0.03)
                else:
                    perceptron = neuron([-.2, .2], activation=activation)
                    epoch, _, p = perceptron.train(X, labels, learning_rate=0.03)
                

                if np.all(p == labels):
                    writer.writerow([perceptron.name(), i, epoch])
                    epochs.append(epoch)
                else:
                    print('Wrong prediction')
            
            if epochs:
                val = perceptron.name()
                xx.append(str(f'{val}'))
                yy.append(np.mean(epochs))
                y_err.append([np.mean(epochs) - np.min(epochs), np.max(epochs) - np.mean(epochs)])
        
        y_err = np.transpose(y_err)
        plot_bar(xx, yy, 'Algorytm', 'Liczba epok', 'Wpływ wybranego algorytmu na szybkość uczenia się')
        plt.savefig(f'./results/test_algorithms_{runs}.png')

# test_weights(runs=10, neuron=SimplePerceptron)
# test_weights(runs=100, neuron=SimplePerceptron)
# test_weights(runs=10, neuron=Adaline)
# test_weights(runs=100, neuron=Adaline)

# test_learning_rate(runs=100, neuron=SimplePerceptron)
# test_learning_rate(runs=100, neuron=Adaline)

# test_activation_function(runs=100)

compare_algorithms(runs=100)