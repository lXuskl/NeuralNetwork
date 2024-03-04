import numpy as np
import io
import sys

# Перенаправим стандартный вывод в переменную
old_stdout = sys.stdout
sys.stdout = text_trap = io.StringIO()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_nodes, bias):
        self.weights = np.zeros(input_nodes)
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total)

    def train(self, inputs, outputs, learning_rate):
        iteration = 0
        while True:
            total_error = 0
            binary_total_error = 0
            binary_predictions = []
            for input, output in zip(inputs, outputs):
                prediction = self.feedforward(input)
                error = output - prediction
                total_error += error**2
                binary_prediction = 1 if prediction >= 0.5 else 0
                binary_predictions.append(binary_prediction)
                binary_error = output - binary_prediction
                binary_total_error += binary_error**2
                adjustments = learning_rate * error * sigmoid_derivative(prediction)
                self.weights += input * adjustments
                self.bias += adjustments

            mean_squared_error = total_error / len(inputs)
            binary_mean_squared_error = binary_total_error / len(inputs)
            print(f"Эпоха {iteration+1}:")
            print(f"Вектор весов W: {self.weights}")
            print(f"Бинаризованный выходной сигнал Y: {binary_predictions}")
            print(f"Суммарная ошибка E (непрерывные значения): {mean_squared_error}")
            print(f"Суммарная ошибка E (бинарные значения): {binary_mean_squared_error}\n")

            if binary_mean_squared_error == 0:
                break
            iteration += 1

# Обучающие данные
inputs = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 1, 1],
                   [0, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0],
                   [1, 0, 1, 1],
                   [1, 1, 0, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]])
outputs = np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [1], [0], [0], [0], [1]])


bias = 1  
neural_network = NeuralNetwork(4, bias)
neural_network.train(inputs, outputs, learning_rate=0.3)
print(f"Полное обучение окончено, запуск частичного обучения\n")

min_training_set = None
min_epochs = None

for i in range(1, len(inputs) + 1):
    for j in range(len(inputs) - i + 1):
        subset_inputs = inputs[j:j+i]

        bias = 1
        neural_network = NeuralNetwork(4, bias)
        epochs = neural_network.train(subset_inputs, outputs, learning_rate=0.3)

        if neural_network.feedforward(inputs).round().astype(int).flatten().tolist() == outputs.flatten().tolist():
            min_training_set = subset_inputs
            min_epochs = epochs
            break
    if min_training_set is not None:
        break

min_training_set, min_epochs

sys.stdout = old_stdout

output_text = text_trap.getvalue()

output_filename = 'G:/Neiro/ining_output.txt'
with open(output_filename, 'w') as file:
    file.write(output_text)

output_filename