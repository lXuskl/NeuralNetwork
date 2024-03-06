import numpy as np

# Функция инициализации весов нулями
def initialize_weights():
    return np.zeros(3)  # 2 веса для входов + 1 вес смещения

# Функция вычисления net
def calculate_net(weights, inputs):
    return np.dot(weights, inputs)

# Функция вычисления выхода нейрона
def neuron_output(net):
    return 1 if net >= 0 else 0

def train_neural_network(weights, training_data, eta):
    epoch = 0
    while True:
        total_error = 0
        epoch += 1
        print(f"Эпоха {epoch}")
        print("Начальный вектор весов W:", weights)

        for inputs, expected_output in training_data:
            inputs = np.insert(inputs, 0, 1)  # Добавляем вход смещения
            net = calculate_net(weights, inputs)
            actual_output = neuron_output(net)
            error = expected_output - actual_output
            total_error += abs(error)
            weights += eta * error * inputs

            print(f"Обучающий набор {inputs[1:]}, Net: {net}, Выходной сигнал Y: {actual_output}, Ошибка: {error}")
        
        print("Конечный вектор весов W:", weights)
        print("Суммарная ошибка E:", total_error)
        print("-" * 30)

        # Если суммарная ошибка равна 0, обучение завершено
        if total_error == 0:
            break

# Ввод обучающих данных
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 1),
]

weights = initialize_weights()
eta = 1  # Скорость обучения
train_neural_network(weights, training_data, eta)