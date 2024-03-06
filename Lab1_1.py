import numpy as np

# Функция инициализации весов нулями
def initialize_weights():
    return np.zeros(5)

# Функция вычисления выхода нейрона
def neuron_output(weights, inputs):
    net = np.dot(weights, inputs)
    return 1 if net >= 0 else 0

def train_neural_network(weights, training_data, eta):
    epoch = 0
    while True:
        total_error = 0
        epoch += 1
        print(f"Эпоха {epoch}")
        print("Вектор весов W:", weights)

        for inputs, expected_output in training_data:
            inputs = np.insert(inputs, 0, 1)  # Добавляем вход смещения
            actual_output = neuron_output(weights, inputs)
            error = expected_output - actual_output
            total_error += abs(error)
            weights += eta * error * inputs
            
            print("Выходной сигнал Y:", actual_output)

        print("Суммарная ошибка E:", total_error)
        print("-" * 30)

        # Если суммарная ошибка равна 0, обучение завершено
        if total_error == 0:
            break

# Ввод обучающих данных
training_data = [
    (np.array([0, 0, 0, 0]),0),
    (np.array([0, 0, 0, 1]),0),
    (np.array([0, 0, 1, 0]),0),
    (np.array([0, 0, 1, 1]),1),
    (np.array([0, 1, 0, 0]),0),
    (np.array([0, 1, 0, 1]),0),
    (np.array([0, 1, 1, 0]),0),
    (np.array([0, 1, 1, 1]),1),
    (np.array([1, 0, 0, 0]),0),
    (np.array([1, 0, 0, 1]),0),
    (np.array([1, 0, 1, 0]),0),
    (np.array([1, 0, 1, 1]),1),
    (np.array([1, 1, 0, 0]),0),
    (np.array([1, 1, 0, 1]),0),
    (np.array([1, 1, 1, 0]),0),
    (np.array([1, 1, 1, 1]),0)
]

# Проверка наличия 16 обучающих наборов
if len(training_data) != 16:
    raise ValueError("Необходимо предоставить ровно 16 обучающих наборов.")

# Инициализация весов и обучение
weights = initialize_weights()
eta = 0.3  # Скорость обучения
train_neural_network(weights, training_data, eta)
