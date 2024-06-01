import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def purelin(x):
    return x

def purelin_derivative(x):
    return np.ones_like(x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_size))
        self.weights_input_hidden_before = copy.copy(self.weights_input_hidden)

        self.weights_hidden_output = np.random.uniform(-1, 1, size=(self.hidden_size, self.output_size))
        self.weights_hidden_output_before = copy.copy(self.weights_hidden_output)

    def forward(self, X_train):
        self.hidden_input = np.dot(X_train, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))
        return self.output

    def backward(self, X_train, y_train, output):
        error = y_train - output
        output_delta = error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X_train.T.dot(hidden_delta) * self.learning_rate

    def compute_loss(self, y_train, Y_hat):
        # Binary cross-entropy loss
        m = y_train.shape[0]
        loss = -np.sum(y_train * np.log(Y_hat) + (1 - y_train) * np.log(1 - Y_hat)) / m
        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs, early_stop_threshold):
        train_errors = []
        val_errors = []
        loss_array = []
        consecutive_increases = 0

        for epoch in range(epochs):
            output = self.forward(X_train)

            loss = self.compute_loss(y_train, output)
            loss_array.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

            self.backward(X_train, y_train, output)
            #train_error = np.mean(np.abs(y_train - output))
            train_error = np.mean(np.square(y_train - output))
            train_errors.append(train_error)

            val_output = self.forward(X_val)
            #val_error = np.mean(y_val - val_output)
            val_error = np.mean(np.square(y_val - val_output))
            val_errors.append(val_error)

            if epoch > 0 and val_errors[-1] > val_errors[-2]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases == early_stop_threshold:
                break

            """ 
            #if epoch > 0 and loss_array[-1] > loss_array[-2]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= early_stop_threshold:
                break """
                
        plt.plot(loss_array, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()

    def predict(self, X):
        return self.forward(X)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def treinar():
    # Carregando os dados do Excel
    data = pd.read_excel("dadosmamografia.xlsx")

    # Normalizando os dados de input
    X = normalize_data(data.iloc[:, :-1].values)

    # Selecionando a última coluna pra servir como valor de output
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # Dividindo os dados entre validação, treino e teste
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Criando os parâmetros da rede
    input_size = 5
    hidden_size = 5
    output_size = 1
    learning_rate = 0.1
    epochs = 1000
    early_stop_threshold = 5

    # Criando uma instância da rede com base nos dados acima
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Treinando a rede
    nn.train(X_train, y_train, X_val, y_val, epochs, early_stop_threshold)

    # Mostrar pesos antes e depois do treino
    print("Weights before training:")
    print(nn.weights_input_hidden_before)
    print(nn.weights_hidden_output_before)

    print("Weights after training:")
    print(nn.weights_input_hidden)
    print(nn.weights_hidden_output)

    # Comparando os resultados da rede treinada com os dados de teste
    test_output = nn.predict(X_test)
    test_error = np.mean(np.square(y_test - test_output))
    print("Test Error:", test_error)

def main():
    i = 0
    numeroDeTreinos = 1
    while i<numeroDeTreinos:
        treinar()
        i+=1
if __name__ == "__main__":
    main()
