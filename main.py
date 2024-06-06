import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# Criando as funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def purelin(x):
    return x
def purelin_derivative(x):
    return np.ones_like(x)

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return np.where(x > 0, 1, 0)
    
def step_function(x, threshold=0):
  return (x >= threshold) * 1

def step_function_derivative(x, threshold=0):
  return 2.0 * (x > threshold)

# Criando a rede neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Criando os inputs de entrada, quantidade de neurônios e outputs, além da taxa de aprendizado
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializando os pesos
        self.weights_input_hidden = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_size))
        self.weights_input_hidden_before = copy.copy(self.weights_input_hidden)

        self.weights_hidden_output = np.random.uniform(-1, 1, size=(self.hidden_size, self.output_size))
        self.weights_hidden_output_before = copy.copy(self.weights_hidden_output)

    # Criando o Feed forward
    def forward(self, X_train):
        # Feed forward: Os dados de entrada são multiplicados pelos pesos naquele momento. (#1)
        # Então, o resultado disso (equivalente à camada escondida) é multiplicada pela sua função
        # de ativação naquele ponto.  (#2)
        # Por fim, o output é retornado como sendo a função de ativação multiplicada pelo resultado de #2
        # e os pesos da camada escondida. (#3)
        self.hidden_input = np.dot(X_train, self.weights_input_hidden)  #1
        self.hidden_output = sigmoid(self.hidden_input)     #2
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output)) #3
        return self.output

    # Criando o backpropagation
    def backward(self, X_train, y_train, output):

        # O erro é definido como o resultado esperado menos o resultado do treino naquele momento (#1).
        # Um delta de erro é a derivada da função de ativação multiplicada pelo erro (#2)
        error = y_train - output    #1 
        output_delta = error * sigmoid_derivative(output)   #2

        # Da mesma forma, o erro na camada escondida é esse delta multiplicado por cada peso dessa camada (#3).
        # E assim um delta da camada escondida vai ser a derivada da funçaõ de ativação multiplicada por 
        # cada erro de cada neurônio (#4)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)   #3
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)    #4

        # Aqui os pesos da camada oculta e camada de saída são atualizados com base no sinal 
        # de erro na camada de saída e na camada oculta.
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X_train.T.dot(hidden_delta) * self.learning_rate
    
    # Cálculo da função loss
    def compute_loss(self, y_train, Y_hat):
        # Binary cross-entropy loss
        m = y_train.shape[0]
        loss = -np.sum(y_train * np.log(Y_hat) + (1 - y_train) * np.log(1 - Y_hat)) / m
        return loss
    
    # Treino da rede
    def train(self, X_train, y_train, X_val, y_val, epochs, early_stop_threshold):
        # Inicialização dos arrays de treino, validação e loss
        train_errors = []
        val_errors = []
        #loss_array = []
        consecutive_increases = 0

        #Para a quantidade de épocas...
        for epoch in range(epochs):

            # Feed forward
            output = self.forward(X_train)

            # Cálculo da função loss
            #loss = self.compute_loss(y_train, output)
            #loss_array.append(loss)
            #if epoch % 100 == 0:
            #    print(f'Epoch {epoch}, Loss: {loss}')

            # Backward propagation
            self.backward(X_train, y_train, output)

            # Cálculo do erro médio quadrático do treino
            #train_error = np.mean(np.abs(y_train - output))
            train_error = np.mean(np.square(y_train - output))
            train_errors.append(train_error)

            # Cálculo do erro médio quadrático da validação
            val_output = self.forward(X_val)
            #val_error = np.mean(y_val - val_output)
            val_error = np.mean(np.square(y_val - val_output))
            val_errors.append(val_error)

            # Condição de parada do treino:
            # Se erro de validação aumentar em N épocas consecutivas, o treino deve parar
            if epoch > 0 and val_errors[-1] > val_errors[-2]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases == early_stop_threshold:
                break

            '''
            # Condição de parada do treino:
            # Se a função loss aumentar em N épocas consecutivas, o treino deve parar

            if epoch > 0 and loss_array[-1] > loss_array[-2]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= early_stop_threshold:
                break
            
            '''


        # Plotando a figura do erro de treino e erro de validação
        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Depois de treinada, a rede pode prever novos resultados
    def predict(self, X):
        return self.forward(X)
    
# Normalizar os dados com base nos maiores e menores valores
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
    hidden_size = 10
    output_size = 1
    learning_rate = 0.01
    epochs = 500
    early_stop_threshold = 5

    # Criando uma instância da rede com base nos dados acima
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Treinando a rede
    nn.train(X_train, y_train, X_val, y_val, epochs, early_stop_threshold)

    # Mostrar pesos antes e depois do treino
    print("Pesos antes:")
    print(nn.weights_input_hidden_before)
    print(nn.weights_hidden_output_before)

    print("Pesos depois:")
    print(nn.weights_input_hidden)
    print(nn.weights_hidden_output)

    # Comparando os resultados da rede treinada com os dados de teste
    test_output = nn.predict(X_test)
    test_error = np.mean(np.square(y_test - test_output))
    print("Acurácia:", 1 - test_error)

def main():
    i = 0
    numeroDeTreinos = 1
    while i<numeroDeTreinos:
        treinar()
        i+=1
if __name__ == "__main__":
    main()

