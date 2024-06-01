import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward_propagation(self, X):
        # Input to hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Hidden layer to output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def compute_loss(self, Y, Y_hat):
        # Binary cross-entropy loss
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m
        return loss
    
    def backward_propagation(self, X, Y):
        m = X.shape[0]
        
        # Compute the error in the output layer
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Compute the error in the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            Y_hat = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(Y, Y_hat)
            
            # Backward propagation
            self.backward_propagation(X, Y)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage:
# Assuming input data X and labels Y are available and properly preprocessed
# X = np.array([...])
# Y = np.array([...])
# nn = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1)
# nn.train(X, Y)
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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
nn.train(X_train, y_train,epochs)

# Comparando os resultados da rede treinada com os dados de teste
test_output = nn.predict(X_test)
test_error = np.mean(np.square(y_test - test_output))
print("Test Error:", test_error)