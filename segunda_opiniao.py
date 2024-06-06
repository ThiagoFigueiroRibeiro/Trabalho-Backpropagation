import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation function (ReLU) and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error loss function and its derivative
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return y_pred - y_true

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        return self.z2

    def backward(self, x, y, output):
        m = y.shape[1]
        
        # Output layer gradients
        d_z2 = mse_loss_derivative(y, output)
        d_W2 = np.dot(d_z2, self.a1.T) / m
        d_b2 = np.sum(d_z2, axis=1, keepdims=True) / m
        
        # Hidden layer gradients
        d_a1 = np.dot(self.W2.T, d_z2)
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_W1 = np.dot(d_z1, x.T) / m
        d_b1 = np.sum(d_z1, axis=1, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

    def train(self, x, y, iterations):
        train_errors = []
        val_errors = []
        loss_array = []

        best_loss = 9999999999
        paciencia = 5
        for i in range(iterations):
            output = self.forward(x)
            loss = mse_loss(y, output)
            loss_array.append(loss)

            self.backward(x, y, output)

            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= paciencia:
                print(f"Early stopping at iteration {i}, validation loss: {loss}")
                break

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")
        
        # Criação das figuras
        fig, axs = plt.subplots(2)

        # Plotando a figura do Loss
        axs[0].plot(loss_array, label='Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plotando a figura do erro de treino e erro de validação
        axs[1].plot(train_errors, label='Training Error')
        axs[1].plot(val_errors, label='Validation Error')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Mean Squared Error')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Dummy data (X: 2 features, Y: 1 target)
    ##X = np.random.rand(2, 10)
    ##Y = np.random.rand(1, 10)

    data = pd.read_excel("dadosmamografia.xlsx")

    # Normalizando os dados de input
    X = normalize_data(data.iloc[:, :-1].values)

    # Selecionando a última coluna pra servir como valor de output
    Y = data.iloc[:, -1].values.reshape(-1, 1)
    
    # Parameters
    input_size = X.shape[0]
    hidden_size = 5
    output_size = Y.shape[0]
    learning_rate = 0.01
    iterations = 10
    
    # Initialize and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, Y, iterations)
