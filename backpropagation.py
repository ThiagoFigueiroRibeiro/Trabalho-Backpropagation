import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

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

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = purelin(self.hidden_input)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))
        return self.output

    def backward(self, X, y, output):
        error = y - output
        output_delta = error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * purelin_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate

    def train(self, X_train, y_train, X_val, y_val, epochs, early_stop_threshold):
        train_errors = []
        val_errors = []
        consecutive_increases = 0

        for epoch in range(epochs):
            output = self.forward(X_train)
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

        #plt.plot(train_errors, label='Training Error')
        #plt.plot(val_errors, label='Validation Error')
        #plt.xlabel('Epochs')
        #plt.ylabel('Mean Squared Error')
        #plt.legend()
        #plt.show()

    def predict(self, X):
        return self.forward(X)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def treinar():
    # Load data from Excel file
    data = pd.read_excel("dadosmamografia.xlsx")

    # Normalize input data
    X = normalize_data(data.iloc[:, :-1].values)

    # Extract output data
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # Split data into train, validation, and test sets
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Define neural network parameters
    input_size = X.shape[1]
    hidden_size = 5  # Change this to vary number of neurons in hidden layer
    output_size = 1
    learning_rate = 0.1
    epochs = 1000
    early_stop_threshold = 5

    # Initialize and train neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    nn.train(X_train, y_train, X_val, y_val, epochs, early_stop_threshold)
    # Predict on test data
    test_output = nn.predict(X_test)
    test_error = np.mean(np.square(y_test - test_output))
    print("Test Error:", test_error)

    # Print weights before and after training
    #print("Weights before training:")
    #print(nn.weights_input_hidden)
    #print(nn.weights_hidden_output)

    #print("Weights after training:")
    #print(nn.weights_input_hidden)
    #print(nn.weights_hidden_output)

def main():
    i = 0
    numeroDeTreinos = 10
    while i<numeroDeTreinos:
        treinar()
        i+=1
if __name__ == "__main__":
    main()
