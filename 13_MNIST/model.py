import numpy as np


# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(Z):
    return np.maximum(0, Z)


# Define softmax activation function
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def activation_derivative(input, type=relu):
    if type == relu:
        return input > 0

    elif type == sigmoid:
        s = sigmoid(input)
        return s * (1 - s)

    else:
        print("Invalid activation function")


def loss_derivative(input, Y, type):
    if type == "cross-entropy":
        input = np.where(input == 0, 0.0001, input)
        input = np.where(input == 1, 0.9999, input)

        dv_1 = -np.divide(Y, input)

        dv_2 = np.divide(1 - Y, 1 - input)

        return dv_1 + dv_2

    elif type == "quadratic":
        return input - Y

    else:
        print("invalid loss function")


class NN_MNIST:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        loss_type="cross-entropy",
        weights_init=-1,
        last_func_activation=relu,
        hidden_activation_func=relu,
    ):
        if weights_init == "random" or weights_init == "standard normal":
            self.W1 = np.random.randn(input_size, hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b1 = np.random.randn(1, hidden_size)
            self.b2 = np.random.randn(1, output_size)

        elif weights_init == 1:
            self.W1 = np.ones((input_size, hidden_size))
            self.W2 = np.ones((hidden_size, output_size))
            self.b1 = np.ones((1, hidden_size))
            self.b2 = np.ones((1, output_size))

        elif weights_init == 0:
            self.W1 = np.zeros((input_size, hidden_size))
            self.W2 = np.zeros((hidden_size, output_size))
            self.b1 = np.zeros((1, hidden_size))
            self.b2 = np.zeros((1, output_size))

        elif weights_init == "He":
            limit_1 = np.sqrt(2 / input_size)
            limit_2 = np.sqrt(2 / hidden_size)
            self.W1 = np.random.normal(loc=0.0, scale=limit_1, size=(input_size, hidden_size))
            self.W2 = np.random.normal(loc=0.0, scale=limit_2, size=(hidden_size, output_size))
            self.b1 = np.random.normal(loc=0.0, scale=limit_1, size=(1, hidden_size))
            self.b2 = np.random.normal(loc=0.0, scale=limit_2, size=(1, output_size))

        elif weights_init == "Xavier":
            limit_1 = np.sqrt(6 / (input_size + hidden_size))
            limit_2 = np.sqrt(6 / (hidden_size + output_size))
            self.W1 = np.random.uniform(low=-limit_1, high=limit_1, size=(input_size, hidden_size))
            self.W2 = np.random.uniform(low=-limit_2, high=limit_2, size=(hidden_size, output_size))
            self.b1 = np.random.uniform(low=-limit_1, high=limit_1, size=(1, hidden_size))
            self.b2 = np.random.uniform(low=-limit_2, high=limit_2, size=(1, output_size))

        self.dW2 = 0
        self.db2 = 0
        self.dW1 = 0
        self.db1 = 0

        self.loss_type = loss_type
        self.last_activation = last_func_activation
        self.hidden_activation = hidden_activation_func

        self.hidden_layer_O = 0
        self.hidden_layer_A = 0
        self.output_layer_A = 0
        self.output_layer_O = 0

    def forward(self, X):
        self.hidden_layer_A = np.dot(X.reshape(-1, 784), self.W1) + self.b1
        self.hidden_layer_O = self.hidden_activation(self.hidden_layer_A)

        self.output_layer_A = np.dot(self.hidden_layer_O, self.W2) + self.b2
        self.output_layer_O = self.last_activation(self.output_layer_A)

        return self.output_layer_O

    def backward(self, X, Y):
        # Backward pass
        m = Y.shape[0]

        if self.loss_type == "quadratic":
            delta_output = self.output_layer_O - Y
            delta_output = delta_output * activation_derivative(input=self.output_layer_A, type=self.last_activation)

        # elif self.loss_type == "cross-entropy":
        #     # versione compressa poichè la funzione di attivazione finale è sempre sigmoid.
        #     delta_output = self.output_layer_O - Y

        elif self.loss_type == "cross-entropy":
            delta_output = loss_derivative(self.output_layer_O, Y, type=self.loss_type) * activation_derivative(
                input=self.output_layer_A, type=self.last_activation
            )

        self.dW2 = 1 / m * np.dot(self.hidden_layer_O.T, delta_output)
        self.db2 = 1 / m * np.sum(delta_output, axis=0, keepdims=True)

        delta_hidden = (np.matmul(delta_output, self.W2.T)) * activation_derivative(
            input=self.hidden_layer_A, type=self.hidden_activation
        )

        self.dW1 = 1 / m * np.dot(X.reshape(-1, 784).T, delta_hidden)
        self.db1 = 1 / m * np.sum(delta_hidden, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def get_weights(self):
        weights = {"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2}
        return weights
