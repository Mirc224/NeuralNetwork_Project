import numpy as np

class MyNeuralNetwork:
    def __init__(self, nn_architecture: dict):
        self._layer_params = {}
        self._architecture = nn_architecture
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer['input_dim']
            layer_output_size = layer['output_dim']

            self._layer_params['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            self._layer_params['b' + str(layer_idx)] = np.zeros((layer_output_size, 1))
        self._layer_params['W1'] = np.array([[0.1, 0.6],
                                             [0.2, 0.4],
                                             [0.3, 0.7]])
        self._layer_params['W2'] = np.array([[0.1, 0.4, 0.9]])
        self._z_Outputs = {}
        self._a_Outputs = {}
        self._gradients = {}

    def back_prop(self, Y):
        self._gradients = {}
        # TODO: back prop
        a_curr = self._a_Outputs['a' + str(len(self._architecture))]
        m = a_curr.shape[1]
        error = -(1/m)*(Y - a_curr)

        for layer_idx_prev, layer in reversed(list(enumerate(self._architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer['activation']
            dA_curr = error
            if activ_function_curr == 'sigmoid':
                dZ_curr = self.sigmoid_derivate(a_curr)
            error = dA_curr * dZ_curr
            A_prev = self._a_Outputs['a' + str(layer_idx_prev)]
            dW_curr = np.dot(error, A_prev.T)
            db_curr = np.dot(error, np.ones((1, a_curr.shape[1])).T)
            self._gradients['W' + str(layer_idx_curr)] = dW_curr
            self._gradients['b' + str(layer_idx_curr)] = db_curr
            W_curr = self._layer_params['W' + str(layer_idx_curr)]

            error = np.dot(W_curr.T, error)
            a_curr = self._a_Outputs['a' + str(layer_idx_prev)]

    def update_parameters(self, l_rate=0.05):
        for key in self._gradients:
            self._layer_params[key] = self._layer_params[key] - l_rate*self._gradients[key]

    def feed_forward(self, X):
        self._z_Outputs = {}
        self._a_Outputs = {}
        a_prev = X
        self._a_Outputs['a0'] = X
        for idx, layer in enumerate(self._architecture):
            layer_number = idx + 1
            z_curr = np.dot(self._layer_params['W'+str(layer_number)], a_prev) + self._layer_params['b'+str(layer_number)]
            self._z_Outputs['z' + str(layer_number)] = z_curr
            if layer['activation'] == 'sigmoid':
                a_curr = self.sigmoid(z_curr)
            self._a_Outputs['a' + str(layer_number)] = a_curr
            a_prev = a_curr

    def predict(self, X):
        self.feed_forward(X)
        return self.get_predicted()

    def get_predicted(self):
        return self._a_Outputs['a' + str(len(self._architecture))]

    def calculate_cost(self, Y, Y_hat):
        m = Y_hat.shape[1]
        return (1/(2*m)) * np.sum(np.square(Y - Y_hat))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivate(self, A):
        return A * (1 - A)

    def train(self, X, Y, epochs, l_rate=0.001):
        for epoch in range(epochs):
            self.feed_forward(X)
            print(f'Epoch {epoch} cost: {self.calculate_cost(Y, self.get_predicted())}')
            self.back_prop(Y)
            self.update_parameters(l_rate)




nn = MyNeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': 'sigmoid'},
                      {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}])

nn.train(np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]]), np.array([[0, 1, 1, 0]]), 100000, 5)

print(nn.predict(np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]])))