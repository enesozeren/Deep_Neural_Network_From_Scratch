import numpy as np

class DNN:
    def __init__(self) -> None:
        pass

    def train(self, X, Y, layer_dims, epoch=1_000, learning_rate=0.001):
        """Trains the neural network with given parameters and data X, Y"""

        layer_dims.insert(0, X.shape[1])
        parameters = self.initialize_parameters(layer_dims)

        for i in range(epoch):
            self.forward_prop()
            self.cost()
            self.backward_prop()
            self.parameters = self.update_parameters(learning_rate)

    def predict(self, X):
        """Predicts labels for given data X"""
        pass

    def initialize_parameters(self, layer_dims):
        """Initialize parameters with He initialization method"""
        
        parameters = {}
        for l in range(1, len(layer_dims)):
            parameters["W"+str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1])
            parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))

        return parameters

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass

    def cost(self):
        pass

    def update_parameters(self, learning_rate):
        pass