import numpy as np

class DNN:
    """
    Structure: A Feedforward Neural Network
    Optimization: Gradient Descent Algorithm
    """
    def __init__(self) -> None:
        pass

    def train(self, X, Y, layer_dims, epoch=1_000, learning_rate=0.001):
        """Train the neural network with given parameters and data X, Y"""

        layer_dims.insert(0, X.shape[0])
        self.layer_dims = layer_dims
        parameters = self.initialize_parameters()

        for i in range(epoch):
            # Forward prop and output of the last layer
            forward_vars = self.forward_prop(parameters, X)
            predictions = forward_vars["A"+str(len(layer_dims)-1)]

            # Cost function is applied
            cost_val = self.cost(predictions, Y)
            
            # Backpropagation for calculating gradients and updating parameters
            grads = self.backward_prop(forward_vars, cost_val)
            parameters = self.update_parameters(learning_rate, grads)

        self.learned_parameters = parameters

    def predict(self, X):
        """Predict labels for given data X"""
        pass

    def initialize_parameters(self):
        """Initialize parameters with He initialization method"""
        
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters["W"+str(l)] = np.random.rand(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
            parameters["b"+str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def forward_prop(self, parameters, X):
        """Propagate forward in network"""
        
        forward_vars = {"A0": X}

        for l in range(1, len(self.layer_dims)):
            forward_vars["Z"+str(l)] = np.dot(parameters["W"+str(l)], forward_vars["A"+str(l-1)]) + parameters["b"+str(l)]
            forward_vars["A"+str(l)] = self.sigmoid(forward_vars["Z"+str(l)])

        return forward_vars

    def backward_prop(self):
        """Propagate backward in the network with calculating gradients"""
        pass

    def cost(self, Y_hat, Y):
        """Cross Entropy Loss is applied as Cost (Emprical Risk) """
        m = Y.shape[1]
        cost_value = (1/m) * np.sum(-(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)))

        return cost_value

    def update_parameters(self, learning_rate, grads):
        """Update parameters with gradients"""
        pass

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))