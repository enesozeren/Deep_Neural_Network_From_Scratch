from deep_neural_network_model import DNN
import numpy as np

model = DNN()
X = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]).T
Y = np.array([1,0,0,1]).reshape(1, 4)

model.train(X=X, Y=Y, layer_dims=[5,4,1])