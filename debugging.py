from deep_neural_network_model import DNN
import numpy as np
from matplotlib import pyplot as plt

model = DNN()
X = np.array([[1,1,1], [2,2,3], [4,8,9], [12,12,12]]).T
Y = np.array([1,0,0,1]).reshape(1, 4)

model.train(X=X, Y=Y, layer_dims=[5,4,1])
plt.plot(model.cost_during_training)