from deep_neural_network_model import DNN
import numpy as np

model = DNN()
model.train(X=np.array([[1,2,3],[4,5,6]]), Y=[1,0], layer_dims=[3,2,1])