from deep_neural_network_model import DNN
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, metrics, model_selection, preprocessing
from tensorflow.keras.datasets import cifar10

# Debugging with pseudo data
# model = DNN()
# X = np.array([[1,1,1], [2,2,3], [4,8,9], [12,12,12]]).T
# Y = np.array([1,0,0,1]).reshape(1, 4)

# model.train(X=X, Y=Y, layer_dims=[5,4,1])
# plt.plot(model.cost_during_training)

# Testing on iris dataset
iris = datasets.load_iris()

X = np.array(iris.data[:100])
Y = np.array(iris.target[:100])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.T
Y_train = Y_train.reshape(1,len(Y_train))
X_test = X_test.T
Y_test = Y_test.reshape(1,len(Y_test))

model = DNN()
model.train(X=X_train, Y=Y_train, layer_dims=[5,4,1])

plt.plot(model.cost_during_training)

preds = model.predict(X_test)
pred_labels = preds > 0.5
train_accuracy = np.sum(pred_labels == Y_test) / Y_test.shape[1]
print(train_accuracy)