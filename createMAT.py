import keras
from scipy.io import savemat
import os

outputFolder = "D:\\postCompletion\\research\\data"

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
mdic = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
savemat(os.path.join(outputFolder, "cifar10.mat"), mdic)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
mdic = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
savemat(os.path.join(outputFolder, "cifar100.mat"), mdic)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
mdic = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
savemat(os.path.join(outputFolder, "mnist.mat"), mdic)