""" Neural Network for classic MNIST Dataset """
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_TRAIN[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')


def sigmoid(x):
    """ sigmoid function for normalization """
    return 1 / (1 + np.exp(-x))

def prep(x, y):
    """ data preperation from integer array to 4D shaped_array """
    out_y = keras.utils.to_categorical(y, 10)

    shaped_array = x.reshape(-1, 28, 28, 1)
    preped_x = sigmoid(shaped_array)
    return preped_x, out_y

X_T_PREP, Y_T_PREP = prep(X_TRAIN, Y_TRAIN)

X_TEST_PREP, Y_TEST_PREP = prep(X_TEST, Y_TEST)

M = Sequential()
M.add(Conv2D(100, kernel_size=(3, 3),
             activation='relu',
             data_format='channels_last',
             input_shape=(28, 28, 1)))
M.add(Conv2D(20, kernel_size=(3, 3),
             activation='relu',
             data_format='channels_last'))

M.add(Flatten())
M.add(Dense(50, activation='relu'))
M.add(Dense(10, activation='softmax'))
M.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
M.fit(X_T_PREP, Y_T_PREP, epochs=10)

Y_PRED = M.predict_classes(X_TEST_PREP)
C = confusion_matrix(Y_TEST, Y_PRED)
print(C)
plt.figure(figsize=[10.0, 7.5])
sns.heatmap(C)
plt.show()
