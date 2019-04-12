from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

from matplotlib import pyplot as plt

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(xtrain[i], cmap=plt.cm.Greys)
    plt.axis('off')
