from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

from sklearn.model_selection import KFold
from matplotlib import pyplot

import numpy as np

import cv2 as cv
from DigitProcessor import DigitProcessor
from PiroDataset import Piro


class Classifier:
    def __init__(self, model_path=None, ):
        try:
            self.model = load_model(model_path)
        except IOError:
            self.model = Classifier.create_model()
            self.model = Classifier.train_mnist(self.model)
            try:
                self.model = Classifier.train_piro(self.model, epochs=20)
            except:
                pass

    def predict(self, img):
        img = img.reshape(1, 28, 28, 1)
        img = Classifier.scale_value(img)
        result = self.model.predict_classes(img)
        return result[0]

    def predict_proba(self, img):
        img = img.reshape(1, 28, 28, 1)
        img = Classifier.scale_value(img)
        result = self.model.predict(img)
        result = [np.round(x * 100, 2) for x in result]
        return result

    @staticmethod
    def create_model(conv_kernel=3, pooling_kernel=2, activation='relu', optimizer=SGD(lr=0.001, momentum=0.9)):
        model = Sequential()
        model.add(Conv2D(32, (conv_kernel, conv_kernel), activation=activation, kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((pooling_kernel, pooling_kernel)))
        model.add(Conv2D(64, (conv_kernel, conv_kernel), activation=activation, kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (conv_kernel, conv_kernel), activation=activation, kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((pooling_kernel, pooling_kernel)))
        model.add(Flatten())
        model.add(Dense(100, activation=activation, kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def scale_value(img):
        img_float = img.astype('float32')
        return img_float / 255.0

    @staticmethod
    def load_mnist():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

        return X_train, y_train, X_test, y_test

    @staticmethod
    def one_hot_encode(data):
        return to_categorical(data)

    @staticmethod
    def train_mnist(model, save=False, epochs=10, batch_size=32, verbose=1):
        X_train, y_train, X_test, y_test = Classifier.load_mnist()

        y_train = Classifier.one_hot_encode(y_train)
        y_test = Classifier.one_hot_encode(y_test)

        X_train = Classifier.scale_value(X_train)
        X_test = Classifier.scale_value(X_test)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        Classifier.evaluate(model, X_test, y_test)

        if save:
            model.save('../model/keras_mnist.h5')

        return model

    @staticmethod
    def train_piro(model, save=False, epochs=10, batch_size=32, verbose=1):
        X_train, X_test, y_train, y_test = Piro.load_data()

        X_train = Classifier.scale_value(X_train)
        X_test = Classifier.scale_value(X_test)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        Classifier.evaluate(model, X_test, y_test)

        if save:
            model.save('../model/keras_piro.h5')

        return model

    @staticmethod
    def evaluate(model, X, y, verbose=1):
        _, acc = model.evaluate(X, y, verbose)
        print('> %.3f' % (acc * 100.0))


if __name__ == '__main__':
    clf = Classifier('../model/keras_mnist.h5')
    print('Classifier created')

    X_train, X_test, y_train, y_test = Piro.load_data()

    X_train = Classifier.scale_value(X_train)
    X_test = Classifier.scale_value(X_test)

    Classifier.evaluate(clf.model, X_test, y_test)
    Classifier.train_piro(clf.model, epochs=19, save=True)


