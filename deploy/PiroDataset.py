import numpy as np
import glob
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class Piro:

    @staticmethod
    def load_data():
        def get_image_list(path):
            return glob.glob(path + '*')

        def split(X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            return X_train, X_test, y_train, y_test

        images_path = './PiRO_image_set/Data/'
        target_path = './PiRO_image_set/target.csv'

        image_list = sorted(get_image_list(images_path))
        images = np.array([np.array(cv.imread(path, 0)) for path in image_list])
        images = images.reshape(images.shape[0], 28, 28, 1)
        labels = to_categorical(np.loadtxt(target_path))

        return split(images, labels)


if __name__ == '__main__':
    print(Piro.load_data())
