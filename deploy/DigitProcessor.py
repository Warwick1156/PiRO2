import cv2 as cv
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, square, disk, star

class DigitProcessor:

    @staticmethod
    def process(img):
        mask = np.zeros([3, 3])
        mask[1, :] = 1
        mask[:, 1] = 1
        img = dilation(img, mask)
        # img = dilation(img, mask)

        # mask = img <= 127
        # img[mask] = 0

        cv.imwrite('../test.png', img)
        return img
