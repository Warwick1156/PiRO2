import sys
import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt
import imutils

from Preprocessor import *

def debug_plot_cv_img(img, cmap=None):
    fig, ax = plt.subplots(figsize=(10, 22))
    ax.imshow(img, interpolation='nearest', cmap=cmap)
    plt.tight_layout()


class RowSplitter:

    @staticmethod
    def find_boundries(img, threshold=2):
        hist = cv.reduce(img, 1, cv.REDUCE_AVG).reshape(-1)
        h, w = img.shape[:2]
        uppers = [y for y in range(h - 1) if hist[y] <= threshold and hist[y + 1] > threshold]
        lowers = [y for y in range(h - 1) if hist[y] > threshold and hist[y + 1] <= threshold]
        return uppers, lowers

    @staticmethod
    def draw_boundries(img, upper, lower):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        for y in upper:
            cv.line(img, (0, y), (w, y), (255, 0, 0), 1)

        for y in lower:
            cv.line(img, (0, y), (w, y), (0, 255, 0), 1)

        return img

    @staticmethod
    def get_text_aera(binary_image):
        pts = cv.findNonZero(binary_image)
        return cv.minAreaRect(pts)

    @staticmethod
    def rotate_text(text_aera, binary_image, angle_offset=90):
        (cx, cy), (w, h), ang = text_aera

        M = cv.getRotationMatrix2D((cx, cy), ang + angle_offset, scale=1.0)
        # rotated = cv.warpAffine(binary_image, M, (img.shape[1], img.shape[0]))
        rotated = cv.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))
        return rotated

    @staticmethod
    def split(img, upper, lower, show=True, offset=0):
        assert len(upper) == len(lower)

        rows = []
        print("Splitting...")
        for i in range(len(upper)):
            upper_boundry = max(0, upper[i] - offset)
            lower_boundry = min(img.shape[0], lower[i] + offset)
            #         image = img[upper[i]:lower[i], :]
            image = img[upper_boundry:lower_boundry, :]
            rows.append(image)

        # if show:
        #     for row in rows:
        #         debug_plot_cv_img(row)

        return rows

    @staticmethod
    def split_rows(binary_image, plot=True, threshold=2, show_rows=False, offset=0, angle_offset=90):

        print("Getting text area...")
        orig_image = binary_image.copy()
        binary_image = Preprocessor.dilate(binary_image, 20)
        binary_image = Preprocessor.erode(binary_image, 10)
        aera = RowSplitter.get_text_aera(binary_image)

        print("Rotating text area...")
        rotated = RowSplitter.rotate_text(aera, binary_image, angle_offset)
        cv2.imshow("", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        orig_image_rotated = RowSplitter.rotate_text(aera, orig_image)
        rotated = np.invert(rotated)

        print("Finding boundaries...")
        upper, lower = RowSplitter.find_boundries(rotated, threshold)

        # if plot:
        #     drawing = RowSplitter.draw_boundries(rotated, upper, lower)
        #     debug_plot_cv_img(drawing)

        return RowSplitter.split(orig_image_rotated, upper, lower, show_rows, offset)

