import sys
import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt
import imutils

from Preprocessor import *


class BoundingBoxSplitter:

    @staticmethod
    def get_text_area(binary_image):
        pts = cv.findNonZero(binary_image)
        return cv.minAreaRect(pts)

    @staticmethod
    def rotate_text(text_aera, binary_image, angle_offset=90):
        (cx, cy), (w, h), ang = text_aera

        M = cv.getRotationMatrix2D((cx, cy), ang + angle_offset, scale=1.0)
        rotated = cv.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))
        return rotated

    @staticmethod
    def split(image_orig):
        print("Detecting rows with boundary boxes...")

        image = image_orig.copy()

        image = Preprocessor.erode(image, 16)
        image = Preprocessor.dilate(image, 16)

        image = Preprocessor.dilate(image, 16)
        image = Preprocessor.erode(image, 16)

        rect_kernel_size = (int(image_orig.shape[1]/3), 40)

        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, rect_kernel_size)
        image = Preprocessor.dilate(image, kernel=kernel_rect)

        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        rows = []
        for c in contours:
            boundRect = cv.boundingRect(c)
            x = boundRect[0]
            y = boundRect[1]
            width  = boundRect[2]
            height = boundRect[3]

            margin = 48

            if width > (rect_kernel_size[0] + margin) and height > (rect_kernel_size[1] + margin):
                cv.rectangle(image_orig, (int(boundRect[0]), int(boundRect[1])), \
                    (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (127,127,127), 16)

            # UNCOMMENT THIS
            # row = image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
            # rows.append(row)

        return [image, image_orig] # rows

    @staticmethod
    def split_rows(binary_image, angle_offset=90):
        print("Getting text area...")
        area = BoundingBoxSplitter.get_text_area(binary_image)

        print("Rotating text area...")
        rotated = BoundingBoxSplitter.rotate_text(area, binary_image, angle_offset)

        rotated = np.invert(rotated)

        return BoundingBoxSplitter.split(rotated)
