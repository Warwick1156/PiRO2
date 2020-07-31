import sys
import numpy as np
import cv2 as cv

from Preprocessor import *


class BoundingBoxSplitter:

    @staticmethod
    def get_text_area(binary_image):
        Preprocessor.erode(binary_image, 4)
        pts = cv.findNonZero(binary_image)
        return cv.minAreaRect(pts)

    @staticmethod
    def rotate_text(text_aera, binary_image, angle_offset=0):
        (cx, cy), (w, h), ang = text_aera

        M = cv.getRotationMatrix2D((cx, cy), ang + angle_offset, scale=1.0)
        rotated = cv.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))
        return rotated

    @staticmethod
    def split(image_orig):
        # print("Detecting rows with boundary boxes...")

        image = image_orig.copy()
        image_orig_2 = image_orig.copy()
        image_orig = image_orig.copy()


        image = Preprocessor.erode(image, 4)
        image = Preprocessor.dilate(image, 6)

        # image = Preprocessor.dilate(image, 4)
        # image = Preprocessor.erode(image, 4)

        rect_kernel_size = (int(image_orig.shape[1]/3), 4)

        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, rect_kernel_size)
        image = Preprocessor.dilate(image, kernel=kernel_rect)

        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        rows = []
        i=1
        contours.reverse()
        for c in contours:
            # c = cv.convexHull(c)
            boundRect = cv.boundingRect(c)
            x = boundRect[0]
            y = boundRect[1]
            width  = boundRect[2]
            height = boundRect[3]

            marginX = 0
            marginY = 0

            if width > (rect_kernel_size[0] + marginX) and height > (rect_kernel_size[1] + marginY):
                cv.rectangle(image_orig_2, (int(boundRect[0]), int(boundRect[1])),
                    (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (i,i,i), -1)

                row = (image_orig[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]]).copy()

                row_coords = (int(boundRect[0]), int(boundRect[1]))
                row_no = i
                rows.append((row, row_coords, row_no))
                i+=1

        return rows, image_orig_2

    @staticmethod
    def split_rows(binary_image, angle_offset=90):
        # print("Getting text area...")
        # area = BoundingBoxSplitter.get_text_area(binary_image)

        # print("Rotating text area...")
        # rotated = BoundingBoxSplitter.rotate_text(area, binary_image, angle_offset)
        rotated = np.invert(binary_image)

        return BoundingBoxSplitter.split(rotated)
