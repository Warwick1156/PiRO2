import cv2 as cv

from Preprocessor import *

class WordsExtractor:

    @staticmethod
    def extract(row_image):
        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (16, 64))

        img_copy = row_image.copy()
        img_copy = Preprocessor.erode(img_copy, 4)
        img_copy = Preprocessor.dilate(img_copy, kernel=kernel_rect)
        contours, _ = cv.findContours(img_copy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        words = []

        for i, c in enumerate(contours):
            boundRect = cv.boundingRect(c)

            #             cv.rectangle(row_test, (int(boundRect[0]), int(boundRect[1])), \
            #                 (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (127,127,127), 1)

            word = row_image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
            words.append(word)

        return words
