import cv2 as cv

from Preprocessor import *

class WordsExtractor:

    @staticmethod
    def extract(row_image):
        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (16, 64))

        img_copy = row_image.copy()
        row_image = Preprocessor.dilate(row_image, 4)
        row_image = Preprocessor.erode(row_image, 2)
        row_image = cv.GaussianBlur(row_image, (3,3), 1)

        img_copy = Preprocessor.erode(img_copy, 4)
        img_copy = Preprocessor.dilate(img_copy, kernel=kernel_rect)
        contours, _ = cv.findContours(img_copy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        words = []

        for i, c in enumerate(contours):
            boundRect = cv.boundingRect(c)

            x = boundRect[0]
            y = boundRect[1]
            width = boundRect[2]
            height = boundRect[3]

            marginX = 2
            marginY = 2
            rect_kernel_size = kernel_rect.shape
            if width > (rect_kernel_size[0] + marginX) and height > (rect_kernel_size[1] + marginY):
            #             cv.rectangle(row_test, (int(boundRect[0]), int(boundRect[1])), \
            #                 (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (127,127,127), 1)

                word = row_image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
                words.append(word)

        return words
