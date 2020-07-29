import cv2 as cv

from Preprocessor import *

class WordsExtractor:

    @staticmethod
    def extract(row_image):
        x, y = row_image.shape
        kx, ky = (16, 32)
        x = x if x < kx else kx
        y = y if y < ky else ky
        rect_kernel_size = x, y

        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, rect_kernel_size)

        img_copy = row_image.copy()
        row_image = Preprocessor.dilate(row_image, 4)

        row_image = cv.GaussianBlur(row_image, (3,3), 1)

        img_copy = Preprocessor.erode(img_copy, 2)
        img_copy = Preprocessor.dilate(img_copy, kernel=kernel_rect)
        contours, _ = cv.findContours(img_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        words = []

        row_image_cpy = row_image.copy()

        for i, c in enumerate(contours):
            boundRect = cv.boundingRect(c)

            x = boundRect[0]
            y = boundRect[1]
            width = boundRect[2]
            height = boundRect[3]

            marginX = 1
            marginY = 1

            if width > (rect_kernel_size[0] + marginX) and height > (rect_kernel_size[1] + marginY):
                cv.rectangle(row_image_cpy, (int(boundRect[0]), int(boundRect[1])), \
                    (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (127,127,127), 1)

                word = row_image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
                words.append(word)

        return [row_image_cpy]
        return words
