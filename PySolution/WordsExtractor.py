import cv2 as cv

from Preprocessor import *

class WordsExtractor:

    @staticmethod
    def extract(row_image, row_coords, row_no, image):
        x, y = row_image.shape
        kx, ky = (48, 32) # kx, ky = (64, 24)
        x = (x if x < kx else kx) - 4
        y = (y if y < ky else ky) - 4
        rect_kernel_size = x, y

        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, rect_kernel_size)

        img_copy = row_image.copy()
        row_image = Preprocessor.dilate(row_image, 4)

        row_image = cv.GaussianBlur(row_image, (3,3), 1)

        img_copy = Preprocessor.erode(img_copy, 2)
        img_copy = Preprocessor.dilate(img_copy, kernel=kernel_rect)
        contours, _ = cv.findContours(img_copy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        words = []

        row_image_cpy = row_image.copy()
        image = image.copy().astype('uint8')

        for i, c in enumerate(contours):
            boundRect = cv.boundingRect(c)

            x = boundRect[0]
            y = boundRect[1]
            width = boundRect[2]
            height = boundRect[3]

            marginX = 1
            marginY = 1

            if width > (rect_kernel_size[0] + marginX) and height > (rect_kernel_size[1] + marginY):
                cv.rectangle(image, (int(row_coords[0] + boundRect[0] ), int(row_coords[1] + boundRect[1] )),
                    (int(row_coords[0] + boundRect[0]+boundRect[2]), int(row_coords[1] + boundRect[1]+boundRect[3])), (row_no,row_no,row_no), -1)

                word = row_image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
                words.append((boundRect[0], word))

        words = sorted(words, key=lambda x: x[0])
        words = [w[1] for w in words]
        return words, image
