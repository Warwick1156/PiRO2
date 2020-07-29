import cv2 as cv
from PIL import Image

from Preprocessor import *

class DigitsExtractor:

    @staticmethod
    def extract(words_list):
        kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (4, 16))

        digits = []

        # words are in reverse order, so first element is the last word
        # we assume that the 2 first words are name & surname, so we skip it
        # for word in words_list[:-2]:
        for word in words_list[:-2]:
            word_copy = word.copy()
            word_copy = Preprocessor.erode(word_copy, 4)
            word_copy = Preprocessor.dilate(word_copy, kernel=kernel_rect)
            contours, _ = cv.findContours(word_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for i, c in enumerate(contours):
                boundRect = cv.boundingRect(c)

                x = boundRect[0]
                y = boundRect[1]
                width = boundRect[2]
                height = boundRect[3]

                marginX = 0
                marginY = 0
                rect_kernel_size = kernel_rect.shape
                if width > (rect_kernel_size[0] + marginX) and height > (rect_kernel_size[1] + marginY):
                    #             cv.rectangle(row_test, (int(boundRect[0]), int(boundRect[1])), \
                    #                 (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (127,127,127), 1)
                    pass
                digit = word[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]].copy()
                digits.append(digit)

        digits = digits[::-1]

        return [DigitsExtractor.format_for_mnist(d) for d in digits]

    @staticmethod
    def place_in_center(img):
        img = img.astype(int)
        rows, cols = img.shape
        new_size = rows if rows > cols else cols

        new_image = np.zeros((new_size, new_size)).astype(int)

        x = int(new_size / 2) - int(rows / 2)
        y = int(new_size / 2) - int(cols / 2)

        new_image[x:x + rows, y:y + cols] = img
        return new_image

    @staticmethod
    def format_for_mnist(digit_image):
        t = digit_image
        t = DigitsExtractor.place_in_center(t).astype('uint8')
        t = Image.fromarray(t)
        t = t.resize((28, 28), Image.ANTIALIAS)
        t = np.asarray(t)
        return t
