import sys
import os

from OCR import *

from PIL import Image
import cv2 as cv


class PiRO2:

    def __init__(self):
        self.input_path = ""
        self.n = -1
        self.output_path = ""

    def _process(self, k):
        img = self._load_image(k)
        # print("Loaded image no. {}".format(k))

        indices, processed_img = self._process_image(img)
        # print("Processed image no. {}".format(k))

        self._save_result(k, indices, processed_img)
        # print("Saved result of processing image no. {}\n".format(k))

    def _load_image(self, k):
        suffix = '.png'
        filepath = os.path.join(self.input_path, str(k) + suffix)

        img = cv.imread(filepath)

        return img

    @staticmethod
    def _process_image(img):

        ind, img = OCR().process_image(img)

        return ind, img

    def _save_result(self, k, indices, processed_img):

        img_filepath = os.path.join(self.output_path, str(k) + '-wyrazy.png')
        ind_filepath = os.path.join(self.output_path, str(k) + '-indeksy.txt')

        cv.imwrite(img_filepath, processed_img)

        with open(ind_filepath, "w") as f:
            if len(indices) > 0:
                f.write("\n".join([x for x in indices if x != ""]))
            else:
                f.write("")

    def run(self):
        if len(sys.argv) != 4:
            print("Invalid number of args! Abort.")
            exit()
        _, self.input_path, self.n, self.output_path = sys.argv

        self.n = int(self.n)

        for k in range(self.n):

            self._process(k)

        # print("Done!")


if __name__ == "__main__":
    PiRO2().run()
