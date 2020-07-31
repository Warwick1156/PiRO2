from Preprocessor import *
from BoundingBoxSplitter import *
from MNISTKeras import *

from WordsExtractor import *
from DigitsExtractor import *

class OCR:

    def __init__(self):
        pass

    def process_image(self, image):

        # print("Preprocessing image...")
        image = Preprocessor.process(image)

        out_image = np.zeros(image.shape)

        # print("Splitting rows...")
        rows, processed = BoundingBoxSplitter.split_rows(image)
        # print("Detected " + str(len(rows)) + " rows.")

        clf = Classifier("./keras_piro.h5")

        # print("Processing rows...")
        indices = []
        # test = []
        row_no = 1
        for row, coords, _ in rows:

            words, out_image, row_no  = WordsExtractor.extract(row, coords, row_no, out_image)
            if len(words) > 0:
                # test.append(words[-1])
                digits = DigitsExtractor.extract(words)
            else:
                digits = []

            index = ""
            for digit in digits:
                # test.append(digit)
                predicted = clf.predict(digit)
                index += (str(predicted))

            indices.append(index)

        out_image = Preprocessor.make_out_image(out_image)
        # test.append(out_image)
        return indices, out_image
