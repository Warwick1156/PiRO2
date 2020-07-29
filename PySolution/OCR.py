from Preprocessor import *
from BoundingBoxSplitter import *
from MNISTKeras import *

from WordsExtractor import *
from DigitsExtractor import *

class OCR:

    def __init__(self):
        self.words_detector = None
        self.digits_extractor = None

    def process_image(self, image):

        print("Preprocessing image...")
        image = Preprocessor.process(image)

        print("Splitting rows...")
        rows, image_orig = BoundingBoxSplitter.split_rows(image)
        print("Detected " + str(len(rows)) + " rows.")


        clf = Classifier("../model/keras_mnist.h5")

        print("Processing rows...")
        indices = []
        test = []
        for row in rows:
            words  = WordsExtractor.extract(row)
            digits = DigitsExtractor.extract(words)

            for w in words:
                test.append(w)

            index = []
            for digit in digits:
                predicted = clf.predict(digit)
                index.append(predicted)

            indices.append(index)

        dummy_image = np.zeros((64, 64))

        indices = ["123"]

        return indices, digits
