from Preprocessor import *
from BoundingBoxSplitter import *
from MNISTClassifier import *

from WordsExtractor import *
from DigitsExtractor import *

class OCR:

    def __init__(self):
        self.words_detector = None
        self.digits_extractor = None
        self.mnist_classifier = MNISTClassifier()

    def process_image(self, image):

        print("Preprocessing image...")
        image = Preprocessor.process(image)

        print("Splitting rows...")
        rows = BoundingBoxSplitter.split_rows(image)
        print("Detected " + str(len(rows)) + " rows.")

        clf = MNISTClassifier()
        clf.from_file("../model/mnist_model.pt")

        print("Processing rows...")
        indices = []
        for row in rows:
            words  = WordsExtractor.extract(row)
            digits = DigitsExtractor.extract(words)

            index = []
            for digit in digits:
                predicted = clf.predict(digit)
                index.append(predicted)

            indices.append(index)

        dummy_image = np.zeros((64, 64))
        processed_image = dummy_image

        return indices, processed_image
