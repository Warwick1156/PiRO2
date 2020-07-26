from Preprocessor import *
from BoundingBoxSplitter import *
from MNISTClassifier import *

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

        # self.mnist_classifier
        # ...

        ind = ["00000", "11111", "22222"]

        return ind, rows