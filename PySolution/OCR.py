from Preprocessor import *
from RowSplitter import *
from MNISTClassifier import *

class OCR:

    def __init__(self):
        self.row_splitter = RowSplitter()
        self.words_detector = None
        self.digits_extractor = None
        self.mnist_classifier = MNISTClassifier()

    def process_image(self, image):

        print("Preprocessing image...")
        image = Preprocessor.process(image)
        print("Preprocessing done!")

        # self.row_splitter
        # ...

        # self.

        # self.mnist_classifier
        # ...

        ind = ["00000", "11111", "22222"]

        return ind, image