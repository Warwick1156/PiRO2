from Preprocessor import *
from RowSplitter import *
from MNISTClassifier import *

class OCR:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.row_splitter = RowSplitter()
        self.words_detector = None
        self.digits_extractor = None
        self.mnist_classifier = MNISTClassifier()

    def process_image(self, image):

        # self.preprocessor
        # ...

        # self.row_splitter
        # ...

        # self.

        # self.mnist_classifier
        # ...

        ind = ["00000", "11111", "22222"]
        processed_image = image

        return ind, processed_image