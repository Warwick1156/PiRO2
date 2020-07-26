from Preprocessor import *
from RowSplitter import *
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
        # image = Preprocessor.dilate(image, 3)
        image = RowSplitter.split_rows(image, threshold=4, show_rows=True, offset=150)

        # self.mnist_classifier
        # ...

        ind = ["00000", "11111", "22222"]

        return ind, image