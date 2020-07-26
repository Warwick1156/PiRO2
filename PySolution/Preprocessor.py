from PIL import Image
import cv2 as cv
import numpy as np


class Preprocessor:

    @staticmethod
    def denoise(img, strength=3, output=None, templateWindowSize=7, searchWindowSize=21):
        return cv.fastNlMeansDenoising(img, output, strength, templateWindowSize, searchWindowSize)

    @staticmethod
    def dilate(img, kernel_size=2, kernel=None):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)) if kernel is None else kernel
        return cv.dilate(img, kernel)

    @staticmethod
    def erode(img, kernel_size=2, kernel=None):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)) if kernel is None else kernel
        return cv.erode(img, kernel)

    @staticmethod
    def to_binary(img, block_size=11, constant=2):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray, (3, 3), 0)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant)
        return binary

    @staticmethod
    def set_300_dpi(img=None, file_path=None, save=True, name='default.png', dpi=300):
        if img is not None:
            # img has to be PIL format
            image = img
        else:
            image = Image.open(file_path)

        #     length_x, width_y = image.size

        # upscales to 300dpi
        size = (6400, 10667)
        im_resized = image.resize(size, Image.ANTIALIAS)

        if save:
            im_resized.save(name, format='png', dpi=(300, 300))

        return im_resized

    @staticmethod
    def remove_lines(img):
        gray = img.copy()

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (96, 1))
        h_lines = 255 - cv.filter2D(gray, 0, horizontal_kernel)

        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 96))
        v_lines = 255 - cv.filter2D(gray, 0, vertical_kernel)

        mask1 = v_lines + h_lines
        mask1 = Preprocessor.dilate(mask1, 4)

        gray = np.invert(gray)
        # gray = Preprocessor.erode(Preprocessor.dilate(gray, 4), 4)
        gray = gray - mask1
        gray = np.invert(gray)
        gray = Preprocessor.dilate(gray, 4)
        gray = Preprocessor.erode(gray, 4)

        gray = Preprocessor.erode(gray, 4)
        gray = Preprocessor.dilate(gray, 4)
        return gray

    @staticmethod
    def process(img_pil):
        print("Setting 300 dpi...")
        _300dpi_img = Preprocessor.set_300_dpi(img=img_pil, file_path=None, save=False, name=None)

        print("Denoising... (skipped)")
        opencv_img = np.array(_300dpi_img.convert('RGB'))
        denoised_img = opencv_img
        # denoised_img = Preprocessor.denoise(opencv_img)
        image = denoised_img

        print("Creating mask...")
        mask = np.logical_or(np.logical_and(image[:, :, 0] > 90, image[:, :, 1] >= 84, image[:, :, 2] > 106), image[:, :, 2] > 80)
        image2 = image.copy()
        image2[mask] = 0
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        _, image2 = cv.threshold(image2, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        image2 = np.invert(image2)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1536,64))
        # image2 = Preprocessor.dilate(image2, kernel=kernel)
        image2 = Preprocessor.dilate(image2, 64)

        print("Binarizing...")
        image = Preprocessor.to_binary(image, 11, 3)
        image = Preprocessor.dilate(Preprocessor.erode(image, 8), 8)

        print("Removing lines...")
        lines_removed_img = Preprocessor.remove_lines(image)
        lines_removed_img = Preprocessor.erode(lines_removed_img, 16)
        lines_removed_img = np.invert(lines_removed_img)

        image22 = np.where(image2 > 0, 1, 0)
        lines_removed_img2 = np.where(lines_removed_img > 0, 1, 0)
        res = lines_removed_img2 * image22
        res = np.where(res > 0, 255, 0).astype(int)
        res = np.array(res).astype('uint8')
        res = Preprocessor.dilate(res, 12)
        res = Preprocessor.erode(res, 8)

        return [image, image2, lines_removed_img, res]

