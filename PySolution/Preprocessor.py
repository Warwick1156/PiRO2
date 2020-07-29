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
    def to_binary(img, block_size=11, constant=2, cast_color=True, otsu=False):
        if cast_color:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv.GaussianBlur(gray, (3, 3), 7)
        if otsu:
            ret3, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        else:
            binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant)

        return binary

    @staticmethod
    def color_based_to_binary(image, block_size=11, constant=2):
        # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        mask = np.logical_or(np.logical_and(image[:, :, 0] > 80, image[:, :, 1] >= 74, image[:, :, 2] > 96), image[:, :, 2] > 70)
        image2 = image.copy()
        image2[mask] = 0

        r = np.where(image2 > 10, 255, 0)

        return r


        r = img.copy()
        r = r[:, :, 2].astype('uint8')

        # g = Preprocessor.to_binary(img[:,:,1], block_size, constant)
        # b = Preprocessor.to_binary(img[:,:,2], block_size, constant)

        return r
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

        gray_out = gray.copy()
        gray_out = np.invert(gray_out)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 1))
        h_lines = cv.filter2D(gray, -1, horizontal_kernel)

        mask = h_lines
        mask = np.invert(mask)
        mask = Preprocessor.dilate(mask, 2)
        mask = Preprocessor.erode(mask, 2)
        mask = cv.GaussianBlur(mask, (1, 1), 1)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (32, 1))
        mask = Preprocessor.dilate(mask, kernel=kernel)
        mask = Preprocessor.erode(mask, kernel=kernel)

        mask = np.roll(mask, -1, axis=0)

        gray_out = gray_out - mask

        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 8))
        v_lines = cv.filter2D(gray, -1, vertical_kernel)

        mask = v_lines
        mask = np.invert(mask)
        mask = Preprocessor.dilate(mask, 2)
        mask = Preprocessor.erode(mask, 2)
        mask = cv.GaussianBlur(mask, (1, 1), 1)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 32))
        mask = Preprocessor.dilate(mask, kernel=kernel)
        mask = Preprocessor.erode(mask, kernel=kernel)

        mask = np.roll(mask, -1, axis=1)

        gray_out = gray_out - mask

        gray_out = np.invert(gray_out)
        gray_out = Preprocessor.dilate(gray_out, 1)
        return gray_out

    @staticmethod
    def process(img_pil):
        # print("Setting 300 dpi...")
        # _300dpi_img = Preprocessor.set_300_dpi(img=img_pil, file_path=None, save=False, name=None)
        _300dpi_img = img_pil

        # print("Denoising... (skipped)")
        opencv_img = np.array(_300dpi_img.convert('RGB'))
        denoised_img = opencv_img
        # denoised_img = Preprocessor.denoise(opencv_img)
        image = denoised_img
        # image = image[48:2265, 80:1225, :]
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # print("Creating mask...")
        # mask = np.logical_or(np.logical_and(image[:, :, 0] > 90, image[:, :, 1] >= 84, image[:, :, 2] > 106), image[:, :, 2] > 80)
        # image2 = image.copy()
        # image2[mask] = 0
        # image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        # _, image2 = cv.threshold(image2, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        # image2 = np.invert(image2)
        # # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1536,64))
        # # image2 = Preprocessor.dilate(image2, kernel=kernel)
        # image2 = Preprocessor.dilate(image2, 64)

        print("Binarizing...")

        # return [Preprocessor.color_based_to_binary(image, 51, 11)]


        image = Preprocessor.to_binary(image, 51, 11, otsu=False)

        image = Preprocessor.dilate(image, 2)
        image = Preprocessor.erode(image, 2)

        image = Preprocessor.erode(image, 2)
        image = Preprocessor.dilate(image, 2)

        print("Removing lines...")
        lines_removed_img = Preprocessor.remove_lines(image)
        # lines_removed_img = Preprocessor.to_binary(lines_removed_img, 51, 11, cast_color=False)
        # lines_removed_img = image
        # lines_removed_img = np.invert(lines_removed_img)
        return lines_removed_img

        # image22 = np.where(image2 > 0, 1, 0)
        # lines_removed_img2 = np.where(lines_removed_img > 0, 1, 0)
        # res = lines_removed_img2 * image22
        # res = np.where(res > 0, 255, 0).astype(int)
        # res = np.array(res).astype('uint8')
        # res = Preprocessor.dilate(res, 12)
        # res = Preprocessor.erode(res, 8)

        return [res]

