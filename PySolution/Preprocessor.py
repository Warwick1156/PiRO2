from PIL import Image
import cv2 as cv
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, square, disk, star


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
    def process(img):
        # # TEST
        # img = cv.imread('../data/17.png')
        #
        #
        # img = Preprocessor.denoise(img)
        # img = cv.GaussianBlur(img, (1, 11), 5, )
        # img = Preprocessor.to_binary(img, 51, 11, otsu=False)
        # # img = cv.bitwise_not(img)
        #
        # horizontal = np.copy(img)
        # vertical = np.copy(img)
        #
        # rows = vertical.shape[0]
        # vertical_size = rows // 50
        # vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        # vertical = cv.erode(vertical, vertical_structure)
        # vertical = cv.dilate(vertical, vertical_structure)
        #
        # cols = horizontal.shape[1]
        # horizontal_size = cols // 50
        # horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        # horizontal = cv.erode(horizontal, horizontal_structure)
        # horizontal = cv.dilate(horizontal, horizontal_structure)
        #
        # vertical_mask = vertical > 0
        # horizontal_mask = horizontal > 0
        # image = img.copy()
        #
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
        # vertical = cv.dilate(vertical, kernel)
        #
        # image[vertical_mask] = 0
        # image[horizontal_mask] = 0
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        # image = cv.erode(image, kernel)
        #
        # mask = np.zeros([5, 5])
        # mask[2, :] = 1
        # image = closing(image, mask)
        #
        # mask = np.zeros([7, 7])
        # mask[:, 3] = 1
        # image = closing(image, mask)




        # img = cv.imread('../data/17.png')
        img = Preprocessor.denoise(img)
        img = cv.GaussianBlur(img, (1, 11), 5, )
        # img = cv.GaussianBlur(img, (11, 1), 0, 5)
        # cv.imwrite('../out/out.png', img)
        # img = cv.bitwise_not(img)
        img = Preprocessor.to_binary(img, 51, 11, otsu=False)
        # cv.imwrite('../out/out.png', img)
        img = cv.bitwise_not(img)

        # img = cv.GaussianBlur(img, (1, 51), 10, 0)
        # cv.imwrite('../out/out5.png', img)

        # horizontal = np.copy(img)
        # vertical = np.copy(img)
        #
        # rows = vertical.shape[0]
        # vertical_size = rows // 30
        # vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        # vertical = cv.erode(vertical, vertical_structure)
        # vertical = cv.dilate(vertical, vertical_structure)
        # # cv.imwrite('../out/out6.png', vertical)
        #
        # cols = horizontal.shape[1]
        # horizontal_size = cols //50
        # horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        # horizontal = cv.erode(horizontal, horizontal_structure)
        # horizontal = cv.dilate(horizontal, horizontal_structure)
        # # cv.imwrite('../out/out7.png', horizontal)
        #
        #
        # vertical_mask = vertical > 0
        # horizontal_mask = horizontal > 0
        # image = img.copy()
        # image[vertical_mask] = 0
        # image[horizontal_mask] = 0
        # # cv.imwrite('../out/out8.png', image)
        #
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        # image = cv.erode(image, kernel)
        # # cv.imwrite('../out/out9.png', image)
        #
        # # image = cv.dilate(image, kernel)
        # # image = image[48:2265, 40:1225]
        # # # cv.imwrite('../out/out10.png', image)
        # #
        # # mask1D = np.zeros([5, 5])
        # # mask1D[2, :] = 1
        # # image = closing(image, mask1D)
        # # # cv.imwrite('../out/out11.png', image)
        # #
        # # mask1D = np.zeros([5, 5])
        # # mask1D[:, 2] = 1
        # # image = closing(image, mask1D)
        # # cv.imwrite('../out/out12.png', image)
        #
        # image = image[48:2265, 40:1265]
        # mask1D = np.zeros([7, 7])
        # mask1D[3, :] = 1
        # image = closing(image, mask1D)
        # mask1D = np.zeros([7, 7])
        # mask1D[:, 3] = 1
        # image = closing(image, mask1D)
        # # cv.imwrite('../out/out13.png', image)
        #
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
        # image = cv.erode(image, kernel)
        # # cv.imwrite('../out/out15.png', image)
        #
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # image = cv.dilate(image, kernel)
        # # cv.imwrite('../out/out16.png', image)
        #
        # image = cv.blur(image, (4, 4))
        # mask1D = np.zeros([7, 7])
        # mask1D[:, 3] = 1
        # image = closing(image, mask1D)
        # # cv.imwrite('../out/out17.png', image)
        #
        # mask1D = np.zeros([7, 7])
        # mask1D[3, :] = 1
        # image = closing(image, mask1D)
        # # cv.imwrite('../out/out18.png', image)

        # ---------------------------------------------------------------------------------------------------

        horizontal = np.copy(img)
        vertical = np.copy(img)

        rows = vertical.shape[0]
        vertical_size = rows // 50
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        vertical = cv.erode(vertical, vertical_structure)
        vertical = cv.dilate(vertical, vertical_structure)

        cols = horizontal.shape[1]
        horizontal_size = cols // 50
        horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv.erode(horizontal, horizontal_structure)
        horizontal = cv.dilate(horizontal, horizontal_structure)

        vertical_mask = vertical > 0
        horizontal_mask = horizontal > 0
        image = img.copy()

        # cv.imwrite('../out/out19.png', vertical)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
        # vertical = cv.erode(vertical, kernel)
        vertical = cv.dilate(vertical, kernel)
        # cv.imwrite('../out/out20.png', vertical)

        image[vertical_mask] = 0
        image[horizontal_mask] = 0
        # cv.imwrite('../out/out21.png', image)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        image = cv.erode(image, kernel)
        # cv.imwrite('../out/out22.png', image)

        mask1D = np.zeros([5, 5])
        mask1D[2, :] = 1
        image = closing(image, mask1D)
        # cv.imwrite('../out/out23.png', image)

        mask1D = np.zeros([7, 7])
        mask1D[:, 3] = 1
        image = closing(image, mask1D)
        # cv.imwrite('../out/out24.png', image)

        # mask1D = np.array([
        #     [1, 1, 0, 0, 0, 1, 1],
        #     [1, 1, 0, 0, 0, 1, 1],
        #     [1, 1, 0, 0, 0, 1, 1],
        # ])
        # # mask1D = np.array([[1, 1], [1, 1]])
        # mask1D = np.array([[1,1]])
        # # mask1D[:, 3] = 1
        # image = erosion(image, mask1D)
        # # cv.imwrite('../out/out25.png', image)
        #
        # mask = np.zeros((7, 7))
        # mask[:, 3] = 1
        # mask[3, :] = 1
        # image = dilation(image, mask)
        # # cv.imwrite('../out/out26.png', image)
        #
        # mask = np.zeros((9, 9))
        # mask[:, 4] = 1
        # mask[4, :] = 1
        # image = closing(image, mask)
        # # cv.imwrite('../out/out28.png', image)
        #
        #

        image = cv.bitwise_not(image)
        return image

