import cv2 as cv
import numpy as np

from skimage.morphology import erosion, dilation, opening, closing, square, disk, star

from Preprocessor import Preprocessor

if __name__ == '__main__':
    img = cv.imread('../data/17.png')
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

    horizontal = np.copy(img)
    vertical = np.copy(img)

    rows = vertical.shape[0]
    vertical_size = rows // 30
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    vertical = cv.erode(vertical, vertical_structure)
    vertical = cv.dilate(vertical, vertical_structure)
    # cv.imwrite('../out/out6.png', vertical)

    cols = horizontal.shape[1]
    horizontal_size = cols //50
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontal_structure)
    horizontal = cv.dilate(horizontal, horizontal_structure)
    # cv.imwrite('../out/out7.png', horizontal)


    vertical_mask = vertical > 0
    horizontal_mask = horizontal > 0
    image = img.copy()
    image[vertical_mask] = 0
    image[horizontal_mask] = 0
    # cv.imwrite('../out/out8.png', image)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    image = cv.erode(image, kernel)
    # cv.imwrite('../out/out9.png', image)

    # image = cv.dilate(image, kernel)
    # image = image[48:2265, 40:1225]
    # # cv.imwrite('../out/out10.png', image)
    #
    # mask1D = np.zeros([5, 5])
    # mask1D[2, :] = 1
    # image = closing(image, mask1D)
    # # cv.imwrite('../out/out11.png', image)
    #
    # mask1D = np.zeros([5, 5])
    # mask1D[:, 2] = 1
    # image = closing(image, mask1D)
    # cv.imwrite('../out/out12.png', image)

    image = image[48:2265, 40:1265]
    mask1D = np.zeros([7, 7])
    mask1D[3, :] = 1
    image = closing(image, mask1D)
    mask1D = np.zeros([7, 7])
    mask1D[:, 3] = 1
    image = closing(image, mask1D)
    # cv.imwrite('../out/out13.png', image)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
    image = cv.erode(image, kernel)
    # cv.imwrite('../out/out15.png', image)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    image = cv.dilate(image, kernel)
    # cv.imwrite('../out/out16.png', image)

    image = cv.blur(image, (4, 4))
    mask1D = np.zeros([7, 7])
    mask1D[:, 3] = 1
    image = closing(image, mask1D)
    cv.imwrite('../out/out17.png', image)

    mask1D = np.zeros([7, 7])
    mask1D[3, :] = 1
    image = closing(image, mask1D)
    cv.imwrite('../out/out18.png', image)


