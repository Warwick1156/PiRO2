def erode(img, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv.erode(img, kernel)

def dilate(img, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv.dilate(img, kernel)

def remove_lines(img):
    gray = img.copy()
    
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (96,1))
    h_lines = 255 - cv.filter2D(gray, 0, horizontal_kernel)
    
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,96))
    v_lines = 255 - cv.filter2D(gray, 0, vertical_kernel)

    mask1 = v_lines + h_lines
    mask1 = dilate(mask1, 2)
    
    gray = erode(dilate(np.invert(gray), 4), 4)
    gray = gray - mask1
    gray = np.invert(gray)
    gray = dilate(gray, 8)
    gray = erode(gray, 20)
    gray = dilate(gray, 10)
    return gray

bina = to_binary(img, 11, 3)
bina = dilate(erode(bina, 12),8)

bina = bina[1500:5000, 500:3500]
debug_plot_cv_img(bina, cmap='gray')

img2 = remove_lines(bina)
debug_plot_cv_img(img2, cmap='gray')
