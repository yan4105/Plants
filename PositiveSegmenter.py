from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.morphology import reconstruction
from matplotlib import pyplot as plt
import numpy as np

RGB_HIGH_THRESH = 150
RGB_LOW_THRESH = 0
NIR_THRESH = 25 # 25 is good

def binary_op_for_final_image(isroot, isgreen):
    pass

def binary_op_for_root(bool1, bool2):
    if bool1 & bool2:
        return True
    return False

def custom_mix(img1, img2, op):
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    img3 = np.copy(img2)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            img3[i][j] = op(img1[i][j], img2[i][j])
    return img3

def get_root_binary(img1, img2):
    """
    :param img1: rgb image
    :param img2: nir image
    :return: a binary image with leaves white and roots black
    other areas undefined
    """
    nir_grayscale = rgb2gray(nir)
    rgb_grayscale = rgb2gray(rgb)
    rgb_binary = (rgb_grayscale * 255 < RGB_HIGH_THRESH) & (rgb_grayscale * 255 > RGB_LOW_THRESH)
    nir_binary = nir_grayscale * 255 < NIR_THRESH
    return custom_mix(rgb_binary, nir_binary, binary_op_for_root)

def get_hsv_green_range(epsilon):
    green = np.uint8([0,255,0])
    hsv_green = rgb2hsv(green)
    hsv_green = hsv_green[0]
    return hsv_green - epsilon/2, hsv_green + epsilon/2

def filter_for_green(epsilon, image):
    """
    :param epsilon: green range
    :param image: image to process
    :return: an image with non-green pixels black
    """
    hsv_black = rgb2hsv(np.uint([1,1,1]))
    hsv_white = rgb2hsv(np.uint([0,0,0]))
    lower, upper = get_hsv_green_range(epsilon)
    hsv_image = rgb2hsv(image)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            pixel = hsv_image[i][j]
            if lower < pixel[0] < upper:
                pass
            else:
                hsv_image[i][j] = hsv_black
    image = hsv2rgb(hsv_image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] > 1e-10:
                image[i][j] = np.uint([255,255,255])
            else:
                image[i][j] = np.uint([0,0,0])
    return rgb2gray(image) > 0

def produce_final_segmented_image(rgb, nir):
    root_binary = get_root_binary(rgb, nir)
    green_filtered = filter_for_green(0.1, rgb)



if __name__ == "__main__":
    nir = io.imread('dataset/NIR2.jpg')
    rgb = io.imread('dataset/RGB2.jpg')
    #nir_grayscale = rgb2gray(nir)
    rgb_grayscale = rgb2gray(rgb)
    rgb_binary = (rgb_grayscale * 255 < RGB_HIGH_THRESH) & (rgb_grayscale * 255 > RGB_LOW_THRESH)
    #nir_binary = nir_grayscale * 255 < NIR_THRESH
    #io.imshow(nir_grayscale)
    #io.imshow(rgb_grayscale)
    #io.imshow(rgb_binary)
    #io.imshow(nir_binary)
    #mixed = rgb_binary & nir_binary
    # io.imshow(mixed)

    #seed = np.copy(mixed)
    #seed[1:-1, 1:-1] = mixed.max()
    #mask = mixed

    #filled = reconstruction(seed, mask, method='erosion')
    # io.imshow(filled)

    #custom_mixed = custom_mix(rgb_binary, nir_binary, binary_op_for_root)
    #io.imshow(custom_mixed)
    #plt.show()

    #custom_mix(rgb_binary, nir_binary)

    #rgb_hsv = rgb2hsv(rgb)

    #io.imshow(rgb_binary)
    green_filtered = filter_for_green(0.1, rgb)
    io.imshow(green_filtered)
    plt.show()

