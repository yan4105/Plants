from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize
import torch
from utils_GetSegmentationResult import get_mask_for_classes

class ImageManager:
    lower_green = np.array([40, 0, 0])
    upper_green = np.array([80, 255, 255])
    binaryTHRESH = 0

    @staticmethod
    def get_GT(rgbPath):
        """Other files call this function to get GT"""
        res = ImageManager.segment_green_objects(rgbPath)
        segment_mask = ImageManager.segment_plants(rgbPath, "tree", "grass")
        res = ImageManager.mergeGT(res, segment_mask)
        return res

    @staticmethod
    def segment_green_objects(rgbPath):
        rgb = io.imread(rgbPath)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, ImageManager.lower_green, ImageManager.upper_green)
        res = cv2.bitwise_and(rgb, rgb, mask=mask)
        res = rgb2gray(res)
        res = res > ImageManager.binaryTHRESH
        return res

    @staticmethod
    def segment_plants(rgbPath, *args):
        return get_mask_for_classes(rgbPath, args)

    @staticmethod
    def mergeGT(mask1, mask2):
        assert len(mask1) == len(mask2) and len(mask1[0]) == len(mask2[0])
        dim0 = len(mask1)
        dim1 = len(mask1[0])
        return [[mask1[i][j] and mask2[i][j] for j in range(dim1)] for i in range(dim0)]

    @staticmethod
    def is_healthy(ndvi):
        return 0.3 < ndvi < 0.8

    @staticmethod
    def get_model_output_as_bw_image(model, rgb, nir):
        nir = resize(nir, rgb.shape) * 255
        assert rgb.shape == nir.shape
        result = np.zeros((rgb.shape[0], rgb.shape[1]))
        foreground = ImageManager.get_GT(rgb)
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if not foreground[i][j]:
                    result[i][j] = False
                    continue
                rgb_pixels = rgb[i][j]
                nir_pixels = nir[i][j]
                input = np.concatenate([rgb_pixels, nir_pixels])
                input = [input]
                output = model(torch.FloatTensor(input))
                result[i][j] = ImageManager.is_healthy(output)
        return result

    @staticmethod
    def reshape(rgb, nir):
        if rgb.shape != nir.shape:
            nir = resize(nir, rgb.shape) * 255
            assert rgb.shape == nir.shape
        return rgb, nir

    @staticmethod
    def get_input_sample(rgbPixel, nirPixel, is_positive):
        X = np.concatenate((np.array(rgbPixel), np.array(nirPixel)), axis=0)
        y = 0
        if is_positive:
            y = 1
        y = np.array([y])
        X = np.concatenate((X, y), axis=0)
        return X

    @staticmethod
    def get_model_output_as_rgb_image(model, path):
        rgb = io.imread(path + 'RGB.jpg')
        nir = io.imread(path + 'NIR.jpg')
        ImageManager.reshape(rgb, nir)
        yellow = np.array([255, 255, 0])
        green = np.array([0, 128, 0])
        result = np.zeros((rgb.shape[0], rgb.shape[1], 3))
        foreground = imageManager.segment_green_objects(rgb)
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if not foreground[i][j]:
                    result[i][j] = rgb[i][j]
                    continue
                rgb_pixels = rgb[i][j]
                nir_pixels = nir[i][j]
                input = np.concatenate([rgb_pixels, nir_pixels])
                input = [input]
                ndvi_output = model(torch.FloatTensor(input))
                if ImageManager.is_healthy(ndvi_output):
                    result[i][j] = green
                else:
                    result[i][j] = yellow
        print('result generated')
        plt.imshow(result.astype(np.uint8))
        plt.show()
        return result


if __name__ == '__main__':
    imageManager = ImageManager()
    inputImage = io.imread('./dataset/positive/1/RGB.jpg')
    GT = imageManager.segment_green_objects(inputImage)
    io.imshow(GT)
    plt.show()