from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
from skimage import io
import os
from ImageManager import ImageManager
import random

class MyDataset(Dataset):
    def __init__(self, balanced=True, data=[], p='./dataset/positive', n='./dataset/negative', update_GT=False):
        super(Dataset, self).__init__()
        self.RGB_HIGH_THRESH = 150
        self.RGB_LOW_THRESH = 0
        self.ground_truth_img_name = "GT.jpg"
        self.data = data
        self.positiveData = []
        self.negativeData = []
        self.positive_data_path = p
        self.negative_data_path = n
        self.imageManager = ImageManager()
        self.positivePixelCount = 0
        self.negativePixelCount = 0
        self.isBalanced = balanced
        self.update_GT = update_GT
        self.add_data()

    def add_data(self):
        if not self.data:
            for rgbPath, nirPath, subfoler_path in self._get_iter(self.positive_data_path):
                self._add_data(rgbPath, nirPath, True, subfoler_path, update_GT=self.update_GT)
            for rgbPath, nirPath, subfoler_path in self._get_iter(self.negative_data_path):
                self._add_data(rgbPath, nirPath, False, subfoler_path, update_GT=self.update_GT)
            if self.isBalanced:
                self._balance()
            self._merge()

    def _get_iter(self, root_path):
        for subfolder in os.listdir(root_path):
            if subfolder[0] == '.':  # .DS_Store
                continue
            subfolder_path = root_path + '/' + subfolder
            rgb_path, nir_path = None, None
            for filename in os.listdir(subfolder_path):
                if filename[0] == 'R':
                    rgb_path = subfolder_path + '/' + filename
                elif filename[0] == 'N':
                    nir_path = subfolder_path + '/' + filename
            yield rgb_path, nir_path, subfolder_path

    def _prepare_plant_mask(self, rgb_path, nir_path, root_path, update_GT=False):
        ground_truth_image_path = root_path + "/" + self.ground_truth_img_name
        if not update_GT and self.ground_truth_img_name in os.listdir(root_path):
            return io.imread(ground_truth_image_path)
        else:
            plant_mask = self._compute_ground_truth(rgb_path, nir_path)
            plant_mask = np.asanyarray(plant_mask)
            io.imsave(ground_truth_image_path, plant_mask)
            return plant_mask

    def _add_data(self, rgb_path, nir_path, is_positive, root_path, update_GT=False):
        plant_mask = self._prepare_plant_mask(rgb_path, nir_path, root_path, update_GT)
        rgb = io.imread(rgb_path)
        nir = io.imread(nir_path)
        if rgb.shape != nir.shape:
            nir = resize(nir, rgb.shape) * 255
            io.imsave(nir_path, nir)
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if not plant_mask[i][j]:
                    continue
                X = np.concatenate((np.array(rgb[i][j]), np.array(nir[i][j])), axis=0)
                y = self._get_label(plant_mask[i][j], is_positive)
                y = np.array([y])
                X = np.concatenate((X, y), axis=0)
                if is_positive:
                    self.positivePixelCount += 1
                    self.positiveData.append(X)
                else:
                    self.negativePixelCount += 1
                    self.negativeData.append(X)

    def _balance(self):
        if self.positivePixelCount == self.negativePixelCount:
            return
        if self.positivePixelCount > self.negativePixelCount:
            self.positiveData = random.sample(self.positiveData, self.negativePixelCount)
        else:
            self.negativeData = random.sample(self.negativeData, self.positivePixelCount)

    def _merge(self):
        self.data = np.concatenate((self.positiveData, self.negativeData))

    def _get_label(self, ground_truth_label, is_positive):
        assert ground_truth_label, "do not add background into dataset"
        if is_positive:
            return 1
        return 0

    def _compute_ground_truth(self, rgb_path, _):
        return self.imageManager.get_GT(rgb_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index][0:6], self.data[index][6]
        return X, y

if __name__ == '__main__':
    dataset = MyDataset()