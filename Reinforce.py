from skimage import io
from skimage.transform import resize
from ImageManager import ImageManager
import cv2
import numpy as np
from MyDataSet import MyDataset
from torch.utils.data import DataLoader
from Model1 import train_model, Model
import torch
from CustomLayers import NDVILoss

class Reinforcer():
    def __init__(self, model=Model(), load_path = './trained_models/model1_10epochs.pt', save_path = './trained_models/reinforced.pt'):
        self.model = model
        self.POSITIVE_THRESH = 0.5
        self.load_path = load_path
        self.save_path = save_path

    def __get_ground_truth_area(self, rgb, nir):
        GT = ImageManager.get_GT(rgb)
        return cv2.countNonZero(GT)

    def __get_positive_area(self, rgb, nir):
        BW = ImageManager.get_model_output_as_bw_image(self.model, rgb, nir)
        return cv2.countNonZero(BW)

    def __get_correct_percentage(self, rgb, nir):
        GT_area = self.__get_ground_truth_area(rgb, nir)
        positive_area = self.__get_positive_area(rgb, nir)
        return positive_area / GT_area

    def reinforce(self, rgb, nir, is_positive):
        ImageManager.reshape(rgb, nir)
        GT = ImageManager.get_GT(rgb)
        prediction = ImageManager.get_model_output_as_bw_image(self.model, rgb, nir)
        wrong_predictions = []
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if GT[i][j] and prediction[i][j] != is_positive:
                   input_sample = ImageManager.get_input_sample(rgb[i][j], nir[i][j], is_positive)
                   wrong_predictions.append(input_sample)
        dataset = MyDataset(data=wrong_predictions)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        self.model.load_state_dict(torch.load(self.load_path))
        criterion = NDVILoss.apply
        train_model(self.model, dataloader, criterion, max_epochs=1)
        torch.save(self.model.state_dict(), self.save_path)

if __name__ == '__main__':
    r = Reinforcer()
    path = './dataset/negative/9'
    rgb = io.imread(path + '/RGB.jpg')
    nir = io.imread(path + '/NIR.jpg')
    r.reinforce(rgb, nir, False)
    ImageManager.get_model_output_as_bw_image(r.model, rgb, nir)









