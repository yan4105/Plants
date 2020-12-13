import os
import shutil
from skimage import io
from ImageManager import ImageManager

class MyDataSetFolder:
    def __init__(self, root='./dataset'):
        self.root = root
        self.rgbFileName = 'RGB.jpg'
        self.nirFileName = 'NIR.jpg'
        self.GTFileName = 'GT.jpg'
        self.imageManager = ImageManager()

    @property
    def positivepath(self):
        return self.root + '/' + 'positive'

    @property
    def negativepath(self):
        return self.root + '/' + 'negative'

    def __get_folder_path(self, isPositive):
        if isPositive:
            return self.positivepath
        return self.negativepath

    def __get_total_number_samples(self, isPositive):
        folderPath = self.__get_folder_path(isPositive)
        return len(os.listdir(folderPath))

    def __len__(self):
        return len(os.listdir(self.positivepath)) + len(os.listdir(self.negativepath)) - 2  # discount .DS_Store

    def relabel(self):
        self.__relabel_folder(self.positivepath)
        self.__relabel_folder(self.negativepath)

    def delete(self, ispositive, number):
        sampleFolderPath = self.__get_folder_path(ispositive)
        self.__delete_sample_from_folder(sampleFolderPath, number)

    def add(self, ispositive, rgb, nir):
        # self.relabel()
        sampleFolderPath = self.__get_folder_path(ispositive) + '/' + str(len(self.__get_total_number_samples(ispositive)))
        os.mkdir(sampleFolderPath)
        rgbPath = sampleFolderPath + '/' + self.rgbFileName
        nirPath = sampleFolderPath + '/' + self.nirFileName
        io.imsave(rgbPath, rgb)
        io.imsave(nirPath, nir)

    def update_ground_truth(self):
        for folderPath in self.iter_positive_folder_path():
            rgbPath = folderPath + '/' + self.rgbFileName
            groundTruthPath = folderPath + '/' + self.GTFileName
            rgb = io.imread(rgbPath)
            groundTruth = self.imageManager.segment_green_objects(rgb)
            io.imsave(groundTruthPath, groundTruth)

    def iter_positive_folder_path(self):
        for folder in os.listdir(self.positivepath):
            if folder[0] == '.':
                continue
            folderPath = self.positivepath + '/' + folder
            yield folderPath

    def iter_positive_rgb_nir(self):
        for positiveSamplePath in self.iter_positive_folder_path():
            rgbPath = positiveSamplePath + '/' + self.rgbFileName
            nirPath = positiveSamplePath + '/' + self.nirFileName
            rgb = io.imread(rgbPath)
            nir = io.imread(nirPath)
            yield rgb, nir

    def addFromSDCopy(self, SDFolderPath):
        pass

    def __delete_sample_from_folder(self, sampleFolderPath, number):
        assert str(number) in os.listdir(sampleFolderPath), "sample {sampleNumber} does not exist in {path}".format(sampleNumber=number, path=sampleFolderPath)
        sampleFolder = sampleFolderPath + '/' + str(number)
        shutil.rmtree(sampleFolder)
        self.relabel()

    def __relabel_folder(self, folderpath):
        count = 1
        for dir in os.listdir(folderpath):
            if dir[0] == '.':
                continue
            dirpath = folderpath+ '/' + dir
            newdirpath = folderpath + '/' + 'new-' + str(count)
            count += 1
            os.rename(dirpath, newdirpath)
        count = 1
        for i, dir in enumerate(os.listdir(folderpath)):
            if dir[0] == '.':
                continue
            dirpath = folderpath + '/' + dir
            newdirpath = folderpath + '/' + str(count)
            count += 1
            os.rename(dirpath, newdirpath)

    def create_temp_folder(self, directory_name, rgb, nir, is_positive):
        """
        creates a directory under dataset and put rgb and nir in there
        """
        directory_path = self.root + '/' + directory_name
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        dataset = MyDataSetFolder(root=directory_path)


    def delete_temp_folder(self, directory_name):
        """
        delete everything in and including directory_name
        """
        directory_path = self.root + '/' + directory_name
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)

if __name__ == '__main__':
    dataset = MyDataSetFolder()
    dataset.create_temp_folder("reinforce", None, None, None)
    dataset.delete_temp_folder("reinforce")
