import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from MyDataSet import MyDataset
from DeepLearningEffectivenessExplorer import ExplorereDataset
from CustomLayers import NDVILayer, NDVILoss
from ImageManager import ImageManager

RGB_HIGH_THRESH = 150
RGB_LOW_THRESH = 0
max_epoches = 10 # 100

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(6, 2)
        #self.fc2 = nn.Linear(5, 4)
        #self.fc3 = nn.Linear(4, 3)
        #self.fc4 = nn.Linear(3, 2)
        self.ndvi = NDVILayer

    def forward(self, x):
        #x = F.tanh(self.fc1(x))
        #x = F.tanh(self.fc2(x))
        #x = F.tanh(self.fc3(x))
        x = self.fc1(x)
        x = self.ndvi.apply(x)
        return x

def print_param_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad)

def train_model(model, trainloader, criterion, max_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            #print('{index} loss: {loss} output: {output} label: {label}'.format(index=i, loss=loss, output=outputs, label=labels))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1:
                print_param_grad(model)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    for name, param in model.named_parameters():
        print(name, param.data)
    print('Finished Training')

def is_healthy(ndvi):
    return 0.3 < ndvi < 0.8

def is_prediction_correct(prediction, labels):
    if is_healthy(prediction[0][0]):
        return labels[0] == 1
    return labels[0] == 0

def evaluate_model(model, loader, isTest):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            input, labels = data
            outputs = model(input.float())
            total += labels.size(0)
            if is_prediction_correct(outputs, labels):
                correct += 1
    if isTest:
        print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    else:
        print('Accuracy of the network on the training set: %d %%' % (100 * correct / total))

def get_model_output_as_image(model, rgb, nir, isColored):
    assert rgb.shape == nir.shape
    #yellow = np.array([255,255,0])
    imageManager = ImageManager()
    result = np.zeros((rgb.shape[0], rgb.shape[1]))
    foreground = imageManager.segment_green_objects(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if not foreground[i][j]:
                if not isColored:
                    result[i][j] = False
                else:
                    result[i][j] = 0
                continue
            rgb_pixels = rgb[i][j]
            nir_pixels = nir[i][j]
            input = np.concatenate([rgb_pixels, nir_pixels])
            input = [input]
            output = model(torch.FloatTensor(input))
            if not isColored:
                result[i][j] = is_healthy(output)
            else:
                result[i][j] = output * 255
                #if (i * rgb.shape[0] + j) % 1000 == 0:
                #    print("pixel value at {i}, {j}: {pixel}".format(i=i,j=j, pixel=input),output)
    print('result generated')
    return result

def get_model_output_as_rgb_image(model, path):
    rgb = io.imread(path + 'RGB.jpg')
    nir = io.imread(path + 'NIR.jpg')
    nir = resize(nir, rgb.shape) * 255
    assert rgb.shape == nir.shape
    rgb_path = path + "RGB.jpg"
    yellow = np.array([255,255,0])
    green = np.array([0, 128, 0])
    red = np.array([255, 0, 0])
    imageManager = ImageManager()
    result = np.zeros((rgb.shape[0], rgb.shape[1], 3))
    foreground = imageManager.segment_green_objects(rgb_path)
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
            if is_healthy(ndvi_output):
                result[i][j] = green
            else:
                result[i][j] = yellow
            # if (i * rgb.shape[0] + j) % 1000 == 0:
            #    print("pixel value at {i}, {j}: {pixel}".format(i=i,j=j, pixel=input),output)
    print('result generated')
    return result

def split_dataset(dataset: MyDataset):
    if dataset.isBalanced:
        count = 5000
        return torch.utils.data.random_split(dataset, [count, len(dataset)-count])
    return torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//10, len(dataset)//10])

def get_images(model, paths, isColored):
    for path in paths:
        rgb = io.imread(path+'RGB.jpg')
        nir = io.imread(path+'NIR.jpg')
        nir = resize(nir, rgb.shape) * 255
        model_output_image = get_model_output_as_image(model, rgb, nir, isColored)
        io.imshow(model_output_image)
        plt.show()

def get_ground_truth(dataset, path):
    rgb = io.imread(path + 'RGB.jpg')
    nir = io.imread(path + 'NIR.jpg')
    nir = resize(nir, rgb.shape) * 255
    GT = dataset._compute_ground_truth(rgb, nir)
    io.imshow(GT)
    plt.show()

if __name__ == '__main__':
    #dataset = MyDataset(balanced=False, update_GT=True)
    #trainset, testset = split_dataset(dataset)
    #trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    #testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    criterion = NDVILoss.apply
    model = Model()
    #train_model(model, trainloader, criterion, max_epochs=1)
    #torch.save(model.state_dict(), './trained_models/model1_GT2.pt')
    model.load_state_dict(torch.load('./trained_models/model1_10epochs.pt'))
    #for name, param in model.named_parameters():
    #    print(name, param.data)
    #evaluate_model(model, trainloader, False)
    #evaluate_model(model, testloader, True)
    paths = ['./dataset/test/2/']
    #get_ground_truth(dataset, paths[0])
    rgb_image = get_model_output_as_rgb_image(model, paths[0])
    io.imsave("demo1.jpg", rgb_image)
    plt.imshow(rgb_image.astype(np.uint8))
    plt.show()
