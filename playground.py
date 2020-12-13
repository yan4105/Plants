from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
import numpy as np

class MyLinear(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print("called", grad_output)
        return 3*grad_output

class PlaygroundModel(nn.Module):
    def __init__(self):
        super(PlaygroundModel, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        torch.nn.init.uniform_(self.fc1.weight, -0.5, 0.5)
        self.fc2 = nn.Linear(2, 1)
        torch.nn.init.uniform_(self.fc2.weight, -0.5, 0.5)
        self.layer3 = MyLinear.apply

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.layer3(x)
        return x

if __name__ == '__main__':
    a = np.array([[True, False],[False, False]])
    b = np.array([[False,True],[False, False]])
    c = (a or b).all(axis=1)
    print(a, b, c)

