## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## DONE: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1    = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2    = nn.Conv2d(32, 64, 5)
        self.pool2    = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3    = nn.Conv2d(64, 8, 3)
        self.pool3    = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.2)

        # TODO: if we parametrize the above layers we can calculate the size of the Linear layer in place of having magic numbers
        self.linear = nn.Linear(8 * 25 * 25, 2 * 68)

        
    def forward(self, input_image):
        ## DONE: Define the feedforward behavior of this model
        input_image = self.pool1(F.relu(self.conv1(input_image)))
        input_image = self.dropout1(input_image)

        input_image = self.pool2(F.relu(self.conv2(input_image)))
        input_image = self.dropout2(input_image)

        input_image = self.pool3(F.relu(self.conv3(input_image)))
        input_image = self.dropout3(input_image)

        input_image = input_image.view(input_image.size(0), -1)
        input_image = self.linear(input_image)

        return input_image
