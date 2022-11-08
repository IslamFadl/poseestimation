from torch import nn
from torchvision.models import resnet50, resnet34, resnet101


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, stride = 1, kernel_size=3) # O = [(W−K+2P)/S]+1, padding = 0 , I = 224, O = ((224-3+2*0)/2) +1 = 222
        self.res_torch = resnet50()
        self.final = nn.Linear(1000,1)

    def forward(self, x, angles = None): # why do I need angles=None here? bug1 # try nn.seq to
        x = self.conv(x)
        #print('Output shape of the first convolution:   ', x.shape)
        x = self.res_torch(x)
        #print('Output shape of the resnet:              ', x.shape)
        #x = self.final(x) # this executed two times
        #print('Output shape of the final linear layer:  ', x.shape, '\n')
        return self.final(x)