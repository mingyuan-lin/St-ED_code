import torch
import torch.nn as nn
import torchvision.models

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        '''
         use vgg19 conv3_3 feature, before relu layer
        '''
        self.feature_list = [14]
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])


    def forward(self, x):
        x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)

        return features