import torch
import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):

    def __init__(self, num_keypoints=4, grad_from=3):
        super(MyModel, self).__init__()
        self.effnet = models.efficientnet_b0(pretrained=False)
        self.n_ouputs_last_layer = 1280 * 7 * 7

        for name, param in self.effnet.features.named_parameters():
            if int(name.split('.')[0]) < grad_from:
                param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.n_ouputs_last_layer, num_keypoints),
        )

    def forward(self, x):
        x = self.effnet.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)

        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x