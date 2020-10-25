import torch
import torch.nn as nn
import torch.nn.functional as F


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
            )
        

    def forward(self, x, y):
        x = self.conv(x)
        # applied
        y = y.view(-1, 1, 3, 1)
        x = torch.matmul(x, y)
        x = x.view(x.size(0), -1)
        return x


class Net(nn.Module):
    def __init__(self, nfeat, outdims=3):
        super(Net, self).__init__()
        self.nfeat = nfeat
        self.outdims = outdims

        self.GNetList = nn.ModuleList([GNet() for _ in range(nfeat)])
        self.fc = nn.Sequential(
            nn.BatchNorm1d(3*nfeat),
            nn.Linear(3*nfeat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, outdims),
            )

    def forward(self, x, y):
        x = torch.cat([subnet(x[:,i:i+1,:,:], y[:,i*3:i*3+3]) for i, subnet in enumerate(self.GNetList)], dim=1)
        x = self.fc(x)
        x = F.normalize(x)
        return x



def load_model(path_model, nfeatures):
    model = Net(nfeatures)
    model.load_state_dict(torch.load(path_model))
    return model



