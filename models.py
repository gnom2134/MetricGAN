import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hidden = None
        self.lstm = nn.LSTM(257, 200, num_layers=2, bidirectional=True, dropout=0.15).to(device)
        self.fc = nn.Sequential(
            nn.Linear(400, 200),
            nn.Linear(200, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(300, 257),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        x, h = self.lstm(x)
        x = self.fc.forward(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 15, (5, 5)),
            nn.LeakyReLU(),
            nn.Conv2d(15, 25, (7, 7)),
            nn.LeakyReLU(),
            nn.Conv2d(25, 40, (9, 9)),
            nn.LeakyReLU(),
            nn.Conv2d(40, 50, (11, 11)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            Flatten(),
            nn.Linear(200, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x, clean_signals):
        x = torch.stack((x, clean_signals), 1)
        return self.network(x).reshape((1, -1))


class DiscriminatorLoss(nn.Module):
    def __init__(self, device):
        self.device = device
        super().__init__()

    def forward(self, x, metric_value):
        result = torch.FloatTensor().to(self.device)
        for i in range(x.shape[0]):
            result = torch.cat((result, - torch.log(1 - (x[i] - metric_value)**2)), 0)
        return result


class GeneratorLoss(nn.Module):
    def __init__(self, device, s=1.):
        super().__init__()
        self.device = device
        self.s = s

    def forward(self, x):
        result = torch.FloatTensor().to(self.device)
        for i in range(x.shape[0]):
            result = torch.cat((result,  - torch.log(1 - (x[i] - self.s)**2)), 0)
        return result
