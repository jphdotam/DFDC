import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from .backbones import *


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ImageTCN(nn.Module):

    def __init__(self, 
                 # Encoder
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 # TCN
                 num_inputs,
                 num_channels, 
                 kernel_size=3, 
                 tcn_dropout=0.5):

        super().__init__()
        self.encoder, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, tcn_dropout)
        self.linear1 = nn.Conv1d(num_inputs, 1, kernel_size=1)
        self.linear2 = nn.Linear(dim_feats, num_classes)

        self.num_inputs = num_inputs
        self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode=mode)
        self._freeze_encoder()

    def forward_train(self, x):
        features = []
        for i in range(x.size(2)):
            features.append(self.dropout1(self.encoder(x[:,:,i])))
        features = torch.stack(features, dim=1)
        features = self.tcn(features)
        out = self.linear1(features)
        out = self.dropout2(out)
        out = self.linear2(out[:,0])
        return out[:,0]

    def forward_test(self, x):
        # x.shape = (1, C, N, H, W)
        if x.size(0) != 1: raise Exception('Batch size must be 1 for inference')
        if x.size(2) < self.num_inputs:
            x = torch.cat([x, torch.from_numpy(np.zeros((x.shape[0],x.shape[1],self.num_inputs-x.shape[2],x.shape[3],x.shape[4]))).float()], dim=2)
        preds = []
        for i in range(0, x.size(2)-self.num_inputs+1, self.num_inputs):
            preds.append(self.forward_train(x[:,:,i:i+self.num_inputs]))
        if x.size(2) > self.num_inputs and x.size(2) % self.num_inputs != 0:
            preds.append(self.forward_train(x[:,:,x.size(2)-self.num_inputs:x.size(2)]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)



if __name__ == '__main__':
    import torch, numpy as np
    from factory.models.tcn import ImageTCN
    model = ImageTCN('se_resnext50', 1, 0.5, None, 50, [50,50,50,50])
    model.eval()
    X = torch.from_numpy(np.ones((1,3,60,64,64))).float()
    out = model(X)





