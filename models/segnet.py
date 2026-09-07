import torch
import torch.nn as nn
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def encoder_block(in_channels, out_channels, num_convs=2, dropout=False):
    layers = []
    for i in range(num_convs):
        conv_in = in_channels if i == 0 else out_channels
        layers.append(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

    if dropout:
        layers.append(nn.Dropout2d(p=config.DROPOUT_RATE))
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    def __init__(self, n_classes=1):
        super(SegNet, self).__init__()
        self.n_classes = n_classes

        self.enc1 = encoder_block(3, 64, num_convs=2, dropout=False)
        self.enc2 = encoder_block(64, 128, num_convs=2, dropout=False)
        self.enc3 = encoder_block(128, 256, num_convs=3, dropout=False)
        self.enc4 = encoder_block(256, 512, num_convs=3, dropout=True)
        self.enc5 = encoder_block(512, 512, num_convs=3, dropout=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def encode(self, x):
        pool_indices = []

        x = self.enc1(x)  # (3,512,512)->(64,512,512)
        x, idx1 = self.pool(x)  # (64,256,256)
        pool_indices.append(idx1)

        x = self.enc2(x)  # (64,256,256)->(128,256,256)
        x, idx2 = self.pool(x)
        pool_indices.append(idx2)  # (128,128,128)

        x = self.enc3(x)  # (256,128,128)
        x, idx3 = self.pool(x)
        pool_indices.append(idx3)  # (256,64,64)

        x = self.enc4(x)  # (512,64,64)
        x, idx4 = self.pool(x)
        pool_indices.append(idx4)  # (512,32,32)

        x = self.enc5(x)  # (512,32,32)
        x, idx5 = self.pool(x)
        pool_indices.append(idx5)  # (512,16,16)

        return x, pool_indices

    def forward(self, x):
        x, pool_indices = self.encode(x)
        return x, pool_indices
