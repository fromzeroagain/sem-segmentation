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


def decoder_block(in_channels, out_channels, num_convs=2, dropout=False):
    layers = []
    for i in range(num_convs):
        in_channels = in_channels if i == 0 else out_channels
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
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

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec5 = decoder_block(512, 512, num_convs=3, dropout=True)
        self.dec4 = decoder_block(512, 256, num_convs=3, dropout=True)
        self.dec3 = decoder_block(256, 128, num_convs=3, dropout=False)
        self.dec2 = decoder_block(128, 64, num_convs=2, dropout=False)
        self.dec1 = decoder_block(64, 64, num_convs=2, dropout=False)

        self.output_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)  # gamma starts at 1
                nn.init.constant_(module.bias, 0)  # beta starts at 0

    def forward(self, x):
        # pool_indices = []

        x = self.enc1(x)  # (3,512,512)->(64,512,512)
        x, idx1 = self.pool(x)  # (64,256,256)
        # pool_indices.append(idx1)

        x = self.enc2(x)  # (64,256,256)->(128,256,256)
        x, idx2 = self.pool(x)
        # pool_indices.append(idx2)  # (128,128,128)

        x = self.enc3(x)  # (256,128,128)
        x, idx3 = self.pool(x)
        # pool_indices.append(idx3)  # (256,64,64)

        x = self.enc4(x)  # (512,64,64)
        x, idx4 = self.pool(x)
        # pool_indices.append(idx4)  # (512,32,32)

        x = self.enc5(x)  # (512,32,32)
        x, idx5 = self.pool(x)
        # pool_indices.append(idx5)  # (512,16,16)

        # return x, pool_indices

        x = self.unpool(x, idx5)
        x = self.dec5(x)
        x = self.unpool(x, idx4)
        x = self.dec4(x)
        x = self.unpool(x, idx3)
        x = self.dec3(x)
        x = self.unpool(x, idx2)
        x = self.dec2(x)
        x = self.unpool(x, idx1)
        x = self.dec1(x)

        logits = self.output_conv(x)  # (B,64,512,512)->(B,1,512,512)
        return logits


if __name__ == "__main__":
    # device = config.DEVICE
    # model = SegNet(n_classes=1).to(device)
    # total_params = sum(p.numel() for p in model.parameters())

    # print(f"Parameters so far (encoder only): {total_params:,}")

    # dummy = torch.randn(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(config.DEVICE)

    # print(f"input shape:{dummy.shape}")

    # model.eval()
    # with torch.no_grad():
    #     bottleneck, indices = model(dummy)

    # print(f"Bottleneck shape: {bottleneck.shape}; (expect [2, 512, 16, 16])")
    # for i, idx in enumerate(indices):
    #     print(f"stage {i + 1} indices, shape:{idx.shape}")
    # assert bottleneck.shape == (2, 512, 16, 16), (
    #     "Wrong bottleneck shape : check pooling stages"
    # )
    device = config.DEVICE
    print(f"DEVICE: {device}")
    model = SegNet(n_classes=1).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:{total}")

    dummy = torch.randn(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)

    print(f"Input shape:{dummy.shape}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy)

    print(f"Output shape:{logits.shape}")

    print(f"Logit range : [{logits.min()},{logits.max()}")
    probs = torch.sigmoid(logits)
    print(f"prob range:{probs.min()},{probs.max()}")

    model.train()
    dummy_train = torch.randn(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(
        config.DEVICE
    )

    dummy_mask = (
        torch.randint(0, 2, (2, 1, config.IMG_HEIGHT, config.IMG_WIDTH))
        .float()
        .to(config.DEVICE)
    )

    logits_train = model(dummy_train)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits_train, dummy_mask)
    loss.backward()

    print(f"Loss(random dummy data):{loss.item()}")
