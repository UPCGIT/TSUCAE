import torch.nn as nn


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


class Unmixing(nn.Module):
    def __init__(self, band_Number, endmember_number, drop_out, col):
        super(Unmixing, self).__init__()
        self.endmember_number = endmember_number
        self.band_number = band_Number
        self.col = col
        self.layer1 = nn.Sequential(
            conv33(band_Number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.encodelayer = nn.Sequential(nn.Softmax(dim=1))

        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ), )

    def forward(self, x, mask):
        layer1out = self.layer1(x)

        en_result1 = layer1out / mask

        en_result1 = self.encodelayer(en_result1)
        de_result1 = self.decoderlayer4(en_result1)

        return en_result1, de_result1
