import torch
import torch.nn as nn
import torchvision.models as backbone_




class EncoderCNN(nn.Module):
    def __init__(self, hp=None):
        super(EncoderCNN, self).__init__()
        self.feature = backbone_.vgg16(pretrained=True).features
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        self.fc_mu = nn.Linear(512, 128)
        self.fc_std = nn.Linear(512, 128)

    def forward(self, x):
        backbone_feature = self.feature(x)
        x = torch.flatten(self.pool_method(backbone_feature), start_dim=1)
        mean = self.fc_mu(x)
        log_var = self.fc_std(x)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return backbone_feature, posterior_dist


# class EncoderCNN(nn.Module):
#     def __init__(self, hp=None):
#         super(EncoderCNN, self).__init__()
#         self.feature = Unet_Encoder(in_channels=3)
#         self.fc_mu = nn.Linear(512, 128)
#         self.fc_std = nn.Linear(512, 128)
#
#     def forward(self, x):
#         x = self.feature(x)
#         mean = self.fc_mu(x)
#         log_var = self.fc_std(x)
#         posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
#         return posterior_dist

class DecoderCNN(nn.Module):
    def __init__(self, hp=None):
        super(DecoderCNN, self).__init__()
        self.model = Unet_Decoder(out_channels=3)
    def forward(self, x):
        return self.model(x)


class Unet_Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet_Encoder, self).__init__()

        self.down_1 = Unet_DownBlock(in_channels, 32, normalize=False)
        self.down_2 = Unet_DownBlock(32, 64)
        self.down_3 = Unet_DownBlock(64, 128)
        self.down_4 = Unet_DownBlock(128, 256)
        self.down_5 = Unet_DownBlock(256, 256)
        self.linear_encoder = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)
        x = self.down_5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_encoder(x)
        x = self.dropout(x)
        return x


class Unet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Unet_Decoder, self).__init__()
        self.linear_1 = nn.Linear(128, 8*8*256)
        self.dropout = nn.Dropout(0.5)
        self.deconv_1 = Unet_UpBlock(256, 256)
        self.deconv_2 = Unet_UpBlock(256, 128)
        self.deconv_3 = Unet_UpBlock(128, 64)
        self.deconv_4 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1), nn.Tanh()])

    def forward(self, x):
        x = self.linear_1(x)
        x = x.view(-1, 256, 8, 8)
        x = self.dropout(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.final_image(x)
        return x


class Unet_UpBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc):
        super(Unet_UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(outer_nc),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Unet_DownBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, normalize=True):
        super(Unet_DownBlock, self).__init__()
        layers = [nn.Conv2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(outer_nc))
        layers.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGG_encoder(nn.Module):
    def __init__(self, hp):
        super(VGG_encoder, self).__init__()
        self.feature = backbone_.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        x = self.dropout(x)
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':

    pass