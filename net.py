import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from utils import Whiten2d, PONO, MS


class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            torch.nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            torch.nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU(0.2))
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = ConvBlock(ch_in=3, ch_out=64)
        self.conv1 = ConvBlock(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=64)
        self.conv3 = ConvBlock(ch_in=64, ch_out=64)
        self.conv4 = ConvBlock(ch_in=64, ch_out=64)
        self.IW1 = Whiten2d(64)
        self.IW2 = Whiten2d(64)
        self.IW3 = Whiten2d(64)
        self.IW4 = Whiten2d(64)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_in(x)

        x1, x1_mean, x1_std = PONO(x)
        x1 = self.conv1(x)
        x2 = self.pool(x1)

        x2, x2_mean, x2_std = PONO(x2)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)

        x3, x3_mean, x3_std = PONO(x3)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)

        x4, x4_mean, x4_std = PONO(x4)
        x4 = self.conv4(x4)

        x4_iw = self.IW4(x4)
        x3_iw = self.IW3(x3)
        x2_iw = self.IW2(x2)
        x1_iw = self.IW1(x1)

        return x1_iw, x2_iw, x3_iw, x4_iw, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()

        self.device = device

        self.encoder = Encoder()
        self.UpConv4 = ConvBlock(ch_in=64, ch_out=64)
        self.Up3 = Up()
        self.UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.Up2 = Up()
        self.UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.Up1 = Up()
        self.UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_u4 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s4 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u3 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s3 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        out_conv = []
        for i in range(1):
            out_conv.append(ResBlock(64))
        out_conv.append(nn.ReflectionPad2d(1))
        out_conv.append(nn.Conv2d(64, 3, kernel_size=3, padding=0))
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, Input):
        x1, x2, x3, x4, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std = self.encoder.forward(Input)

        # x4->x3
        x4_mean = self.conv_u4(x4_mean)
        x4_std = self.conv_s4(x4_std)
        x4 = MS(x4, x4_mean, x4_std)
        x4 = self.UpConv4(x4)
        d3 = self.Up3(x4)
        # x3->x2
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.UpConv3(d3)
        x3_mean = self.conv_u3(x3_mean)
        x3_std = self.conv_s3(x3_std)
        d3 = MS(d3, x3_mean, x3_std)
        d2 = self.Up2(d3)
        # x2->x1
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.UpConv2(d2)
        x2_mean = self.conv_u2(x2_mean)
        x2_std = self.conv_s2(x2_std)
        d2 = MS(d2, x2_mean, x2_std)
        d1 = self.Up1(d2)
        # x1->out
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.UpConv1(d1)
        x1_mean = self.conv_u1(x1_mean)
        x1_std = self.conv_s1(x1_std)
        d1 = MS(d1, x1_mean, x1_std)
        out = self.out_conv(d1)

        return out


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.device = torch.device(opt.device)
        self.decoder = Decoder(device=self.device).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.VGG16 = PerceptionLoss().to(self.device)

    def forward(self, Input):
        return self.decoder.forward(Input)

    def loss(self, outputs, labels):
        reconstruction_loss = self.criterion(outputs, labels)
        vgg16_loss = self.VGG16(outputs, labels)

        loss = reconstruction_loss + 0.1 * vgg16_loss
        return loss

