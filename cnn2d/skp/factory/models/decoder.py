import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = 32
        if out_channels % num_groups != 0 or out_channels == num_groups:
            num_groups = out_channels // 2

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=None, **_):
        super().__init__()
        self.up_sample = up_sample

        self.block = nn.Sequential(
                Conv3x3GNReLU(in_channels, out_channels),
                Conv3x3GNReLU(out_channels, out_channels),
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x, skip = x
        else:
            skip = None

        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.block(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout=0.2,
                 out_channels=[256, 128, 64, 32, 16],
                 **_):
        super().__init__()
        self.out_channels = out_channels

        self.center = DecoderBlock(in_channels=in_channels, out_channels=in_channels,
                                   up_sample=None)

        self.layer1 = DecoderBlock( in_channels, out_channels[0], up_sample=None)
        self.layer2 = DecoderBlock(out_channels[0], out_channels[1], up_sample=2)
        self.layer3 = DecoderBlock(out_channels[1], out_channels[2], up_sample=2)
        self.layer4 = DecoderBlock(out_channels[2], out_channels[3], up_sample=2)
        self.layer5 = DecoderBlock(out_channels[3], out_channels[4], up_sample=2)

        self.final = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)
        )

    def forward(self, x, size):

        center = self.center(x)
        decodes = self.layer1(center)
        #decodes = F.interpolate(decodes, scale_factor=size//32, mode='bilinear')
        decodes = self.layer2(decodes) 
        decodes = self.layer3(decodes) 
        decodes = self.layer4(decodes) 
        decodes = self.layer5(decodes)

        decodes4 = F.interpolate(decodes, (size,size), mode='bilinear')
        outputs = self.final(decodes4)
        return torch.sigmoid(outputs)


class VAE(nn.Module):
    def __init__(self, in_channels, loss_weight=0.2, dropout=0.2, out_channels=[256, 128, 64, 32, 16]):
        super().__init__()

        # num_classes=3 for 3-channel image
        self.loss_weight = loss_weight
        self.decoder = Decoder(3, in_channels, dropout, out_channels)
        self.dim_feats = in_channels

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, size):
        return self.decoder(z, size)

    def forward(self, x, size):
        mu, logvar = x[:,:self.dim_feats], x[:,self.dim_feats:]
        z = self.reparameterize(mu, logvar)
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.decode(z, size), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        print(recon_x.shape)
        print(x.shape)
        dist = F.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = (dist + KLD)*self.loss_weight
        print('logvar {:.4f} mu {:.4f}'.format(logvar.mean(), mu.mean()))
        print('DIST {:.4f} KLD {:.4f}'.format(dist, KLD))
        print('{:.4f}'.format(recon_loss))
        return recon_loss

