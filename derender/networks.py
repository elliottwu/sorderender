import numpy as np
import torch
import torch.nn as nn
import torchvision


EPS = 1e-7


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=64, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
        ]
        add_downsample = int(np.log2(in_size//64))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
                    nn.ReLU(inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)
        ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)


class SoRNet(nn.Module):
    def __init__(self, cin, cout2=5, in_size=64, out_size=32, zdim=128, nf=64, activation=nn.Tanh):
        super(SoRNet, self).__init__()
        encoder = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
        ]

        add_downsample = int(np.log2(in_size//64))
        if add_downsample > 0:
            for _ in range(add_downsample):
                encoder += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
                    nn.ReLU(inplace=True),
                ]

        encoder += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        ]
        self.encoder = nn.Sequential(*encoder)

        out_net1 = []
        add_upsample = int(np.log2(out_size//2))
        if add_upsample > 0:
            for _ in range(add_upsample):
                out_net1 += [
                    nn.Upsample(scale_factor=(2,1), mode='nearest'),  # 1x1 -> 2x1
                    nn.Conv2d(zdim, zdim, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False, padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                ]

        out_net1 += [
            nn.Upsample(scale_factor=(2,1), mode='nearest'),  # 16x1 -> 32x1
            nn.Conv2d(zdim, 1, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False, padding_mode='replicate'),
        ]
        if activation is not None:
            out_net1 += [activation()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, cout2),
            nn.Sigmoid(),
            # nn.Tanh(),
        ]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        z = self.encoder(input)
        out1 = self.out_net1(z).view(input.size(0), -1)
        out2 = self.out_net2(z.view(input.size(0), -1)) # /2+0.5
        return out1, out2


class EnvMapNet(nn.Module):
    def __init__(self, cin, cout, cout2=None, in_size=64, out_size=16, zdim=128, nf=64, activation=nn.Tanh):
        super(EnvMapNet, self).__init__()
        ## downsampling
        encoder = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            # nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            # nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            # nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*8, nf*8),
            # nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                encoder += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    nn.GroupNorm(16*8, nf*8),
                    # nn.BatchNorm2d(nf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        encoder += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)
        ]
        self.encoder = nn.Sequential(*encoder)

        ## upsampling
        decoder_envmap = [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=(2,6), stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
        ]

        add_upsample = int(np.log2(out_size//16))
        if add_upsample > 0:
            for _ in range(add_upsample):
                decoder_envmap += [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate'),
                    nn.GroupNorm(16*8, nf*8),
                    nn.ReLU(inplace=True),
                ]

        decoder_envmap += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate'),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate'),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate'),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False, padding_mode='replicate')
        ]
        self.decoder_envmap = nn.Sequential(*decoder_envmap)
        if activation is not None:
            self.act = activation()
        else:
            self.act = None

        if cout2 is not None:
            decoder_light_param = [
                nn.Linear(zdim, zdim),
                nn.ReLU(inplace=True),
                nn.Linear(zdim, cout2),
                nn.Sigmoid()
            ]
            self.decoder_light_param = nn.Sequential(*decoder_light_param)
        else:
            self.decoder_light_param = None

    def forward(self, input):
        z = self.encoder(input)
        env_map = self.decoder_envmap(z)
        env_map = env_map - 2  # initial sigmoid(-2)
        # env_map = env_map - 3  # initial sigmoid(-3), for 32x96 env_map
        if self.act is not None:
            env_map = self.act(env_map)

        if self.decoder_light_param is not None:
            light_param = self.decoder_light_param(z.view(*z.shape[:2]))
            return env_map, light_param
        else:
            return env_map


class DiscNet(nn.Module):
    def __init__(self, cin, cout, nf=64, norm=nn.InstanceNorm2d, activation=None):
        super(DiscNet, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            norm(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            norm(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            norm(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            # norm(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)
