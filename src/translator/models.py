"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/translator/models.py
"""
from utils.utils import *


class ResidualBlock(nn.Module):
    """
    Residual Block with batch normalization.
    """
    def __init__(self, dim_in, dim_out):
        """
        Class initializer.
        """
        super(ResidualBlock, self).__init__()
        self.norm = nn.BatchNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(dim_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        """
        Forward propagation.
        """
        return x + self.main(x)


class Encoder(nn.Module):
    """
    Encoder with CNN structure.
    """
    def __init__(self, in_channels=3, out_channels=32, num_down=3, num_blocks=3):
        """
        Class initializer.
        """
        super(Encoder, self).__init__()
        self.norm = nn.BatchNorm2d
        # first convolution
        down_layers = []
        down_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
        down_layers.append(self.norm(out_channels, affine=True, track_running_stats=True))
        down_layers.append(nn.ReLU(inplace=True))

        # down-sampling
        curr_dim = out_channels
        for i in range(num_down):
            down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            down_layers.append(self.norm(curr_dim*2, affine=True, track_running_stats=True))
            down_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # down layers ad sequential
        self.down = nn.Sequential(*down_layers)

        # bottleneck
        bot_layers = []
        for i in range(num_blocks):
            bot_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bottleneck = nn.Sequential(*bot_layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = self.bottleneck(self.down(x))
        return out


class Decoder(nn.Module):
    """
    Decoder with transposed-CNN structure.
    """
    def __init__(self, in_channels, out_channels=3, num_up=3, num_blocks=3):
        """
        Class initializer.
        """
        super(Decoder, self).__init__()
        self.norm = nn.BatchNorm2d
        # bottleneck
        curr_dim = in_channels
        bot_layers = []
        for i in range(num_blocks):
            bot_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bottleneck = nn.Sequential(*bot_layers)

        # up-sampling
        up_layers = []
        for i in range(num_up):
            up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            up_layers.append(self.norm(curr_dim//2, affine=True, track_running_stats=True))
            up_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        up_layers.append(nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
        up_layers.append(nn.Tanh())

        self.up = nn.Sequential(*up_layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = self.up(self.bottleneck(x))
        return out


class DomainWeight(nn.Module):
    """
    Domain embedding factors.
    """
    def __init__(self, num_domain, dim, temperature):
        """
        Class initializer.
        """
        super(DomainWeight, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_domain, embedding_dim=dim)
        self.temperature = temperature

    def forward(self, tgt_idx, src_idx):
        """
        Forward propagation.
        """
        tgt = self.emb(tgt_idx)
        src = self.emb(src_idx)
        weight = (tgt*src).sum(dim=1) / self.temperature
        weight = weight.softmax(dim=0)
        return weight
