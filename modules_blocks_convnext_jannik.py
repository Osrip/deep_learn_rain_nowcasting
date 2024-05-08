import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange


class ConvNext(nn.Module):
    """
    Convolutional block based on "A ConvNet for the 2020s"
    """

    def __init__(self, channels: int, channels_c: int, k: int = 7, r: int = 4):
        """
        channels: number of input channels
        channels_c: number of conditioning channels
        k: kernel size
        r: expansion ratio
        """
        super().__init__()
        self.channels_last = Rearrange('b c h w -> b h w c')
        self.channels_first = Rearrange('b h w c -> b c h w')

        self.dwconv = nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding="same", groups=channels)
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * r),
            nn.SiLU(),
            nn.Linear(channels * r, channels)
        )
        self.affine_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels_c, channels * 3),
            Rearrange('b (c ssg) -> ssg b () () c', ssg=3)
        )
        self.channels = channels

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, C_c]
        returns: output tensor [N, C, H, W]
        """
        res = x
        scale, shift, gate = self.affine_map(
            c if c is not None else torch.zeros((x.shape[0], self.channels), device=x.device))
        x = self.dwconv(x)
        x = self.channels_last(x)
        x = self.norm(x)
        x = (1 + scale) * x + shift
        x = self.mlp(x)
        x = x * gate
        x = self.channels_first(x)
        x = x + res
        return x


class Resize(nn.Module):
    """
    Resize block based on "A ConvNet for the 2020s"
    """

    def __init__(self, channels_in: int, channels_out: int, channels_c: int, scale: int = 2, transpose: bool = False):
        """
        channels_in: number of input channels
        channels_out: number of output channels
        channels_c: number of conditioning channels
        scale: scaling factor
        transpose: whether to use transposed convolution (i.e. upsampling) or not (i.e. downsampling)
        """
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=scale, stride=scale)
        else:
            self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=scale, stride=scale)

        self.channels_last = Rearrange('b c h w -> b h w c')
        self.channels_first = Rearrange('b h w c -> b c h w')

        self.norm = nn.LayerNorm(channels_out, elementwise_affine=False)
        self.affine_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels_c, channels_out * 2),
            Rearrange('b (c ss) -> ss b () () c', ss=2)
        )
        self.channels = channels_out

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, C_c]
        returns: output tensor [N, C, H', W']
        """
        shift, scale = self.affine_map(
            c if c is not None else torch.zeros((x.shape[0], self.channels), device=x.device))
        x = self.conv(x)
        x = self.channels_last(x)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        x = self.channels_first(x)
        return x


class FrequencyEmbedding(nn.Module):
    def __init__(self, channels: int, freq_dim: int = 256, max_period: float = 1e4, learnable: bool = True):
        """ Time step embedding module learns a set of sinusoidal embeddings for the input sequence.
        dim: dimension of the input
        freq_dim: number of frequencies to use
        max_period: maximum period to use
        learnable: whether to learn the time step embeddings
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, channels), nn.SiLU(),
                                 nn.Linear(channels, channels)) if learnable else nn.Identity()
        self.freq_dim = freq_dim
        self.max_period = max_period

    @staticmethod
    def frequency_embedding(t: torch.Tensor, num_freqs: int, max_period: float):
        """
        t: [N] tensor of time steps to embed
        num_freqs: number of frequencies to use
        max_period: maximum period to use
        returns: [N, num_freqs] tensor of embedded time steps
        """
        half = num_freqs // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if num_freqs % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # zero pad
        return embedding

    def forward(self, t: torch.Tensor):
        """
        t: [N] tensor of time steps to embed
        returns: [N, dim] tensor of embedded time steps
        """
        t_freqs = self.frequency_embedding(t, self.freq_dim, self.max_period)
        return self.mlp(t_freqs)


class U_Net(nn.Module):
    """
    U-Net architecture with ConvNext blocks and conditioning.
    """

    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 num_channels: list = [64, 256, 512],
                 num_encoder_blocks: list = [2, 2],
                 num_decoder_blocks: list = [4, 4],
                 scales: list = [4, 2],
                 num_tails: int = 1,
                 k: int = 7,
                 r: int = 4,
                 num_freqs: int = 64,
                 channels_c: int = 256):
        """
        channels_in: number of input channels
        channels_out: number of output channels
        num_channels: number of channels in each stage
        num_encoder_blocks: number of ConvNext blocks in the encoder
        num_decoder_blocks: number of ConvNext blocks in the decoder
        scales: scaling factors in each stage
        num_tails: number of tails to ensemble
        k: kernel size
        r: expansion ratio
        num_freqs: number of frequencies to embed the conditioning
        channels_c: number of conditioning channels
        """
        super().__init__()
        self.num_stages = len(num_channels)

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        self.conditioning = FrequencyEmbedding(channels_c, num_freqs)
        # add linear blocks
        self.encoder["linear"] = nn.Conv2d(channels_in, num_channels[0], kernel_size=1)
        self.decoder["linear"] = nn.ModuleList(
            [nn.Conv2d(num_channels[0], channels_out, kernel_size=1) for _ in range(num_tails)])
        # multi-stage U-Net:
        for i in range(1, self.num_stages):
            # add Resize blocks
            self.encoder[f"resize_{i}"] = Resize(num_channels[i - 1], num_channels[i], channels_c, scale=scales[i - 1])
            self.decoder[f"resize_{i}"] = Resize(num_channels[i], num_channels[i - 1], channels_c, scale=scales[i - 1],
                                                 transpose=True)
            # add ConvNext blocks
            self.encoder[f"blocks_{i}"] = nn.ModuleList(
                [ConvNext(num_channels[i], channels_c, k=k, r=r) for _ in range(num_encoder_blocks[i - 1])])
            self.decoder[f"blocks_{i}"] = nn.ModuleList(
                [ConvNext(num_channels[i], channels_c, k=k, r=r) for _ in range(num_decoder_blocks[i - 1])])

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            # initialise the weights of the convolutions
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        def _ada_init(module):
            # initialise the affine map to zero
            if "affine_map" in module._modules:
                nn.init.zeros_(module.affine_map[1].weight)
                if module.affine_map[1].bias is not None:
                    nn.init.zeros_(module.affine_map[1].bias)

        self.apply(_ada_init)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N]
        returns: output tensor [N, C_out, H, W]
        """
        c = self.conditioning(c)
        skips = []

        x = self.encoder["linear"](x)
        for i in range(1, self.num_stages):
            x = self.encoder[f"resize_{i}"](x, c=c)
            for block in self.encoder[f"blocks_{i}"]:
                x = block(x, c)
            skips.append(x)

        for i in reversed(range(1, self.num_stages)):
            x = skips[i - 1] if i == self.num_stages else x + skips[i - 1]  # no skip connection for the deepest stage
            for block in self.decoder[f"blocks_{i}"]:
                x = block(x, c)
            x = self.decoder[f"resize_{i}"](x, c=c)

        x = torch.stack([tail(x) for tail in self.decoder["linear"]], dim=1)  # ensemble tails
        return x