import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
from einops import rearrange, repeat
from torch import nn
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


# ========== Vision Transformer Encoder ==========
'VTL-AE paper: https://arxiv.org/abs/2106.06716'

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            # nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, z_dim = 512, VAE=False, aggregate_across_patches=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.VAE = VAE
        self.aggregate_across_patches = aggregate_across_patches

        if self.VAE:
            self.mu = nn.Linear(dim, z_dim)
            self.logvar = nn.Linear(dim, z_dim)

    def forward(self, img, mask = None):
        # img shape: (b, c, H, W)
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # x shape: (b, num_patches, patch_size * patch_size * channels)
        x = self.patch_to_embedding(x)
        # x shape: (b, num_patches, dim)
        b, n, _ = x.shape
        # b: batch size, n: number of patches, _: dim

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # cls_tokens shape: (b, 1, dim)

        x = torch.cat((cls_tokens, x), dim=1)
        # x shape: (1b num_patches + 1, dim)
        x += self.pos_embedding[:, :(n + 1)]
        # x shape: (b, num_patches + 1, dim)

        x = self.transformer(x, mask)
        # x shape: (b, num_patches + 1, dim)

        x = self.to_cls_token(x[:,1:,:])
        # x shape: (b, num_patches, dim)
       
        if self.VAE:
            # Agregate across patches
            if self.aggregate_across_patches:
                mu = self.mu(x.mean(dim=1))
                logvar = self.logvar(x.mean(dim=1))
            else:
                mu = self.mu(x)
                logvar = self.logvar(x)
            z = None
        else:
            z = x
            mu, logvar = None, None
        
        return z, mu, logvar, None



# ========== Convolutional Encoder ==========
'https://github.com/dariocazzani/pytorch-AE/blob/master/architectures.py'

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 240, 240), VAE=False):
        super(CNN_Encoder, self).__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.VAE = VAE
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=1,
                     padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        if VAE:
            self.mu = nn.Linear(self.flat_fts, self.output_size)
            self.logvar = nn.Linear(self.flat_fts, self.output_size)
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.flat_fts, self.output_size),
                nn.BatchNorm1d(self.output_size),
                nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x, mask=None):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        if self.VAE:
            mu = self.mu(x)
            z_logvar = self.logvar(x)
            z = None
        else:
            z = self.linear(x)
            mu, z_logvar = None, None
    
        return z, mu, z_logvar, None
        

# ========== Convolutional Encoder (ResNet Backbone) ==========
'https://github.com/jusiro/constrained_anomaly_segmentation/blob/main/code/models/models.py'

class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_blocks=4, resnet='18'):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2**(4-n_blocks))

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)
        if resnet == '18':
            resnet_model = torchvision.models.resnet18(pretrained=False)
        elif resnet == '34':
            resnet_model = torchvision.models.resnet34(pretrained=False)
        self.resnet = torch.nn.Sequential(*(list(resnet_model.children())[i+4] for i in range(0, self.n_blocks)))

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        x = self.input(x)
        F = []
        for iBlock in range(0, self.n_blocks):
            x = list(self.resnet.children())[iBlock](x)
            F.append(x)

        return x, F


class ResnetEncoder(torch.nn.Module):
    def __init__(self, fin=1, z_dim=128, dense=False, VAE=False, n_blocks=4, spatial_dim=7, gap=False, resnet='18'):
        super(ResnetEncoder, self).__init__()
        self.fin = fin
        self.z_dim = z_dim
        self.dense = dense
        self.n_blocks = n_blocks
        self.gap = gap
        self.VAE = VAE
        self.resnet = resnet

        # 1) Feature extraction
        self.backbone = Resnet(in_channels=self.fin, n_blocks=self.n_blocks, resnet=self.resnet)
        # 2) Latent space (dense or spatial)
        if self.dense:  # dense
            if gap:
                if self.VAE:
                    self.mu = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))
                    self.log_var = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))
                else:
                    self.z = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))
            else:
                if self.VAE:
                    self.mu = torch.nn.Linear(self.backbone.nfeats*spatial_dim**2, z_dim)
                    self.log_var = torch.nn.Linear(self.backbone.nfeats*spatial_dim**2, z_dim)
                else:
                    self.z = torch.nn.Linear(self.backbone.nfeats * spatial_dim ** 2, z_dim)
        else:  # spatial
            if self.VAE:
                self.mu = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))
                self.log_var = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))
            else:
                self.z = torch.nn.Conv2d(self.backbone.nfeats, z_dim, (1, 1))

    def forward(self, x, mask=None):
        # [b, c, h, w]

        # 1) Feature extraction
        x, allF = self.backbone(x)
        # [b, 512, spatial_dim, spatial_dim]

        if self.dense and not self.gap:
            x = torch.nn.Flatten()(x)
            # [b, 512 * spatial_dim * spatial_dim]

        if self.dense and self.gap:
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), self.backbone.nfeats, 1, 1)
            # [b, 512, 1, 1]

        # 2) Latent space
        if self.VAE:
            z_mu = self.mu(x)
            z_logvar = self.log_var(x)
            z = None
        else:
            z = self.z(x)
            z_mu, z_logvar = None, None

        # [b, z_dim]
        return z, z_mu, z_logvar, allF
    


# ========== Convolutional Encoder (nnUNet) ==========
'''https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/plain_conv_encoder.py#L12'''

def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if issubclass(conv_op , nn.Conv2d):
            return [scalar] * 2
        elif issubclass(conv_op , nn.Conv3d):
            return [scalar] * 3
        elif issubclass(conv_op , nn.Conv1d):
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar


def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if issubclass(conv_op, nn.Conv1d):
        return 1
    elif issubclass(conv_op, nn.Conv2d):
        return 2
    elif issubclass(conv_op, nn.Conv3d):
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))
    

def get_matching_pool_op(conv_op: Type[_ConvNd] = None,
                         dimension: int = None,
                         adaptive=False,
                         pool_type: str = 'avg') -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if dimension == 1:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dimension == 2:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dimension == 3:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d
    

class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)
    

class StackedConvBlocks(nn.Module):
    def __init__(self,
                num_convs: int,
                conv_op: Type[_ConvNd],
                input_channels: int,
                output_channels: Union[int, List[int], Tuple[int, ...]],
                kernel_size: Union[int, List[int], Tuple[int, ...]],
                initial_stride: Union[int, List[int], Tuple[int, ...]],
                conv_bias: bool = False,
                norm_op: Union[None, Type[nn.Module]] = None,
                norm_op_kwargs: dict = None,
                dropout_op: Union[None, Type[_DropoutNd]] = None,
                dropout_op_kwargs: dict = None,
                nonlin: Union[None, Type[torch.nn.Module]] = None,
                nonlin_kwargs: dict = None,
                nonlin_first: bool = False
                ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
