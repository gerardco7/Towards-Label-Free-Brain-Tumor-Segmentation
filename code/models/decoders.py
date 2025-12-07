from torch import nn
import torch
import numpy as np
from einops import rearrange
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd

from models.encoders import StackedConvBlocks, PlainConvEncoder

# ========== Convolutional Decoder ==========

# Key formula for output size of ConvTranspose2d
# O = S * (I - 1) + K - 2P

# First decoder
# 610.209 parameters
# 4 Convolution Transpose layers
class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, rescale = True):
        super(CNNDecoder, self).__init__()
        layers = [
            # In b, 8, 8, 8 
             nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=7, stride=3, padding=3),  # Output: (b, 128, 22, 22)
             nn.BatchNorm2d(128, affine=True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(128, 32, kernel_size=11, stride=3),  # Output: (b, 32, 74, 74)
             nn.BatchNorm2d(32, affine=True),
             nn.ReLU(True),   
             nn.ConvTranspose2d(32, 16, kernel_size=11, stride=3),  # Output: (b, 16, 233, 233)
             nn.BatchNorm2d(16, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(16, out_channels=out_channels, kernel_size=11, stride=1)  # Output: (b, 1, 240, 240)
        ]
        
        if rescale:
            layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
         x = x.view(x.shape[0], -1, 8, 8)
         # x.shape = (b, in_channels, 8, 8)
         recon = self.decoder(x)
         # recon.shape = (b, out_channels, 240, 240)
         return recon
    
# Second decoder
# 638.625 parameters
# 6 Convolution Transpose layers
class Large_CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, rescale = True):
        super(Large_CNNDecoder, self).__init__()

        layers = [
            # In b, 8, 8, 8 
             nn.ConvTranspose2d(in_channels=in_channels, out_channels=16, kernel_size=11, stride=2, padding=2),  # Output: (b, 16, 21, 21)
             nn.BatchNorm2d(16, affine=True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(16, 32, kernel_size=11, stride=2, padding=1),  # Output: (b, 32, 49, 49)
             nn.BatchNorm2d(32, affine=True),
             nn.ReLU(True),   
             nn.ConvTranspose2d(32, 64, kernel_size=11, stride=2, padding=1),  # Output: (b, 64, 105 , 105)
             nn.BatchNorm2d(64, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(64, 32, kernel_size=11, stride=2, padding=1),  # Output: (b, 32, 217, 217)
             nn.BatchNorm2d(32, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(32, 16, kernel_size=11, stride=1),  # Output: (b, 16, 227, 227)
             nn.BatchNorm2d(16, affine=True),
             nn.ReLU(True),
             nn.ConvTranspose2d(16, out_channels=out_channels, kernel_size=14, stride=1)  # Output: (b, 1, 240, 240)
        ]
        
        if rescale:
            layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
         x = x.view(x.shape[0], -1, 8, 8)
         # x.shape = (b, in_channels, 8, 8)
         recon = self.decoder(x)
         # recon.shape = (b, out_channels, 240, 240)
         return recon

# ========== ViT Decoder ==========
# Third decoder
class ViTDecoder(nn.Module):
    def __init__(self,
                 image_size: int = 240, 
                 embed_dim: int = 768, 
                 nhead: int = 12, 
                 num_layers: int = 2,
                 channels: int = 1,
                 patch_size: int = 24):
        super().__init__()

        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.channels = channels
        self.ph = self.pw = patch_size
        self.nh = self.nw = image_size // patch_size

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead), num_layers=num_layers
        )

        self.reconstruction_head = nn.Linear(embed_dim, patch_size * patch_size * channels)  

    def forward(self, x):
    
        decoded_tokens = self.transformer_decoder(x, x)
        r = self.reconstruction_head(decoded_tokens).view(x.shape[0], -1)
        r = rearrange(
            r, 'b (nh nw c ph pw) -> b c (nh ph) (nw pw)',
            nh=self.nh, nw=self.nw, ph=self.ph, pw=self.pw, c=self.channels
        )

        return r
    
# ========== ResnetDecoder ==========
'https://github.com/jusiro/constrained_anomaly_segmentation/blob/main/code/models/models.py'

class ResnetDecoder(torch.nn.Module):

    def __init__(self, fin=256, nf0=128, n_channels=1, dense=False, n_blocks=4, spatial_dim=7, rescale=False):
        super(ResnetDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.dense = dense
        self.spatial_dim = spatial_dim
        self.fin = fin
        self.rescale = rescale

        if self.dense:
            self.dense = torch.nn.Linear(fin, fin*spatial_dim**2)

        # Set number of input and output channels
        n_filters_in = [fin] + [nf0//2**i for i in range(0, self.n_blocks)]
        n_filters_out = [nf0//2**(i-1) for i in range(1, self.n_blocks+1)] + [n_channels]

        self.blocks = torch.nn.ModuleList()
        for i in np.arange(0, self.n_blocks):
            self.blocks.append(ResBlock(n_filters_in[i], n_filters_out[i]))
        self.out = torch.nn.Conv2d(n_filters_in[-1], n_filters_out[-1], kernel_size=(3, 3), padding=(1, 1))
        
        if self.rescale:
            self.tanh = torch.nn.Tanh()

    def forward(self, x):

        if self.dense:
            x = self.dense(x)
            x = torch.nn.Unflatten(-1, (self.fin, self.spatial_dim, self.spatial_dim))(x)
            # [b, z_dim, spatial_dim, spatial_dim]

        for i in np.arange(0, self.n_blocks):
            x = self.blocks[i](x)
        f = x
        out = self.out(f)

        # Apply Tanh if rescale is True
        if self.rescale:
            out = self.tanh(out)

        # [b, n_channels, spatial_dim, spatial_dim]

        return out


class ResBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(ResBlock, self).__init__()
        self.conv_straight_1 = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_1 = torch.nn.BatchNorm2d(fout)
        self.conv_straight_2 = torch.nn.Conv2d(fout, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_2 = torch.nn.BatchNorm2d(fout)
        self.conv_skip = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.upsampling = torch.nn.Upsample(scale_factor=(2, 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x_st = self.upsampling(x)
        x_st = self.conv_straight_1(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_1(x_st)
        x_st = self.conv_straight_2(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_2(x_st)

        x_sk = self.upsampling(x)
        x_sk = self.conv_skip(x_sk)

        out = x_sk + x_st

        return out


# ========== UNetDecoder ==========
'https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/unet_decoder.py'

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
    

def get_matching_convtransp(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.ConvTranspose1d
    elif dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
