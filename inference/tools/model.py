import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


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
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels):
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

        z = self.to_cls_token(x[:,1:,:])
        # z shape: (b, num_patches, dim)
        
        return z
    

# ========== CNN Decoder ==========
class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNDecoder, self).__init__()

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
             nn.ConvTranspose2d(16, out_channels=out_channels, kernel_size=14, stride=1)  # Output: (b, out_channels, 240, 240)
        ]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
         x = x.view(x.shape[0], -1, 8, 8)
         # x.shape = (b, in_channels, 8, 8)
         recon = self.decoder(x)
         # recon.shape = (b, out_channels, 240, 240)
         return recon
    

# ========== Unsupervised Anomaly Detector Model ==========
class Model(nn.Module):  
    def __init__(self, channels, image_size, patch_size, encoder_dim, depth, heads, mlp_dim, z_dim, device):
       
        super(Model, self).__init__()
        
        # Device assignment
        self.device = device

        # Initialize model hyperparameters
        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size

        # Initialize encoder parameters
        self.e_dim = encoder_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

        # Initialize decoder parameters
        self.z_dim = z_dim

        # Initialize the Encoder
        self.Encoder = ViTEncoder(image_size=self.image_size,
                                    patch_size=self.patch_size,
                                    dim=self.e_dim,
                                    depth=self.depth,
                                    heads=self.heads,
                                    mlp_dim=self.mlp_dim,
                                    channels=self.channels,
                                ).to(self.device)
 
        # Initialize the Decoder
        self.Decoder = CNNDecoder(in_channels=self.z_dim // 64, 
                                    out_channels=self.channels, 
                                ).to(self.device)

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.mask = torch.ones(1, self.image_size//self.patch_size, self.image_size//self.patch_size).bool().to(self.device)

        # Initialize fuse layer
        self.linear = nn.Linear(self.num_patches*self.e_dim, self.z_dim).to(self.device)


    def forward(self, x):
        b = x.shape[0] # batch size
        x = x.to(self.device)

        # Encoder forward pass
        z = self.Encoder(x, mask=self.mask)

        # Fuse layer 
        z = z.view(b, -1)
        z = self.linear(z)
   
        # Decoder forward pass
        x = self.Decoder(z)

        return x
    
    def get_model_from_checkpoint(self, checkpoint_path):
        """
        Load the model from a checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, 
                                map_location=self.device,  
                                weights_only=True
                                )   
        self.load_state_dict(checkpoint)
        self.eval()
        return self
