import torch 
import torch.nn as nn
import os 

from methods.utils import initialize_weights, add_noise
from models.encoders import ViTEncoder, CNN_Encoder, ResnetEncoder, PlainConvEncoder
from models.decoders import CNNDecoder, Large_CNNDecoder, ViTDecoder, ResnetDecoder, UNetDecoder

# Unsupervised Anomaly Detector Model class
class Model(nn.Module):  
    def __init__(self, model_name, encoder, decoder, fuse_layer, device, is_training=False, image_size=240,
                patch_size=240, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, dense=False, n_blocks=4, spatial_dim=7, gap=False,
                channels=1, z_dim=512, resnet='18', rescale=True, VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False, 
                ):
       
        super(Model, self).__init__()
        
        # Initialize model hyperparameters
        self.model_name = model_name
        self.encoder = encoder
        self.decoder = decoder
        self.fuse_layer = fuse_layer
        self.device = device
        self.channels = channels
        self.is_training = is_training
        self.image_size = image_size
        self.patch_size = patch_size
        self.multimodal = multimodal
        self.VAE = VAE
        self.aggregate_across_patches = aggregate_across_patches
        self.resnet = resnet
        self.grad_cam = grad_cam

        # Initialize encoder parameters
        self.e_dim = encoder_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dense = dense
        self.n_blocks = n_blocks
        self.spatial_dim = spatial_dim
        self.gap = gap

        # Initialize decoder parameters
        self.z_dim = z_dim
        self.rescale = rescale

        # Initialize the model
        encoder_args = {
            ViTEncoder: {'image_size': self.image_size, 'patch_size': self.patch_size, 'dim': self.e_dim, 'depth': self.depth, 'heads': self.heads, 'mlp_dim': self.mlp_dim, 'channels': self.channels, 'z_dim': self.z_dim, 'VAE': self.VAE, 'aggregate_across_patches': self.aggregate_across_patches},
            CNN_Encoder: {'output_size': self.e_dim, 'VAE': self.VAE},
            ResnetEncoder: {'fin': self.channels, 'z_dim': self.z_dim, 'dense': self.dense, 'VAE': self.VAE, 'n_blocks': self.n_blocks, 'spatial_dim': self.spatial_dim, 'gap': self.gap, 'resnet': self.resnet},
        }

        if self.encoder in encoder_args:
            self.Encoder = self.encoder(**encoder_args[self.encoder]).to(self.device)
        else:
            raise ValueError("Encoder not supported")

        # Initialize the decoder
        decoder_args = {
            CNNDecoder: {'in_channels': self.z_dim // 64, 'out_channels': self.channels, 'rescale': self.rescale},  
            Large_CNNDecoder: {'in_channels': self.z_dim // 64, 'out_channels': self.channels, 'rescale': self.rescale},
            ViTDecoder: {'image_size': self.image_size, 'embed_dim': self.z_dim, 'nhead': self.heads, 'num_layers': self.depth, 'channels': self.channels, 'patch_size': self.patch_size},
            ResnetDecoder: {'fin': self.z_dim, 'nf0': self.z_dim//2, 'n_channels': self.channels, 'dense': self.dense, 'n_blocks': self.n_blocks, 'spatial_dim': self.spatial_dim, 'rescale': self.rescale},
        }
        
        if self.multimodal:
            if self.decoder in decoder_args:
                self.Decoders = nn.ModuleList([self.decoder(**decoder_args[self.decoder]).to(self.device) for _ in range(self.channels)])
            else:
                raise ValueError("Decoder not supported")
        else:
            if self.decoder in decoder_args:
                self.Decoder = self.decoder(**decoder_args[self.decoder]).to(self.device)
            else:
                raise ValueError("Decoder not supported")
            
        # Initialize the weights
        if self.is_training:
            print("\nInitializing network weights.........")
            if self.multimodal:
                initialize_weights(self.Encoder, *self.Decoders)
            else:
                initialize_weights(self.Encoder, self.Decoder)

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.mask = torch.ones(1, self.image_size//self.patch_size, self.image_size//self.patch_size).bool().to(self.device)

        if self.fuse_layer:
            if self.VAE:
                if self.aggregate_across_patches:
                    self.linear = nn.Linear(self.z_dim, self.z_dim).to(self.device)
                else:
                    self.linear = nn.Linear(self.num_patches*self.z_dim, self.z_dim).to(self.device)
            else:
                self.linear = nn.Linear(self.num_patches*self.e_dim, self.z_dim).to(self.device)


    def forward(self, x):
        b = x.shape[0] # batch size
        x = x.to(self.device)
        # Forward pass

        #TODO: implement allF encoder extraction
        z, mu, log_var, allF= self.Encoder(x, mask=self.mask)

        if self.VAE: 
            z = self.reparameterize(mu, log_var)
    
        else: 
            if self.is_training:
                z = add_noise(z)

        if self.fuse_layer:
            z = z.view(b, -1)
            z = self.linear(z)
        
        if self.multimodal:
            z = torch.chunk(z, 4, dim=1)
            z = [decoder(z[i]) for i, decoder in enumerate(self.Decoders)]
            x = torch.cat(z, dim=1)
        else:     
            x = self.Decoder(z)

        return x, mu, log_var, allF


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + (eps*std)