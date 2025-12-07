import torch
import os
from torch.optim import Adam

from models.model import Model
from models.encoders import ViTEncoder, CNN_Encoder, ResnetEncoder, PlainConvEncoder
from models.decoders import CNNDecoder, Large_CNNDecoder, ViTDecoder, ResnetDecoder, UNetDecoder
from methods.trainers.train import Trainer
from models.get_network_from_plans import get_network_from_plans


class UnsupervisedAnomalyDetectorModels:
    def __init__(self, model_name, model_code, device, checkpoint, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, 
                 dense=False, n_blocks=4, spatial_dim=7, gap=False, channels=1, z_dim=256, rescale=True, is_training=True,
                 optimizer='adam', input_shape=(1, 240, 240), VAE=False, aggregate_across_patches=False, resnet='18', Rec_Loss='l1', SSIM_Loss=False, SSIM_weight=0, Kl_Loss=False, grad_cam=False, alpha_ae=1, 
                 p_activation_cam=0.5, t=0.5, expansion_loss_penalty=0.1, pre_training_epochs=10, level_cams=3, nnUnet=False):
        
        self.model_name = model_name
        self.model_code = model_code
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.decoder = decoder
        self.fuse_layer = fuse_layer

        # Initialize model hyperparameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.z_dim = z_dim
        self.rescale = rescale
        self.VAE = VAE
        self.aggregate_across_patches = aggregate_across_patches
        self.dense = dense
        self.resnet = resnet
        self.n_blocks = n_blocks
        self.spatial_dim = spatial_dim
        self.gap = gap
        self.nnUnet = nnUnet

        # Initialize trainer arguments
        self.is_training = is_training
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.Rec_Loss = Rec_Loss
        self.SSIM_Loss = SSIM_Loss
        self.SSIM_weight = SSIM_weight
        self.Kl_Loss = Kl_Loss

        # Initialize Grad-CAM arguments
        self.grad_cam = grad_cam
        self.alpha_ae = alpha_ae
        self.p_activation_cam = p_activation_cam
        self.t = t
        self.expansion_loss_penalty = expansion_loss_penalty
        self.pre_training_epochs = pre_training_epochs
        self.level_cams = level_cams
        

    ###############################################################################
    #                                AE Models
    ###############################################################################
        if self.model_name == 'AE':
            # Basic Autoencoder
            if self.model_code == '1.1.1':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                            image_size=240, z_dim=512)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # Autoencoder with SSIM loss
            elif self.model_code == '1.2.1.1':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device,  is_training=self.is_training, 
                            image_size=240, z_dim=512)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False
            
            # Autoencoder with SSIM loss and weight 10
            elif self.model_code == '1.2.1.1.1' or self.model_code == '1.0.2.3':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                            image_size=240, z_dim=512)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 10
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False
            
            # Autoencoder with SSIM loss and weight 100
            elif self.model_code == '1.2.1.1.2':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                            image_size=240, z_dim=512)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # Autoencoder with SSIM loss and weight 10 and not scaling the input
            elif self.model_code == '1.0.2.3.1':    
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                            image_size=240, z_dim=512, rescale=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 10
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # Autoencoder with SSIM loss and weight 100
            elif self.model_code == '1.2.1.1.2':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                            image_size=240, z_dim=512)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False
            
    ###############################################################################
    #                                VAE Models
    ###############################################################################        

        elif self.model_name == 'VAE':
            # Basic VAE
            if self.model_code == '1.1.3':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = False

            # VAE with ResNet
            elif self.model_code == '1.1.3.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                 image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = False

            # VAE with SSIM loss
            if self.model_code == '1.2.1.2':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = False

            # VAE with SSIM loss
            if self.model_code == '1.2.1.2':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = False

            # VAE with KL_weight 1/10
            if self.model_code == '1.2.3.1':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1/10
                self.grad_cam = False

            # VAE with KL_weight 1/100
            if self.model_code == '1.2.3.2':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1/100
                self.grad_cam = False

            # VAE with KL_weight 1/1000
            if self.model_code == '1.2.3.3':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1/1000
                self.grad_cam = False

            # VAE with KL_weight 1/1000 0
            if self.model_code == '1.2.3.4':
                self.model = Model(self.model_name, encoder=CNN_Encoder, decoder=CNNDecoder, fuse_layer=False, device=self.device, is_training=self.is_training, 
                                image_size=240, z_dim=512, VAE=True)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1/10000
                self.grad_cam = False


    ###############################################################################
    #                               ResNet-AE models
    ###############################################################################

        elif self.model_name == 'ResNet-AE': 
            # Basic ResNet-AE
            if self.model_code == '1.1.2':   
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False
            
            # ResNet-AE with SSIM loss
            if self.model_code ==  '1.2.1.3' or self.model_code == '1.0.2.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ResNet-AE with SSIM loss and weight 10
            if self.model_code ==  '1.2.1.3.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 10
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ResNet-AE with SSIM loss and weight 100
            if self.model_code ==  '1.2.1.3.2' or self.model_code == '1.4.1.2' or self.model_code == '1.0.4.2' or self.model_code == '1.0.4.2.1' or self.model_code == '1.0.2.4' or self.model_code == '1.0.2.4.1' or self.model_code == '1.5.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ResNet-AE with SSIM loss and weight 100
            if self.model_code ==  '1.2.1.3.3':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1000
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ResNet-AE with SSIM loss and scaling the input [-1, 1]
            if self.model_code == '1.0.2.1.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, rescale=True, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ResNet-AE with SSIM loss weight 100, no scaling the input [-1, 1]
            if  self.model_code == '1.0.2.4.2' or self.model_code == '1.0.2.4.3' or self.model_code == '1.9.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, rescale=False, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False
        
            # ResNet-AE 34 with SSIM loss weight 100 
            if self.model_code == '1.1.2.1':
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, resnet='34', dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=False)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False


    ###############################################################################
    #                                ViT-AE Models
    ###############################################################################


        elif self.model_name == 'ViT-AE':
            # Basic ViT-AE
            if self.model_code == '1.1.5':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with SSIM loss
            if self.model_code == '1.2.1.4' or self.model_code == '1.0.2.2':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with SSIM loss and weight 10
            if self.model_code == '1.2.1.4.1':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 10
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with SSIM loss and weight 100
            if self.model_code == '1.2.1.4.2':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with SSIM loss and not scaling the input
            if self.model_code == '1.0.2.2.1':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, rescale=False,
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 1
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with encoder dim 768 SSIM loss and weight 100
            if self.model_code == '1.1.5.1':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=768, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE with encoder dim 768, head 12, depth 12 SSIM loss and weight 100
            if self.model_code == '1.1.5.2':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=768, depth=12, heads=12, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

             # ViT-AE with encoder dim 384, head 8, depth 6 SSIM loss and weight 100
            if self.model_code == '1.1.5.3':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=384, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE multimodal with SSIM loss and weight 100
            if self.model_code == '1.4.2.1' or self.model_code == '1.4.2.1.2':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=4, z_dim=512, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False


            # ViT-AE multimodal with SSIM loss and weight 100, z_dim 1024
            if self.model_code == '1.4.2.1.1':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=4, z_dim=1024, 
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

            # ViT-AE multimodal with SSIM loss and weight 100 and not scaling the input
            if self.model_code == '1.4.2.1.3' :
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=4, z_dim=512, rescale=False,
                            VAE=False, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = True
                self.SSIM_weight = 100
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False



    ###############################################################################
    #                                ViT-VAE Models
    ###############################################################################
        
        elif self.model_name == 'ViT-VAE':  
            # Basic ViT-VAE
            if self.model_code == '1.1.6':
                self.model = Model(self.model_name, encoder=ViTEncoder, decoder=Large_CNNDecoder, fuse_layer=True, device=self.device, is_training=self.is_training, 
                            image_size=240, patch_size=24, encoder_dim=512, depth=6, heads=8, mlp_dim=1024, channels=1, z_dim=512, 
                            VAE=True, aggregate_across_patches=False, multimodal=False, grad_cam=False)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = False

    ###############################################################################
    #                               GradCAMCons Models
    ###############################################################################

        elif self.model_name == 'GradCAMCons':
            if self.model_code == '1.1.7':   
                self.model = Model(self.model_name, encoder=ResnetEncoder, decoder=ResnetDecoder, fuse_layer=False, device=self.device, is_training=self.is_training,
                                image_size=240, encoder_dim=512, z_dim=128, dense=True, n_blocks=4, spatial_dim=self.image_size//2**4, gap=False, channels=1, VAE=True)

                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = True
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = True
                self.Kl_weight = 1
                self.grad_cam = True
                self.alpha_ae = 1
                self.p_activation_cam = 0.2
                self.t = 25
                self.expansion_loss_penalty = 'l2'
                self.pre_training_epochs = 0
                self.level_cams = -4
    ###############################################################################
    #                                nnU-Net
    ###############################################################################
        elif self.model_name == 'nnU-Net':
            if self.model_code == '1.8.1.1':
                self.model = get_network_from_plans(plans_path="D:/models/nnUnet_results/Dataset001_BraTSMEN/nnUNetTrainer__nnUNetPlans__2d/plans.json").to(self.device)
                
                self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
                self.input_shape = (1, 240, 240)
                self.VAE = False
                self.Rec_Loss = 'l2'
                self.SSIM_Loss = False
                self.SSIM_weight = 0
                self.Kl_Loss = False
                self.Kl_weight = 0
                self.grad_cam = False

    ###############################################################################
    #                                Other Models
    ###############################################################################
        else:
            self.model = Model(self.model_name, encoder=self.encoder, decoder=self.decoder, fuse_layer=self.fuse_layer, device=self.device, is_training=self.is_training, 
                          image_size=self.image_size, patch_size=self.patch_size, encoder_dim=self.encoder_dim, depth=self.depth, heads=self.heads, mlp_dim=self.mlp_dim, 
                          dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.spatial_dim, gap=self.gap,
                          channels=self.channels, z_dim=self.z_dim, VAE=self.VAE, aggregate_across_patches=self.aggregate_across_patches, multimodal=False, grad_cam=self.grad_cam)

        # Print model architecture
        print("\nModel architecture:")
        print(self.model)

        # Print model parameters
        print("\nModel number parameters:")
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if self.checkpoint:
            checkpoint_name = self.checkpoint.split('/')[-1]
            checkpoint_code = checkpoint_name.split('-')[-2]
            assert checkpoint_code == self.model_code, f"Model code {self.model_code} does not match checkpoint code {checkpoint_code}."
            if self.model_name == 'nnU-Net':
                self.model.load_state_dict(torch.load(self.checkpoint, weights_only=False)['network_weights'])
            else:
                self.model.load_state_dict(torch.load(self.checkpoint, weights_only=True))
            print(f"\nModel loaded from checkpoint: {self.checkpoint}")

        # Set trainer
        self.trainer = Trainer(model=self.model, model_name=self.model_name, model_code=self.model_code, optimizer=self.optimizer, device=self.device, input_shape=self.input_shape, 
                               VAE=self.VAE, Rec_Loss=self.Rec_Loss, SSIM_Loss=self.SSIM_Loss, SSIM_weight=self.SSIM_weight, Kl_Loss=self.Kl_Loss, KL_weight=self.Kl_weight,
                               grad_cam=self.grad_cam, alpha_ae=self.alpha_ae, p_activation_cam=self.p_activation_cam, t=self.t, 
                               expansion_loss_penalty=self.expansion_loss_penalty, pre_training_epochs=self.pre_training_epochs, 
                               level_cams=self.level_cams)
    
    def train(self, train_loader, valid_loader, epochs):
        self.trainer.train_model(train_loader, valid_loader, epochs)

    def get_model(self):
        return self.model   
    
    def state_dict(self):
        return self.model.state_dict()
    
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")