import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import time

from methods.losses.losses import kl_loss, ssim_loss, log_barrier
from methods.utils import grad_cam, recycle_bin

class Trainer:
    def __init__(self, model, model_name, model_code, optimizer, device, input_shape, VAE, Rec_Loss, SSIM_Loss, SSIM_weight, Kl_Loss, KL_weight, grad_cam, alpha_ae, p_activation_cam, t, expansion_loss_penalty, pre_training_epochs, level_cams):
        #Initialize trainer hyperparameters
        self.model = model
        self.model_name = model_name    
        self.model_code = model_code
        self.optimizer = optimizer
        
        self.device = device
        self.input_shape = input_shape
        self.VAE = VAE

        # Initialize trainer losses    
        self.Rec_Loss = Rec_Loss
        self.SSIM_Loss = SSIM_Loss
        self.SSIM_weight = SSIM_weight
        self.Kl_Loss = Kl_Loss
        self.KL_weight = KL_weight

        # Initialize Grad-CAM arguments
        self.grad_cam = grad_cam
        self.alpha_ae = alpha_ae
        self.p_activation_cam = p_activation_cam
        self.t = t
        self.expansion_loss_penalty = expansion_loss_penalty
        self.pre_training_epochs = pre_training_epochs
        self.level_cams = level_cams

        # Set loss functions
        if Rec_Loss == 'l1':
            self.Rec_Loss = nn.L1Loss()
        elif Rec_Loss == 'l2':
            self.Rec_Loss = nn.MSELoss()
        elif Rec_Loss == 'bce':
            self.Rec_Loss = nn.BCELoss()
        
        if SSIM_Loss:
            self.SSIM_Loss = ssim_loss
        
        if Kl_Loss:
            self.Kl_Loss = kl_loss

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, valid_loader, epochs):  
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.iterations = len(self.train_loader)

        # Initialize best loss
        self.best_loss = np.inf

        # Initialize losses
        rec_l = torch.tensor(0.0)
        ssim_l = torch.tensor(0.0)
        kl_l = torch.tensor(0.0)
        lae = torch.tensor(0.0)

        # Initialize old epoch
        self.old_epoch = 0  

        for self.i_epoch in range(self.epochs):

            self.Rec_L = []
            self.SSIM_L = []
            self.KL_L = []
            self.LAE_L = []
            self.L = []
            self.V_Rec_L = []
            self.V_SSIM_L = []
            self.V_KL_L = []
            self.V_LAE_L = []
            self.V_L = []

            start_time = time.time()

            for self.i_batch, x in enumerate(self.train_loader):

                # Move data to device
                x = x.to(self.device)

                loss = 0
                # Forward pass
                x_hat, mu, log_var, allF = self.forward(x)

                # Compute the loss
                rec_l = self.Rec_Loss(x_hat, x) / (self.train_loader.batch_size)
                loss += rec_l
                if self.SSIM_Loss:
                    ssim_l = - self.SSIM_Loss(x_hat, x) / (self.train_loader.batch_size)
                    loss += ssim_l * self.SSIM_weight
                if self.Kl_Loss:
                    kl_l = self.Kl_Loss(mu, log_var) / (self.train_loader.batch_size)
                    loss += kl_l * self.KL_weight 

                
                if self.grad_cam:
                    # Compute grad-cams
                    gcam = grad_cam(allF[self.level_cams], torch.sum(mu), normalization='sigm',
                                    avg_grads=True)
                    
                    gcam = torch.nn.functional.interpolate(gcam.unsqueeze(1),
                                                            size=(self.input_shape[-1], self.input_shape[-1]),
                                                            mode='bilinear',
                                                            align_corners=True).squeeze()

                    self.lae_iteration = torch.mean(gcam)

                    if self.i_epoch >= self.pre_training_epochs:
                        if self.expansion_loss_penalty == 'l1':  # L1
                            lae = torch.mean(torch.abs(-torch.mean(gcam, (-1)) + 1 - self.p_activation_cam))
                        elif self.expansion_loss_penalty == 'l2':  # L2
                            lae = torch.mean(torch.sqrt(torch.pow(-torch.mean(gcam, (-1)) + 1 - self.p_activation_cam, 2)))
                        elif self.expansion_loss_penalty == 'log_barrier':
                            z = -torch.mean(gcam, (1, 2)).unsqueeze(-1) + 1
                            lae = log_barrier(z - self.p_activation_cam, t=self.t) / self.train_loader.batch_size
                        loss += self.alpha_ae * lae.squeeze()

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Save losses
                self.Rec_L.append(rec_l.item())
                self.SSIM_L.append(ssim_l.item())
                self.KL_L.append(kl_l.item())
                self.LAE_L.append(lae.item())
                self.L.append(loss.item())

                # Print batch 
                if self.i_batch % 100 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    start_time = end_time
                    print(f"Epoch: {self.i_epoch}, Batch: {self.i_batch}, Loss: {loss.item()}, Rec Loss: {rec_l.item()}, SSIM Loss: {ssim_l.item()}, KL Loss: {kl_l.item()}, LAE Loss: {self.alpha_ae*lae.item()}, Time: {elapsed_time}")
                    #wandb.log({'GCAM': [wandb.Image(gcam[0].cpu().detach().numpy())]}) 
                    if x_hat.shape[1] == 4:
                        wandb.log({'image t1c': [wandb.Image(x[0][0].cpu().detach().numpy())]})
                        wandb.log({'reconstruction t1c': [wandb.Image(x_hat[0][0].cpu().detach().numpy())]}) 
                        wandb.log({'image t1c': [wandb.Image(x[0][1].cpu().detach().numpy())]})
                        wandb.log({'reconstruction t1c': [wandb.Image(x_hat[0][1].cpu().detach().numpy())]}) 
                        wandb.log({'image t1c': [wandb.Image(x[0][2].cpu().detach().numpy())]})
                        wandb.log({'reconstruction t1c': [wandb.Image(x_hat[0][2].cpu().detach().numpy())]}) 
                        wandb.log({'image t1c': [wandb.Image(x[0][3].cpu().detach().numpy())]})
                        wandb.log({'reconstruction t1c': [wandb.Image(x_hat[0][3].cpu().detach().numpy())]}) 
                    else:
                        wandb.log({'image t1c': [wandb.Image(x[0].cpu().detach().numpy())]})
                        wandb.log({'reconstruction t1c': [wandb.Image(x_hat[0].cpu().detach().numpy())]})


            for self.i_batch, x in enumerate(self.valid_loader):
                loss = 0
                with torch.no_grad():
                    # Move data to device
                    x = x.to(self.device)

                    # Forward pass
                    x_hat, mu, log_var, allF = self.forward(x)

                    # Compute the loss
                    rec_l = self.Rec_Loss(x_hat, x) / (self.train_loader.batch_size)
                    loss += rec_l
                    if self.SSIM_Loss:
                        ssim_l = - self.SSIM_Loss(x_hat, x) / (self.train_loader.batch_size)
                        loss += ssim_l * self.SSIM_weight
                    if self.Kl_Loss:    
                        kl_l = self.Kl_Loss(mu, log_var) / (self.train_loader.batch_size)
                        loss += kl_l * self.KL_weight    

                    # Print batch 
                    if self.i_batch % 100 == 0: 
                        print(f"Epoch: {self.i_epoch}, Batch: {self.i_batch}, Validation Loss: {loss.item()}, Rec Loss: {rec_l.item()}, SSIM Loss: {ssim_l.item()}, KL Loss: {kl_l.item()}, LAE Loss: {self.alpha_ae*lae.item()}")

                    # Save losses
                    self.V_Rec_L.append(rec_l.item())
                    self.V_SSIM_L.append(ssim_l.item())
                    self.V_KL_L.append(kl_l.item())
                    self.V_LAE_L.append(lae.item())
                    self.V_L.append(loss.item())
                
            # Epoch log
            print(f'Epoch {self.i_epoch} Trainning loss: {np.mean(self.L)} Validation loss: {np.mean(self.V_L)}')
            wandb.log({'Trainning loss': np.mean(self.L), 'Validation loss': np.mean(self.V_L), 
                       'Rec Loss': np.mean(self.Rec_L) if self.Rec_L else float('nan'), 
                       'SSIM Loss': np.mean(self.SSIM_L) if self.SSIM_L else float('nan'), 
                       'KL Loss': np.mean(self.KL_L) if self.KL_L else float('nan'), 
                       'LAE Loss': np.mean(self.LAE_L) if self.LAE_L else float('nan'), 
                       'V_Rec Loss': np.mean(self.V_Rec_L) if self.V_Rec_L else float('nan'), 
                       'V_SSIM Loss': np.mean(self.V_SSIM_L) if self.V_SSIM_L else float('nan'), 
                       'V_KL Loss': np.mean(self.V_KL_L) if self.V_KL_L else float('nan'), 
                       'V_LAE Loss': np.mean(self.V_LAE_L) if self.V_LAE_L else float('nan')})   

            # Save model
            if np.mean(self.V_L) < self.best_loss:
                self.best_loss = np.mean(self.V_L)
                os.makedirs('./saved_model', exist_ok=True)

                # Remove old model
                previous_model_pth = f'./saved_model/{self.model_name}-{self.model_code}-ep{str(self.old_epoch)}.pt'
                if os.path.exists(previous_model_pth):
                    os.remove(previous_model_pth)
                    recycle_bin()   
                
                # Save new model
                torch.save(self.model.state_dict(), f'./saved_model/{self.model_name}-{self.model_code}-ep{str(self.i_epoch)}.pt')
                self.old_epoch = self.i_epoch
