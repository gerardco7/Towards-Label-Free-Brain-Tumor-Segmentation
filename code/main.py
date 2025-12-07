"""
Main training script for unsupervised anomaly detection models.

This script:
1) Parses command-line arguments
2) Loads training/validation datasets
3) Instantiates the chosen model architecture
4) Trains the model
5) Saves checkpoints and experiment metadata

Models are defined in `models/models.py`
Datasets are loaded via `methods/datasets/data.py`
"""

import argparse
import wandb
import json
import os
import torch

from methods.datasets.data import loadData
from models.models import UnsupervisedAnomalyDetectorModels


"python3 main.py --model_name ' ' --model_code '' "
# TODO: implement synthesis generation

# TODO: implement Grad-CAM model

# TODO: study how nnUnet evaluates the results
# TODO: implement evaluation metrics
    # - Box plots
    # - Ask Julia and Carles for metrics

# TODO: implement multimodality

# TODO: implement MAE 

# TODO: implement depthwise separable convolutions: https://ieeexplore.ieee.org/document/10309848
    # - Autoencoder
    # - U-Net

# TODO: implement Supervised Model:
    # - U-Net
    # - Swin UNETR

def main(args):
    exp = {
        # ---------------- Hyperparameters ----------------
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,

        # ---------------- Model identification ----------------
        "project": args.project,
        "model_name": args.model_name,
        "model_code": args.model_code,
        "dataset": args.dataset,
        "device": args.device,
        "checkpoint": args.checkpoint,

        # ---------------- Dataset configuration ----------------
        "data_path": args.data_path,
        "modalities": args.modalities,
        "transform": args.transform,
        "blosc2": args.blosc2,
        "fold": args.fold,

        # ---------------- Model architecture ----------------
        "preset_model": args.preset_model,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "fuse_layer": args.fuse_layer,
        "encoder_dim": args.encoder_dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "dense": args.dense,
        "n_blocks": args.n_blocks,
        "spatial_dim": args.spatial_dim,
        "gap": args.gap,
        "channels": args.channels,
        "z_dim": args.z_dim,
        "rescale": args.rescale,
        "optimizer": args.optimizer,
        "input_shape": args.input_shape,
        "VAE": args.VAE,
        "aggregate_across_patches": args.aggregate_across_patches,
        "resnet": args.resnet,

        # ---------------- Loss configuration ----------------
        "Rec_Loss": args.Rec_Loss,
        "SSIM_Loss": args.SSIM_Loss,
        "SSIM_weight": args.SSIM_weight,
        "Kl_Loss": args.Kl_Loss,

        # ---------------- Grad-CAM settings ----------------
        "grad_cam": args.grad_cam,
        "alpha_ae": args.alpha_ae,
        "p_activation_cam": args.p_activation_cam,
        "t": args.t,
        "expansion_loss_penalty": args.expansion_loss_penalty,
        "pre_training_epochs": args.pre_training_epochs,
        "level_cams": args.level_cams,

        # ---------------- Output ----------------
        "dir_out": args.dir_out,
    }

    # Set wandb logging
    wandb.init(project=exp["project"], config={
        "learning_rate": exp["learning_rate"],
        "batch_size": exp["batch_size"],
        "epochs": exp["epochs"],
        "patch_size": exp["patch_size"],
        "weight_decay": exp["weight_decay"],
        "image_size": exp["image_size"],
        "architecture": exp["model_name"],
        "dataset": exp["dataset"],
        "code": exp["model_code"]
    })

    # Load training and validation data
    trainLoader, validLoader = loadData(exp['data_path'], exp['modalities'], exp['transform'], exp['batch_size'], exp['blosc2'], exp['fold'])

    # Set model
    if exp['preset_model']:
        model = UnsupervisedAnomalyDetectorModels(model_name=exp['model_name'], model_code=exp['model_code'], device=exp['device'], checkpoint=exp['checkpoint'])
    else:
        model = UnsupervisedAnomalyDetectorModels(
            model_name=exp['model_name'], model_code=exp['model_code'], encoder=exp['encoder'], decoder=exp['decoder'], checkpoint=exp['checkpoint'], fuse_layer=exp['fuse_layer'],
            device=exp['device'], image_size=exp['image_size'], patch_size=exp['patch_size'], encoder_dim=exp['encoder_dim'], 
            depth=exp['depth'], heads=exp['heads'], mlp_dim=exp['mlp_dim'], dense=exp['dense'], n_blocks=exp['n_blocks'],
            spatial_dim=exp['spatial_dim'], gap=exp['gap'], channels=exp['channels'], z_dim=exp['z_dim'], rescale=exp['rescale'],
            optimizer=exp['optimizer'], input_shape=exp['input_shape'], VAE=exp['VAE'], aggregate_across_patches=exp["aggregate_across_patches"], resnet=exp['resnet'], 
            Rec_Loss=exp['Rec_Loss'], SSIM_Loss=exp['SSIM_Loss'], SSIM_weight=exp['SSIM_weight'], Kl_Loss=exp['Kl_Loss'], grad_cam=exp['grad_cam'], alpha_ae=exp['alpha_ae'], 
            p_activation_cam=exp['p_activation_cam'], t=exp['t'], expansion_loss_penalty=exp['expansion_loss_penalty'], 
            pre_training_epochs=exp['pre_training_epochs'], level_cams=exp['level_cams']
        )

    # Save experiment setup
    if exp['dir_out'] is not None:
        os.makedirs(exp['dir_out'], exist_ok=True)
        with open(os.path.join(exp['dir_out'], f"{exp['model_name']}-{exp['model_code']}.json"), 'w') as fp:
            json.dump(exp, fp)
    
    # Train model
    model.train(trainLoader, validLoader, exp["epochs"])

    # Print end of experiment
    print("Experiment finished")

    # Save last epoch
    name = f"{exp['model_name']}-{exp['model_code']}.pt"
    torch.save(model.state_dict(), os.path.join(exp['dir_out'], name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train unsupervised anomaly detection models using autoencoders, VAEs or Vision Transformers."
    )
    # Settings
    parser.add_argument("--dir_out", default=None, type=str)
    parser.add_argument("--project", default="ViT-Autoencoder", type=str)
    parser.add_argument("--model_name", default="ViT-VAE", type=str)
    parser.add_argument("--model_code", default="0.0", type=str)
    parser.add_argument("--dataset", default="BraTS-MEN-2023", type=str)
    parser.add_argument("--modalities", default=["t1c"], type=str, nargs="+")
    parser.add_argument("--data_path", default="D:/data/BraTS2023-MEN-adapted/train", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)

    # Hyperparameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patch_size", default=24, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--image_size", default=240, type=int)

    # Dataset preprocessing
    parser.add_argument("--transform", default="minmax[-1,1]", type=str)

    # Blosc2
    parser.add_argument("--blosc2", default=False, type=bool)
    parser.add_argument("--fold", default=0, type=int)

    # Model architecture
    parser.add_argument("--preset_model", default=True, type=bool)
    parser.add_argument("--encoder", default="", type=str)
    parser.add_argument("--decoder", default="", type=str)
    parser.add_argument("--fuse_layer", default=True, type=bool)
    parser.add_argument("--encoder_dim", default=512, type=int)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--mlp_dim", default=1024, type=int)
    parser.add_argument("--dense", default=False, type=bool)
    parser.add_argument("--n_blocks", default=4, type=int)
    parser.add_argument("--spatial_dim", default=7, type=int)
    parser.add_argument("--gap", default=False, type=bool)
    parser.add_argument("--channels", default=1, type=int)
    parser.add_argument("--z_dim", default=256, type=int)
    parser.add_argument("--rescale", default=True, type=bool)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--input_shape", default=(1, 240, 240), type=tuple)
    parser.add_argument("--VAE", default=False, type=bool)
    parser.add_argument("--aggregate_across_patches", default=False, type=bool)
    parser.add_argument("--resnet", default="18", type=str)
    # Loss
    parser.add_argument("--Rec_Loss", default="l1", type=str)
    parser.add_argument("--SSIM_Loss", default=False, type=bool)
    parser.add_argument("--SSIM_weight", default=0, type=float)
    parser.add_argument("--Kl_Loss", default=False, type=bool)
    # Grad-CAM 
    parser.add_argument("--grad_cam", default=False, type=bool)
    parser.add_argument("--alpha_ae", default=1, type=float)
    parser.add_argument("--p_activation_cam", default=0.5, type=float)
    parser.add_argument("--t", default=0.5, type=float)
    parser.add_argument("--expansion_loss_penalty", default="log_barrier", type=str)
    parser.add_argument("--pre_training_epochs", default=10, type=int)
    parser.add_argument("--level_cams", default=3, type=int)

    args = parser.parse_args()
    main(args)
