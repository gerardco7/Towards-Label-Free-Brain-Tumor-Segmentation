import argparse
import os

from tqdm import tqdm

from tools.preprocessing import load_patient_volume
from tools.model import Model
from tools.inference import inference

def main(args):
    input_path = args.input
    output_path = args.output

    # Define paths for temporary processing and model checkpoints
    checkpoint_path_UAD = "checkpoints/ViT-AE-1.4.2.1.3-ep92.pt"
    device = "cuda:0"

    # Process the input data if it does not exist

    # Load Multimodal Vision Transformer Autoencoder (MViT-AE) model
    model = Model(channels=4,
                    image_size=240,
                    patch_size=24,
                    encoder_dim=512,
                    depth=6,
                    heads=8,
                    mlp_dim=1024,
                    z_dim=512,
                    device=device
                ).get_model_from_checkpoint(
                    checkpoint_path=checkpoint_path_UAD
                )

    print(f"UAD model loaded from {checkpoint_path_UAD}.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    volume_keys = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    # Iterate through the dataset
    for volume_key in tqdm(volume_keys, desc="Processing volumes"):
        volume = load_patient_volume(volume_key, input_path)

        inference(
            model=model,
            volume=volume,
            volume_key=volume_key,
            output_path=output_path,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Towards label-free segmentation')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    main(args)
