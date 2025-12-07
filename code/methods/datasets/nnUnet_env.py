import os
import argparse

#TODO: run in linux
'''
cd nnUNet_Frame

# Windows
nnunet_env\Scripts\activate

set nnUnet_raw=D:\data\nnUnet_raw
set nnUnet_preprocessed=D:\data\nnUnet_preprocessed
set nnUnet_results=D:\models\nnUnet_results

nnUNetv2_train 001 2d 0
'''

def main(args):
    nnUNet_raw_data_base = args.nnUNet_raw_data_base
    nnUNet_preprocessed = args.nnUNet_preprocessed
    nnUnet_results= args.nnUnet_results
    
    os.environ['nnUNet_raw_data_base'] = nnUNet_raw_data_base
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
    os.environ['nnUnet_results'] = nnUnet_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adapt BraTS dataset for nnUnet')
    parser.add_argument('--nnUNet_raw_data_base', default='D:/data/nnUnet_raw', type=str)
    parser.add_argument('--nnUNet_preprocessed', default='D:/data/nnUnet_preprocessed', type=str)
    parser.add_argument('--nnUnet_results', default='D:/models/nnUnet_results', type=str)
    args = parser.parse_args()
    main(args)
