import os
import json
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
'''''
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity
"'''


def create_json(path_out_task, num_training, modalities, labels, file_ending):
    # Create the json file
    json_dict = {}

    # add the modalities as channels, skipping 'seg'
    json_dict['channel_names'] = {i: modality for i, modality in enumerate(modalities) if modality != 'seg'}

    # add the labels
    json_dict['labels'] = {label: i for i, label in enumerate(labels)}

    # num_trainning cases 
    json_dict['numTraining'] = num_training

    # file ending 
    json_dict['file_ending'] = file_ending

    # save json file
    with open(os.path.join(path_out_task, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f)


# The brats dataset is adapted for nnUnet by creating a new directory structure and copying the files to the new structure
def main(args):

    path_brats = args.path_brats   
    path_out = args.path_out
    task = args.task
    modalities = args.modalities
    file_ending = args.file_ending
    labels = args.labels

    # Create the directory structure
    os.makedirs(path_out, exist_ok=True)
    path_out_task = os.path.join(path_out, 'Dataset'+task+'_BraTS')
    os.makedirs(path_out_task, exist_ok=True)
    os.makedirs(os.path.join(path_out_task, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(path_out_task, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(path_out_task, 'labelsTr'), exist_ok=True)

    # Get the patients
    patients = sorted([d for d in os.listdir(path_brats) if os.path.isdir(os.path.join(path_brats, d))])

    # Copy the files to the new directory structure
    import time
    for patient in patients: 
        for i, modality in enumerate(modalities):
            patient_name = '-'.join(patient.split('-')[:-1])
            path_patient = os.path.join(path_brats, patient, patient + '-' + modality +'.nii.gz').replace('\\', '/')

            if modality == 'seg': 
                path_out_set = os.path.join(path_out_task, 'labelsTr').replace('/', '\\')
                new_name = f"{patient_name}{file_ending}"
            else:
                path_out_set = os.path.join(path_out_task, 'imagesTr').replace('/', '\\')
                new_name = f"{patient_name}_000{i}{file_ending}"

            path_patient = path_patient.replace('/', '\\')
            os.system(f'copy {path_patient} {os.path.join(path_out_set, new_name)} > nul')

    # TODO: check if num_training is correct
    num_training = len(os.listdir(os.path.join(path_out_task, 'labelsTr')))

    create_json(path_out_task, num_training, modalities, labels, file_ending)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adapt BraTS dataset for nnUnet') 
    parser.add_argument('--path_brats', default='D:/data/BraTS2023-MEN', type=str)
    parser.add_argument('--path_out', default='D:/data/nnUnet_raw', type=str)
    parser.add_argument('--task', default='001', type=str)
    parser.add_argument('--modalities', default=['t1c', 't1n', 't2f', 't2w', 'seg'], type=list, nargs="+")
    parser.add_argument('--labels', default=['background', 'edema', 'non-enhancing tumor', 'enhancing tumor'], type=list, nargs="+")
    parser.add_argument('--file_ending', default='.nii.gz', type=str)

    args = parser.parse_args()
    main(args)