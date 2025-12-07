import argparse
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.models import UnsupervisedAnomalyDetectorModels
from methods.datasets.data import loadData
from evaluation.methods.pipeline import pipeline
from methods.datasets.DatasetBlosc2 import DatasetBlosc2
from evaluation.methods.metrics import save_results

"""
python3 test_evalutation.py --json evaluate_model_2.json
"""

def main(args):

    if args.json is not None:
        with open(args.json) as json_file:
            args_dict = json.load(json_file)
            args = argparse.Namespace(**args_dict)

            if  args.number_of_models != len(args.test_modalities) or args.number_of_models != len(args.model_name) or args.number_of_models != len(args.model_code) or args.number_of_models != len(args.checkpoint) or args.number_of_models != len(args.modalities) or args.number_of_models != len(args.postprocess):
                raise ValueError("The number of models must correspond with the length of model_name, model_code, checkpoint, modalities, test modalities and postprocess")
            
            if len(args.model_name) <= 1 and args.ensamble:
                raise ValueError("Ensemble is only supported for multiple models")
            
            #checker function :)
    
    models = []
    datasets = []

    for i in range(args.number_of_models):
        # Get model
        model = UnsupervisedAnomalyDetectorModels(
            model_name = args.model_name[i],
            model_code = args.model_code[i],
            device = args.device,
            is_training = False,
            checkpoint = args.checkpoint[i]
        ).get_model()

        models.append(model)

        # Get data keys
        dataset = DatasetBlosc2(folder=args.data_path[i], identifiers=None)
        datasets.append(dataset)

    data_keys = datasets[0].identifiers

    # Pipeline
    results = pipeline(
        modalities = args.modalities,
        test_modalities = args.test_modalities,
        models = models,
        datasets = datasets,
        data_keys = data_keys,
        transform = args.transform,
        device = args.device,
        plot = args.plot,
        postprocess = args.postprocess,
        brain_contour = args.brain_contour,
        convex_hull = args.convex_hull,
        circle_masking = args.circle_masking,
        connected_components = args.connected_components,
        intensity_component = args.intensity_component,
        metrics = args.metrics,
        ensamble = args.ensamble,
        multimodal = args.multimodal,
        bimodal = args.bimodal,
        nnUnet = args.nnUnet,
        SAM = args.SAM,
        SAM_ensemble = args.SAM_ensemble,
        plans_path=args.plans_path,
    )

    if not args.ensamble:
        # Save results
        checkpoint_name = '.'.join(os.path.basename(args.checkpoint[0]).split('.')[:-1])

        # Determine the output file name based on postprocessing options
        if args.postprocess == 'not_scaled_brain_mask':
            file_name = os.path.join(args.dir_out, 'Not_scaled_brain_mask.csv')
        else:
            postprocess_suffix = args.postprocess[0]
            if args.brain_contour:
                postprocess_suffix += '_brain_contour'
            if args.convex_hull:
                postprocess_suffix += '_convex_hull'
            if args.circle_masking:
                postprocess_suffix += '_circle_masking'
            if args.connected_components:
                postprocess_suffix += '_connected_components'
                if args.intensity_component:
                    postprocess_suffix += '_intensity_component'
                    
            modality_suffixes = {
                't2f': '_t2f',
                't1c': '_t1c',
                't1n': '_t1n',
                't2w': '_t2w'
            }
            postprocess_suffix += '_t1c'
            file_name = os.path.join(args.dir_out, f"{checkpoint_name}_{postprocess_suffix}.csv")
    else:
        checkpoint_name = ''
        for i, model in enumerate(models):
            checkpoint_name += '.'.join(os.path.basename(args.checkpoint[i]).split('.')[:-1])
            # Determine the output file name based on postprocessing options
            if args.postprocess == 'not_scaled_brain_mask':
                file_name = os.path.join(args.dir_out, f'Not_scaled_brain_mask_{checkpoint_name}.csv')
            else:
                postprocess_suffix = args.postprocess
                if args.brain_contour:
                    postprocess_suffix += '_brain_contour'
                if args.convex_hull:
                    postprocess_suffix += '_convex_hull'
                if args.circle_masking:
                    postprocess_suffix += '_circle_masking'
                if args.connected_components:
                    postprocess_suffix += '_connected_components'
                    if args.intensity_component:
                        postprocess_suffix += '_intensity_component'
                        
                modality_suffixes = {
                    't2f': '_t2f',
                    't1c': '_t1c',
                    't1n': '_t1n',
                    't2w': '_t2w'
                }
                postprocess_suffix += ''.join(modality_suffixes[modality] for modality in args.modalities[i] if modality in modality_suffixes)
        file_name = os.path.join(args.dir_out, f"{checkpoint_name}_{postprocess_suffix}.csv")

    # Save the results to the determined file
    save_results(results, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised Anomaly Detection Evaluation')
    # JSON arguments   
    parser.add_argument('--json', type=str, default=None, help='JSON file')
    # Model arguments
    parser.add_argument('--model_name', type=str, default='AE', help='Model name')
    parser.add_argument('--model_code', type=str, default='1.1.1', help='Model code')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint')
    # Data arguments
    parser.add_argument('--data_path', type=str, default='D:/data/BraTS2023-MEN-adapted/train', help='Data path')
    parser.add_argument('--modalities', type=list, default=['t1c'], help='Modalities')
    parser.add_argument('--transform', type=str, default='base', help='Transform')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--blosc2', type=bool, default=False, help='Blosc2')
    # Pipeline arguments
    parser.add_argument('--plot', type=bool, default=False, help='Plot')
    parser.add_argument('--postprocess', type=str, default='StRegA', help='Postprocess')
    parser.add_argument('--brain_contour', type=bool, default=False, help='Brain contour')
    parser.add_argument('--convex_hull', type=bool, default=False, help='Convex hull')
    parser.add_argument('--circle_masking', type=bool, default=False, help='Circle masking')
    parser.add_argument('--connected_components', type=bool, default=False, help='Connected components')
    parser.add_argument('--intensity_component', type=bool, default=False, help='Intensity component')
    parser.add_argument('--metrics', type=str, nargs='+', default=['dice', 'hausdorff_distance', 'confusion_matrix'], help='Metrics to evaluate (e.g., dice, hausdorff_distance, confusion_matrix)')
    # Output arguments
    parser.add_argument('--dir_out', type=str, default='csv_metrics2', help='Output directory')

    args = parser.parse_args()
    main(args)