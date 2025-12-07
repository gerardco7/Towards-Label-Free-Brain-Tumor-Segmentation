import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import os

def plot_results(x, x_hat, diff, diff_post, mask, metrics):
    """
    Plots original, reconstructed, difference, postprocessed difference, mask, and Dice scores for each modality.
    Shows Dice scores for ET, TC, and WT (ignores the 4th if present).
    """
    dice_labels = ['ET', 'TC', 'WT']
    dice_scores = []

    # Extract dice scores for ET, TC, WT (ignore 4th if present)
    for label in dice_labels:
        # Find the dice metric and extract the correct index for ET, TC, WT
        dice_metric = next((m for m in metrics if m['Metric'] == 'dice'), None)
        if dice_metric is not None and isinstance(dice_metric['Value'], (list, np.ndarray)):
            # Map label to index: ET=0, TC=1, WT=2
            idx = dice_labels.index(label)
            score = dice_metric['Value'][idx] if idx < len(dice_metric['Value']) else None
        else:
            score = None
        dice_scores.append(score)

    num_modalities = x.shape[0]
    for modality in range(num_modalities):
        # Plot histograms
        x_flat = x[modality].flatten()
        x_hat_flat = x_hat[modality].flatten()

        plt.figure(figsize=(12, 4))
        plt.hist(x_flat[x_flat > 0], bins=100, alpha=0.5, color='b', label=f'Original MRI - Modality {modality}')
        plt.hist(x_hat_flat[x_hat_flat > 0], bins=100, alpha=0.5, color='r', label=f'Reconstructed MRI - Modality {modality}')
        plt.legend(loc='upper right')
        plt.title(f'Histograms - Modality {modality}')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.tight_layout()
        # plt.show()

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs[0, 0].imshow(x[modality], cmap='gray')
        axs[0, 0].set_title(f'Original MRI - Modality {modality}')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(x_hat[modality], cmap='gray')
        axs[0, 1].set_title(f'Reconstructed MRI - Modality {modality}')
        axs[0, 1].axis('off')

        im = axs[0, 2].imshow(diff[modality], cmap='viridis')
        axs[0, 2].set_title(f'Difference - Modality {modality}')
        axs[0, 2].axis('off')
        fig.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

        axs[1, 0].imshow(mask.detach().cpu().numpy(), cmap='gray')
        axs[1, 0].set_title(f'Mask')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(diff_post, cmap='gray')
        axs[1, 1].set_title(f'Difference Postprocessed - Modality {modality}')
        axs[1, 1].axis('off')

        # Dice scores table
        cell_text = [[label, f"{score:.4f}" if score is not None else "N/A"] for label, score in zip(dice_labels, dice_scores)]
        axs[1, 2].axis('off')
        table = axs[1, 2].table(cellText=cell_text, colLabels=["Dice Type", "Score"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        axs[1, 2].set_title('Dice Scores (ET, TC, WT)')

        plt.tight_layout()
        #plt.show()
        
        # obtain slice number from metrics
        slice_number = metrics[0]['Slice']  
        volume = metrics[0]['Volume']
        
        model_name = '1.9.1'

        if not os.path.exists(f'C:/Users/gerar/Documents/TFG/TFG memory/presentació/{model_name}/{volume}'):
            os.makedirs(f'C:/Users/gerar/Documents/TFG/TFG memory/presentació/{model_name}/{volume}')

        # save each plot as a separate image
        fig.savefig(f'C:/Users/gerar/Documents/TFG/TFG memory/presentació/{model_name}/{volume}/Figure_{slice_number}.png', bbox_inches='tight')
        plt.close(fig)


def create_results_plot(df_metrics, model_name = 'ViT-AE-1.4.2.1.3-99+95+40-t2f'):
    # We will plot the histogram representing the distribution of the WT Dice Score values, but for each column we will have multiple stacked bars
    # representing the different kind of slices that we have in the dataset.

    # The structure of the df is:
    # df_tumour_type:
    #   - Patient: the patient ID
    #   - Slice: the slice number
    #   - Tumor Type: the type of tumor in the slice (e.g., "Enhancing Tumor", "Non-Enhancing Tumor", "Edema")
    # df_metrics:
    #   - Patient: the patient ID
    #   - Slice: the slice number
    #   - Metric: the metric name (e.g., "dice")
    #   - Value: the value of the metric

    tumour_type_path = r"C:\Users\gerar\Documents\TFG\visual_transformers_anomaly_segmentation\data_analysis\type_tumor_slices.csv"
    df_tumour_type = pd.read_csv(tumour_type_path)

    alt.data_transformers.disable_max_rows()

    df_merged = pd.merge(df_tumour_type, df_metrics, on=['Volume', 'Slice'])

    df_dice = df_merged[df_merged['Metric'] == 'dice'].copy()

    df_dice['Value'] = df_dice['Value'].apply(lambda x: eval(x)[2] if isinstance(x, str) else x[2])
    #df_dice = df_dice[df_dice['Value'] > 0]

    tumor_type_selection = alt.selection_point(
        fields=['Tumor Type'],
        bind=alt.binding_select(options=list(df_dice['Tumor Type'].unique()), name='Select Tumor Type: '),
    )

    hist = alt.Chart(df_dice).mark_bar().encode(
        x=alt.X('Value:Q', bin=alt.Bin(maxbins=50), title='WT Dice Score', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('count()', title='Frequency'),
        color=alt.Color('Tumor Type:N', title='Tumor Type', scale=alt.Scale(
            domain=['Enhancing Tumor', 'Non-Enhancing Tumor', 'Edema', 
                    'Enhancing Tumor + Edema', 'Enhancing Tumor + Non-Enhancing Tumor', 
                    'Non-Enhancing Tumor + Edema', 
                    'Enhancing Tumor + Non-Enhancing Tumor + Edema'],
            range=['#81D4FA', '#D4E157', '#FF8A80', '#B39DDB', '#00897B', '#FFB74D', '#BDBDBD']
        )),
        tooltip=['count()']
    ).add_params(
        tumor_type_selection
    ).transform_filter(
        tumor_type_selection
    ).properties(
        title='Distribution of WT Dice Score by Tumor Type',
        width=600,
        height=400
    )

    # Save the chart as html file 
    hist.save('C:/Users/gerar/Documents/TFG/visual_transformers_anomaly_segmentation/code/evaluation/hist_metrics/histogram_' + model_name + '.html')





