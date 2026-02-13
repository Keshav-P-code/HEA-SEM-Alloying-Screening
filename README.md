# Machine learning-based screening of high-entropy alloy AlCoCrFeNi particles produced via high-energy mechanical alloying method

This repository contains the complete data-processing and deep learning pipeline developed for quantitative screening of compositional homogeneity in mechanically alloyed AlCoCrFeNi high-entropy alloy (HEA) particles using SEM–EDS data.

The workflow is as follows: atomic percent computation → entropy calculation → dimensionality reduction → deep learning regression → interpretability → full-field prediction.

## Project Structure

src/

01_wt_to_atomic5_entropy.py  
→ Converts weight % to atomic % and computes Shannon entropy (H, H_norm)

02_pca_kmeans_entropy_plot.py  
→ PCA + KMeans clustering on entropy/composition space

03_pentagon_simplex_map.py  
→ 5-component compositional visualization

src/04_geometric_augmentation.py
→ Offline geometric augmentation on original SEM images

05_batch_masks_imagej_otsu.py  
→ Binary mask generation using Otsu thresholding

06_densenet121_regression_training.py  
→ DenseNet121 regression model for predicting normalized entropy (H_norm)

07_embeddings_pca_densenet121.py  
→ Embedding-space PCA visualization for interpretability

08_gradcam_regression_densenet121.py  
→ Grad-CAM interpretability for regression models

09_fullfield_prediction_densenet121_hnorm_map.py  
→ Full-field continuous H_norm heatmap prediction pipeline


## Deep Learning Details

Model: DenseNet121 (ImageNet pre-trained)  
Task: Regression (predict H_norm)  
Preprocessing:  
- CLAHE  
- Resize (512×512)  
- Hard gating with binary mask  
- Mask-aware z-normalization  
- 3-channel conversion
- The same training, preprocessing, and evaluation pipeline was also applied to:
- ResNet50
- ConvNeXt-Tiny

Only the backbone architecture was modified, while the dataset splits, preprocessing steps, loss functions, and evaluation metrics remained identical.

## Outputs

- Per-particle entropy predictions (CSV)
- Grad-CAM visualizations
- Full-field entropy heatmaps


## Notes

- Data paths in scripts must be manually edited before running.
- Each script can be run independently after appropriate path configuration.
- The scripts are organized sequentially to reflect the experimental workflow.
- Large SEM datasets and trained model checkpoints are not included in this repository.
- This repository contains code only.

## Associated Work
https://chemrxiv.org/doi/full/10.26434/chemrxiv.10001980/v1
Developed as part of a Master's thesis at the University of New Brunswick 
(Nanocomposites & Mechanics Laboratory).
