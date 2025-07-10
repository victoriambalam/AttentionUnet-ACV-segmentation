# Deep Learning-based Segmentation of Ischemic Brain Infarct Lesions in DWI MRI
This repository contains the code for training an Attention U-Net model for automatic segmentation of ischemic brain infarct lesions in Diffusion Weighted Imaging (DWI) MRI scans, as presented in the manuscript:

📄 **Deep Learning applied to segmentation of ischemic brain infarct lesions in Magnetic Resonance Images**

---
## Overview
The proposed approach performs segmentation of acute ischemic stroke lesions (<6 hours post-infarction) using an Attention U-Net architecture:

**Attention U-Net Architecture**:

*U-Net backbone enhanced with attention gates in skip connections to focus on relevant lesion regions
*Combines local and global contextual information for precise segmentation
*Outputs probability maps of lesion locations

**Key Features:**

*Optimized for acute phase DWI images (0-6 hours post-stroke)
*Validated on follow-up scans (48 hours post-infarction)
*Includes comprehensive postprocessing with optimal thresholding (0.90)

---
The model was trained using 5-fold cross-validation on data from 55 patients and evaluated with standard segmentation metrics:

+Dice Similarity Coefficient (DSC)
+Binary Cross-Entropy (BCE) Loss
+Precision, Recall, F1-Score
+Jaccard Index
+Hausdorff Distance

**For final test performance and reproducibility, refer to the manuscript results section.**
