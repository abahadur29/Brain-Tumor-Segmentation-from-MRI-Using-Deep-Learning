# Brain-Tumor-Segmentation-from-MRI-Using-Deep-Learning

This project implements deep learning-based brain tumor detection and segmentation using the BraTS 2020 dataset. The goal is to classify MRI slices for tumor presence and perform pixel-level tumor segmentation using an Attention U-Net architecture.

---

## Project Overview

- **Dataset:** BraTS 2020 training data (downloaded via Kaggle)
- **Data Processing:**  
  - Load and extract MRI slices and corresponding tumor masks from `.h5` files  
  - Resize images and masks to 128x128 pixels  
  - Normalize image intensities and binarize masks  
- **Tasks:**  
  1. **Binary Classification** - Detect presence or absence of tumor in MRI slices  
  2. **Segmentation** - Generate tumor masks using Attention U-Net deep learning model  

---

## Implementation Details

### Data Handling
- Utilizes `h5py` for reading `.h5` files  
- Preprocessing includes resizing with OpenCV and normalization to [0,1]  
- Splits data into train and test sets with stratification for classification  

### Models
- **Binary Classifier:**  
  - CNN with 2 convolutional layers, max pooling, and dense layers  
  - Uses binary crossentropy loss and accuracy metrics  
- **Attention U-Net for Segmentation:**  
  - Encoder-decoder architecture with attention gates  
  - Uses convolution, batch normalization, and ReLU activations  
  - Compiled with Adam optimizer and binary crossentropy loss  

### Training
- Early stopping callback to avoid overfitting  
- Train/test split for classification (80/20) and segmentation (90/10)  
- Batch sizes: 32 (classification), 16 (segmentation)  
- Epochs: up to 10 (classification), 15 (segmentation)  

---

## Evaluation Metrics

- **Classification:** Accuracy, confusion matrix, classification report  
- **Segmentation:** Dice coefficient, Intersection over Union (IoU)  
- Visualized segmentation overlays and sample predictions  

---

## Additional Utilities

- Functions to plot random samples and segmentation overlays  
- Synthetic brain MRI image and mask generators for testing and visualization  
- Save predicted masks as PNG images  
- Plot training/validation loss and accuracy curves  
- Distribution plots for Dice and IoU scores (histogram, boxplot, violin plot)  

---

## Dependencies

- Python 3.x  
- numpy  
- matplotlib  
- opencv-python  
- h5py  
- scikit-learn  
- tensorflow / keras  
- seaborn  
- pandas  

---

## Usage

1. Download the BraTS 2020 dataset from Kaggle (link in notebook)  
2. Run the notebook cells sequentially to preprocess data, train models, and evaluate  
3. Visualize predictions and save segmentation outputs  

---

## References

- [BraTS 2020 Challenge Dataset](https://www.kaggle.com/awsaf49/brats2020-training-data)  
- Attention U-Net paper: Oktay et al., “Attention U-Net: Learning Where to Look for the Pancreas” (2018)  
- TensorFlow and Keras official documentation  

---

## Author

This notebook was created as part of a brain tumor detection and segmentation project using deep learning and classical image processing techniques.

---

Feel free to reach out with questions or suggestions!
