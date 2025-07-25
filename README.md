# levir-change-detection
Change detection on satellite images using deep learning.
Satellite Change Detection with UNet

This project focuses on detecting **meaningful changes (e.g., urban growth, building construction, demolition)** between two satellite images taken at different times.  
The goal is to track **urban development** by training and deploying a deep learning model based on the **UNet architecture**.

The model takes two-time satellite image pairs as input and produces **pixel-level binary segmentation masks** to localize change areas.

Project Features
-  Change detection from bi-temporal satellite images  
-  UNet-based binary segmentation model  
-  Trained using the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) dataset  
-  Evaluation metrics: IoU, Dice coefficient, Validation Accuracy  
-  Visualization of predicted masks and overlaid results

Main Notebook
All training, evaluation, and inference steps are implemented in:
[`satellite_change_unet.ipynb`](./satellite_change_unet.ipynb)

 Dataset
This project uses the [**LEVIR-CD**](https://justchenhao.github.io/LEVIR/) dataset,  
a large-scale high-resolution building change detection dataset designed for remote sensing applications.
- The dataset is **not distributed with this repository**.
-  Official source and download link: [https://github.com/justchenhao/LEVIR-CD](https://github.com/justchenhao/LEVIR-CD)

  Technologies Used
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Google Colab (with T4 GPU support)

Model Outputs
All predicted masks and overlay results are saved under the `outputs/` folder.

About the Model (UNet Architecture)

The model used in this project is based on the **UNet architecture**, a popular convolutional neural network designed for **semantic segmentation tasks**.

UNet consists of two main parts:
- **Encoder** (contracting path): Extracts deep feature representations from the input image
- **Decoder** (expanding path): Reconstructs the output segmentation map using transposed convolutions and skip connections

In this project, the input to the UNet model is a **6-channel image** created by concatenating two RGB satellite images from different time steps.  
The model outputs a **single-channel binary mask** highlighting the changed areas between the two input images.
