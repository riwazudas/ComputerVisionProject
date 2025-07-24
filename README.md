# COMP90086 - Physical Reasoning Challenge: Block Stack Stability Prediction

This repository contains the code and documentation for our project for the COMP90086 Physical Reasoning Challenge. The goal of this project is to predict the stable height of a stack of blocks from a given image, leveraging various computer vision techniques.

## Authors
- **Kevin Liew Kar Kit** (1508822)
- **Riwaz Udas** (1547555)

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
    - [Preprocessing & Augmentation](#preprocessing--augmentation)
    - [Image Segmentation](#image-segmentation)
    - [Models](#models)
3. [Experiments & Results](#experiments--results)
4. [Discussion & Analysis](#discussion--analysis)
5. [Conclusion & Future Work](#conclusion--future-work)
6. [How to Run](#how-to-run)

## Introduction
The field of computer vision enables a wide range of image-related tasks. In this physical reasoning challenge, our objective is to analyze images of block stacks and predict their stable height, which can range from 1 to 6. The stability of each stack is influenced by complex physical interactions determined by block placement, size, and shape.

Our group aimed to develop a model that could effectively learn these physical relationships. To achieve this, we explored various computer vision techniques, including Convolutional Neural Networks (CNNs), image segmentation, and data augmentation.

## Methodology
Our approach involved several stages, from data preparation to model implementation and tuning.

### Preprocessing & Augmentation
Given the relatively small size of the image dataset, we implemented data augmentation to prevent overfitting and improve the model's generalization capabilities.
- **Augmentation Techniques:** We used the `Albumentations` library to apply horizontal flipping and rotations of up to 20°.
- **Normalization:** All images (training and validation) were normalized using standard mean and standard deviation values to ensure data consistency for our CNN architectures.

### Image Segmentation
To help the model focus on relevant features (the blocks) and ignore background noise, we developed an image segmentation pipeline.
1. **Smoothing:** A 5x5 Gaussian kernel was applied to reduce noise.
2. **Edge Detection:** Canny edge detection was used to highlight prominent edges.
3. **Contour Masking:** The detected edges were dilated, and fill contours were drawn to create a mask, which was then applied to the original image to isolate the block stack.

### Models
We experimented with several model architectures to find the most effective one for this task.

#### 1. Baseline Model: ResNet50
- **Architecture:** We used a ResNet50 model, pre-trained on ImageNet, as our baseline. ResNet is effective at mitigating the vanishing gradient problem in deep networks through its use of skip connections.
- **Configuration:** The model was extended with a Global Max Pooling layer and a dropout layer (rate of 0.5). We used the Adam optimizer.

#### 2. Primary Model: Inception-ResNet-V2
- **Architecture:** Inspired by the original ShapeStack paper, our main model was an Inception-ResNet-V2. This hybrid architecture combines the powerful classification capabilities of Inception with the efficiency of ResNet's skip connections.
- **Configuration:** We used this as a transfer learning model with pre-trained ImageNet weights. The final layer was a classification layer with a SoftMax activation function. It included the same pooling and dropout layers as the baseline, with an added L2 regularization layer.

#### 3. Fusion Model with Metadata
- **Architecture:** We explored a multimodal approach by fusing two neural networks: an Inception-ResNet-V2 for image data and a dense network for metadata. The outputs were concatenated before a final SoftMax prediction layer.
- **Metadata Generation:** Since the test set lacked metadata, we trained five separate CNN models to predict each of the metadata features.

## Experiments & Results
We conducted several experiments to evaluate our different approaches. Hyperparameter tuning was performed using 2-fold cross-validation, which determined the best configuration to be a **dropout rate of 0.5** and an **L2 value of 0.1**.

### Performance Comparison
| Method                          | Train Loss | Train Acc | Val Loss | Val Acc |
| ------------------------------- | ---------- | --------- | -------- | ------- |
| Baseline (ResNet50)             | 0.2287     | 91.96%    | 2.3329   | 52.54%  |
| **Inception-ResNet-V2 (No Seg.)** | **0.3133** | **92.92%**| **1.7151** | **55.92%** |
| Inception-ResNet-V2 (with Seg.) | 0.4412     | 87.95%    | 2.2519   | 51.43%  |
| Fusion Model (with Metadata)    | 1.0311     | 48.22%    | 1.0087   | 48.50%  |

Our best-performing model was the **Inception-ResNet-V2 trained on non-segmented images**.

## Discussion & Analysis
- **Poor Fusion Model Performance:** The fusion model performed the worst, likely because the generated metadata for the test set was inaccurate, introducing noise that hindered the model's learning process.
- **Negative Impact of Segmentation:** Counter-intuitively, our segmentation pipeline degraded the model's performance. Upon review, we found that the segmentation was imperfect, sometimes masking out parts of the blocks or leaving residual background noise. This fed the model incomplete or misleading information.
- **Overfitting:** Despite hyperparameter tuning, overfitting remained a significant issue across all models. This is likely due to the small dataset size and the high complexity of the models, which caused them to learn noise from the training data.

## Conclusion & Future Work
This challenge provided valuable insights into the practical application of computer vision techniques. We learned that each stage of the pipeline—from preprocessing to model selection—is critical to success.

For future improvements, we suggest:
- **Advanced Preprocessing:** Explore more robust image segmentation and noise reduction techniques.
- **Combatting Overfitting:** Implement more aggressive regularization, such as additional dropout layers, and expand the data augmentation pipeline.
- **Exploring Multimodal Models:** Further research into fusion models could be beneficial, especially if more accurate metadata can be obtained.

## How to Run
To replicate our experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/riwazudas/ComputerVisionProject.git](https://github.com/riwazudas/ComputerVisionProject.git)
    cd ComputerVisionProject
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    To train a model, run the main script with the desired configuration.
    ```bash
    python train.py --model InceptionResnetV2 --segmentation False --epochs 50
    ```

4.  **Evaluate the model:**
    To generate predictions on the test set, run the evaluation script.
    ```bash
    python evaluate.py --model_path /path/to/your/trained/model.h5
    ```
