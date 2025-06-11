# Weather Image Prediction with Deep Learning

This project trains a Convolutional Neural Network (CNN) to classify weather conditions from images using transfer learning with MobileNetV2. The workflow includes data preparation, augmentation, model training, evaluation, and prediction.

## Project Structure

```
weather_prediction_training.ipynb
dataset/
    dew/
    fogsmog/
    frost/
    glaze/
    hail/
    lightning/
    rain/
    rainbow/
    rime/
    sandstorm/
    snow/
model/
    weather_cnn.keras
```

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn
- OpenCV
- Google Colab (for file upload utilities)
- kaggle (for dataset download)

Install dependencies:
```sh
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn kaggle opencv-python
```

## Dataset

The dataset is downloaded from Kaggle (`jehanbhathena/weather-dataset`). It contains images categorized into 11 weather classes.

## Workflow

### 1. Install and Import Libraries

All necessary libraries are installed and imported at the start of the notebook.

### 2. Dataset Preparation

- **Kaggle API** is used to download the dataset.
- The dataset is unzipped and organized into class folders under `dataset/`.

### 3. Data Exploration

- The script prints the number of images per class.
- Sample images from each class are displayed using matplotlib.

### 4. Data Processing

#### a. Split Dataset

The dataset is split into training (70%), validation (15%), and test (15%) sets using [`split_dataset`](d:/weather_prediction_project/weather_prediction_training.ipynb).

#### b. Data Augmentation

- Training images are augmented with rotation, shift, shear, zoom, and horizontal flip.
- Validation and test images are only rescaled.

#### c. Data Generators

Keras `ImageDataGenerator` is used to create generators for training, validation, and testing.

### 5. Model Building

- **Transfer Learning:** MobileNetV2 (pre-trained on ImageNet) is used as the base model.
- The base model is frozen initially.
- Custom layers (GlobalAveragePooling, Dense, Dropout, and final Dense with softmax) are added for classification.

### 6. Training

- **Class Imbalance:** Class weights are computed to handle imbalance.
- **Callbacks:** Early stopping and learning rate reduction on plateau are used.
- The model is trained first with the base model frozen, then fine-tuned by unfreezing the last 10 layers.

### 7. Visualization

- Training and validation accuracy/loss are plotted over epochs.

### 8. Evaluation

- The model is evaluated on the test set.
- PCA is applied to the predictions for visualization.
- A confusion matrix and classification report are generated.

### 9. Saving and Loading the Model

- The trained model is saved as `models/weather_cnn.keras`.
- For prediction, the model can be loaded and used to classify new images.

### 10. Test Prediction

Example code is provided to load the model and predict the weather class for a new image.

## Usage

1. **Download the dataset**  
   Upload your `kaggle.json` and run the notebook to download and extract the dataset.

2. **Run the notebook**  
   Execute each cell in [weather_prediction_training.ipynb](d:/weather_prediction_project/weather_prediction_training.ipynb) in order.

3. **Train the model**  
   The notebook will handle data splitting, augmentation, model training, and evaluation.

4. **Predict new images**  
   Use the provided code to load the trained model and predict the class of new images.

## Notes

- The code is designed for use in Google Colab but can be adapted for local use.
- Ensure the dataset directory structure matches the expected format.
- Adjust hyperparameters (batch size, learning rate, epochs) as needed for your hardware and dataset size.

## References

- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)