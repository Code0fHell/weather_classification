# Weather Image Prediction

This project uses deep learning to classify weather conditions from images. It leverages transfer learning with MobileNetV2 and provides a full workflow from data preparation to model deployment.

## Project Structure

```
.gitignore
app.py
README.md
requirements.txt
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
myenv/
static/
    styles.css
    uploads/
templates/
    index.html
uploads/
    test.png
```

- **weather_prediction_training.ipynb**: Jupyter notebook for data processing, model training, evaluation, and testing.
- **app.py**: (Presumably) Flask or FastAPI app for serving predictions (not detailed here).
- **dataset/**: Contains subfolders for each weather class with images.
- **model/**: Stores the trained Keras model.
- **static/**, **templates/**: For web app static files and HTML templates.
- **uploads/**: For user-uploaded images to be predicted.

## Setup

1. **Clone the repository** and create a virtual environment:
    ```sh
    python -m venv myenv
    source myenv/Scripts/activate  # On Windows
    # or
    source myenv/bin/activate      # On Linux/Mac
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
    If `requirements.txt` is missing, install manually:
    ```sh
    pip install tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python kaggle
    ```

3. **Download the dataset**:
    - Place your `kaggle.json` API key in the notebook directory.
    - Run the relevant cells in [weather_prediction_training.ipynb](weather_prediction_training.ipynb) to download and extract the dataset from Kaggle.

## Workflow

### 1. Data Preparation

- The dataset is organized into class folders under `dataset/`.
- The notebook splits the data into training, validation, and test sets (70/15/15).
- Data augmentation is applied to the training set.

### 2. Model Training

- Uses MobileNetV2 as a base (pre-trained on ImageNet).
- Custom dense and dropout layers are added for classification.
- Handles class imbalance with computed class weights.
- Early stopping and learning rate reduction callbacks are used.
- Fine-tuning is performed by unfreezing the last 10 layers of MobileNetV2.

### 3. Evaluation

- Plots training/validation accuracy and loss.
- Evaluates on the test set.
- Visualizes predictions using PCA.
- Displays a confusion matrix and classification report.

### 4. Saving and Loading the Model

- The trained model is saved as `model/weather_cnn.keras`.
- For predictions, the model can be loaded and used on new images.

### 5. Web App (Optional)

- If `app.py` is a web server, it likely uses the trained model to serve predictions via a web interface.
- Static files and templates are used for the frontend.

## Usage

### Training

1. Open [weather_prediction_training.ipynb](weather_prediction_training.ipynb) in Jupyter or VS Code.
2. Run all cells in order to:
    - Download and prepare the dataset.
    - Train and fine-tune the model.
    - Evaluate and save the model.

### Prediction

- To predict a new image:
    1. Place the image in the `uploads/` directory (e.g., `uploads/test.png`).
    2. Use the prediction code in the notebook or via the web app (if implemented).

### Web App

- If you have a web interface, start the server:
    ```sh
    python app.py
    ```
- Visit `http://localhost:5000` (or the specified port) to upload images and see predictions.

## Notes

- Ensure the dataset directory structure matches the expected format.
- Adjust hyperparameters (batch size, learning rate, epochs) as needed.
- The notebook is designed for both local and Colab use.
- For best results, use a GPU-enabled environment.

## References

- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Documentation](https://www.tensorflow.org/)
