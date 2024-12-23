
```
# ECG Signal Classification

This project involves processing Electrocardiogram (ECG) signals and applying machine learning techniques to classify them as either "Normal" or "RBBB" (Right Bundle Branch Block). The application consists of a **Jupyter Notebook** used for model training and a **Tkinter GUI** for testing the trained models with ECG signal data.

## Project Structure

```
ECG_Signal_Classification/
│
├── data/                     # Folder to store ECG data files
├── models/                   # Folder to save trained machine learning models
│   ├── knn_model.pkl         # KNN model
│   ├── svm_model.pkl         # SVM model
│   └── scaler.pkl            # Feature scaler
│
├── jupyter_notebooks/         # Folder with Jupyter notebooks for training and evaluation
│   └── ecg_classification.ipynb
│
├── src/                      # Folder with source code
│   ├── main.py           # Tkinter GUI application
│   ├── preprocessing.py      # Data preprocessing and feature extraction functions
│   └── utils.py              # Helper functions
│
└── README.md                 # Project overview and documentation
```

## Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `scipy`
- `pywt` (PyWavelets)
- `scikit-learn`
- `tkinter`
- `pickle`

To install the required libraries, use `pip`:

```bash
pip install numpy matplotlib scipy pywt scikit-learn
```

### Running the Application

1. **Training the Model (Jupyter Notebook)**:
   - Open `ecg_classification.ipynb` in Jupyter Notebook.
   - Load the ECG data files (Normal & RBBB) and preprocess them using bandpass and notch filters.
   - Extract wavelet features from the ECG signals.
   - Train machine learning models like **KNN**, **SVM**, and **RandomForest** on the processed data.
   - Save the trained model and scaler to the `models/` directory.

2. **Testing the Model (Tkinter GUI)**:
   - Open and run `main.py` to launch the Tkinter GUI.
   - Click on "Load Model" to load a pre-trained model from the `models/` folder.
   - Click on "Load Signal" to select an ECG signal file for testing.
   - The GUI will preprocess the signal and display the performance metrics of the model (Precision, Recall, F1-Score, and Accuracy) based on the test data.

### Sample Usage

- **Model Training**:
   1. Train the model using the Jupyter notebook, and pick the desired model (KNN, SVM, or Random Forest).
   2. The models are saved in the `models/` folder for future use.

- **Model Testing**:
   1. After launching the Tkinter GUI, load a signal file (in `.txt` format) containing ECG signal data.
   2. The GUI will preprocess the signal, extract features, and classify the signal using the loaded model.
   3. The test results (Precision, Recall, F1-Score, and Accuracy) are displayed in a message box.

### Example Data

Data files in the `data/` directory are structured as:

- **Normal**: Contains ECG signals for normal heartbeats.
- **RBBB**: Contains ECG signals with Right Bundle Branch Block (RBBB) abnormalities.

Each file contains ECG signals where each line corresponds to an individual signal.

### File Formats

- **ECG Data Files**: The ECG signals are stored in plain text files (`.txt`), where each signal is represented by a series of floating-point numbers separated by the pipe (`|`) symbol.
  
- **Model Files**: Models and scalers are saved in `.pkl` files using the `pickle` module for easy loading and reuse.

### Example of Signal Format

```plaintext
0.002|0.003|0.004|...|0.002|
0.002|0.001|0.004|...|0.001|
...
```

### Model Evaluation

The trained models are evaluated using the following metrics:
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positive instances.
- **F1-Score**: The harmonic mean of precision and recall.
- **Accuracy**: The overall accuracy of the model.

### Preprocessing Details

The preprocessing pipeline involves:
1. **Bandpass Filtering**: To remove baseline wander and high-frequency noise.
2. **Notch Filtering**: To remove powerline interference.
3. **Normalization**: Using either z-score normalization or min-max scaling to bring all features into a similar range.
4. **Wavelet Transform**: Discrete Wavelet Transform (DWT) is used to extract features from the ECG signal, including energy and entropy metrics.

## Results

After training and evaluating the models, we obtain the following performance metrics:

### KNN Model:

- **Accuracy**: 98.65%
- **Precision**: 0.98
- **Recall**: 0.98
- **F1-Score**: 0.98

### SVM Model:

- **Accuracy**: 99.21%
- **Precision**: 0.99
- **Recall**: 0.99
- **F1-Score**: 0.99

### Random Forest Model:

- **Accuracy**: 98.96%
- **Precision**: 0.99
- **Recall**: 0.98
- **F1-Score**: 0.98

## Conclusion

This project demonstrates how to preprocess ECG signals, extract features using wavelet transforms, and apply machine learning algorithms to classify heart conditions. The trained models can be easily used for ECG signal classification via the Tkinter GUI.

---

