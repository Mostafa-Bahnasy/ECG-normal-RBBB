import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import pywt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

class PreProcessing:
    def __init__(self):
        pass

    # Bandpass filter to remove baseline wander and high-frequency noise
    def bandpass_filter(self,signal, lowcut=0.5, highcut=40, fs=360):
        nyquist = 0.5 * fs
        b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        return filtfilt(b, a, signal)

    # Notch filter to remove powerline interference
    def notch_filter(self,signal, fs=360, freq=50):
        nyquist = 0.5 * fs
        b, a = iirnotch(freq / nyquist, Q=30)
        return filtfilt(b, a, signal)

    def normalize_signal(self,signal, method='zscore'):
        if method == 'minmax': # 0 -> 1
            return (signal - signal.min()) / (signal.max() - signal.min())
        elif method == 'zscore':
            return (signal - signal.mean()) / signal.std()


    def preprocess_ecg(self,ecg_signal, fs=360):
        ecg_signal = ecg_signal - np.mean(ecg_signal)

        ecg_signal= self.bandpass_filter(ecg_signal)
        ecg_signal = self.notch_filter(ecg_signal)
        ecg_signal = self.normalize_signal(ecg_signal)

        return ecg_signal
    def extract_wavelet_features(self,signal, wavelet='db4', level=4):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = []
        for coeff in coeffs:
            features.append(np.sum(np.square(coeff)))  # Energy
            features.append(-np.sum(coeff * np.log2(np.abs(coeff) + 1e-6)))  # Entropy
        return np.array(features)


class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.flag_shuffle = False
        self.model = None
        self.scaler = None
        self.pre_process = PreProcessing()
        self.X_test = [[]]
        self.y_test = []
        self.root.title("Testing model GUI")
        self.root.geometry("300x200")  # Set the window size

        self.load_model_button = tk.Button(root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.load_signal_button = tk.Button(root, text="Load Signal", command=self.load_signal)
        self.load_signal_button.pack(pady=10)

        self.test_model_button = tk.Button(root, text="Test", command=self.test_model)
        self.test_model_button.pack(pady=10)

    def navigate_file(self):
        # Corrected filetypes with separate tuple entries for each extension
        path = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if path:
            print("File loaded successfully.")
            return path
        return None

    def load_signal(self):
        path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text files", "*.txt")])
        data = self.load_ecg_data(path)
        data = self.pre_process.preprocess_ecg(data)
        features = np.array([self.pre_process.extract_wavelet_features(signal) for signal in data])
        if len(self.X_test[0])==0:
            self.X_test = features
        else:
            self.X_test = np.concatenate((self.X_test,features),axis=0)
        filename = os.path.basename(path)
        print(filename)
        if 'Normal' in filename:
            self.y_test = np.hstack([self.y_test, np.zeros(len(features))])
        else:
            self.y_test = np.hstack([self.y_test, np.ones(len(features))])
        print(self.y_test)


    def get_scaler(self):
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

    def test_model(self):
        X_test, y_test = shuffle(self.X_test, self.y_test, random_state=42)

        self.get_scaler()
        # print(self.scaler)

        X_test = self.scaler.transform(self.X_test)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        messagebox.showinfo("Statistics ", f'Precision: {precision:.2f},\n Recall: {recall:.2f},\n F1-Score: {f1:.2f}, \n Accuracy: {accuracy * 100:.2f}%')
        # print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')



    def load_ecg_data(self,file_path):
        signals = []
        with open(file_path, 'r') as f:
            for line in f:
                signal = list(line.strip().split('|'))
                signal = [float(x) if x else 0.0 for x in signal]
                #print(signal)
                signal.pop()
                signals.append(signal)

        # print(signals)
        return np.array(signals)

    def load_model(self):
        file_path = self.navigate_file()
        # Load the model from the file
        with open(file_path, "rb") as file:
            self.model = pickle.load(file)





if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()
