# Speech Emotion Recognition using LSTM

This project focuses on building a **Speech Emotion Recognition (SER)** system using deep learning techniques, particularly a Long Short-Term Memory (LSTM) network, to classify emotions from audio speech data. The dataset used is the **TESS Toronto emotional speech set**.

## 🎯 Project Objectives

- Preprocess and explore the TESS dataset.
- Visualize waveforms and spectrograms of different emotions.
- Extract MFCC (Mel-frequency cepstral coefficients) features from audio.
- Train and evaluate an LSTM-based deep learning model to classify emotions.

## 📁 Dataset

- **Source**: [TESS Toronto emotional speech set](https://tspace.library.utoronto.ca/handle/1807/24487)
- Database : https://drive.google.com/drive/folders/1bc_0XatymeE5Gt4O-NwQQgt_FlNID0dr?usp=sharing
- It includes recordings of speech with emotions like *happy*, *sad*, *angry*, *fear*, *disgust*, *neutral*, etc.

## 🛠️ Project Structure

- **Run.ipynb** – The main script that performs the entire workflow: data loading, visualization, feature extraction, model training, and evaluation.

## 🔍 Exploratory Data Analysis

- Counts of each emotion class are visualized.
- Waveforms and spectrograms for selected emotions are plotted for visual inspection.

## 📊 Feature Engineering

- MFCC features (40 coefficients) are extracted per audio sample using `librosa`.

## 🤖 Model

- A Sequential LSTM model with dropout layers for regularization.
- Output layer has 7 units with softmax activation (for 7 emotion classes).
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Evaluation: Accuracy on validation data

## 📈 Results

- Training and validation accuracy and loss curves are plotted over 50 epochs to observe learning dynamics.

## 🧰 Libraries & Tools

- Python 3
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `librosa` for audio processing
- `keras` (TensorFlow backend) for deep learning model
- Google Colab and Google Drive for development and data handling

