# Speech_Emotion_Recognition
A model to infer emotion from human speech

### Ongoing Project.....
---

# Speech Emotion Recognition

## Overview

This project classifies human emotions from speech audio using a hybrid model that combines **Bidirectional Long Short-Term Memory (BiLSTM)** networks and **Hidden Markov Models (HMM)**. The BiLSTM captures complex temporal patterns in the audio features, while the HMM models the sequential transitions between emotional states. This combination allows the system to benefit from the strengths of both deep learning and probabilistic sequence modeling, improving emotion classification performance, especially in time-dependent speech data.
It extracts audio features like MFCC and feeds them into the model to detect emotions such as **happy**, **sad**, **angry**, **neutral**, and more.
It’s built using Python and popular libraries like Librosa, PyTorch/TensorFlow, and scikit-learn.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributing](#-contribution)
- [License](#-license)

---

## Overview

Speech Emotion Recognition (SER) aims to recognize the underlying emotional state of a speaker from their voice. This project uses audio signal processing techniques and deep learning to build an emotion classifier trained on publicly available datasets and collected ones.

---

## Demo

Try out a sample prediction:
```

python predict.py --file samples/happy.wav

````

---

## Features

- Predicts emotions from audio files (.wav)
- Preprocessing: MFCCs, LFCCs
- Deep learning model (BiLST-MHMM)
- <!--Option to train from scratch or use pre-trained model -->
- Support for batch prediction
- Easily extendable for new datasets

---

## 📂 Dataset

This project supports the following datasets:

- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://www.torontoadventures.ca/tess/)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [Emo-DB](http://emodb.bilderbar.info/start.html)

> Make sure to download and organize the datasets in the `/data` directory as described in the documentation.

---

##  Model Architecture

This project uses a hybrid architecture that combines the strengths of deep learning and probabilistic modeling:

### 🔹 1. Feature Extraction
- Audio signals are preprocessed and transformed into feature representations such as:
  - **MFCCs** (Mel Frequency Cepstral Coefficients)
  - **LFCCs** (Linear Frequency Cepstral Coefficients)
- These features are extracted using `librosa` and normalized for training.

### 🔹 2. Bidirectional LSTM (BiLSTM)
- A deep **Bidirectional Long Short-Term Memory** network is used to capture forward and backward temporal dependencies in speech.
- The BiLSTM outputs a sequence of hidden states representing the emotional dynamics over time.

### 🔹 3. Hidden Markov Model (HMM)
- The sequence output by the BiLSTM is passed to a **Hidden Markov Model**, which models the probabilistic transitions between emotional states.
- The HMM smooths out noisy predictions and improves temporal consistency in emotion recognition.

### 🔹 Model Pipeline Overview

```

Audio (.wav) ──► Feature Extraction ──► BiLSTM ──► HMM ──► Emotion Prediction

```

### 🔹 Training Details
- **Loss Function**: CrossEntropyLoss (for BiLSTM training)
- **Optimizer**: Adam
- **HMM Decoding**: GuassianHMM
- **Epochs**: 40
- **Batch Size**: 30
- **Frameworks Used**: TensorFlow, hmmlearn,

---

This architecture is particularly effective for speech emotion recognition, where emotions evolve over time and have strong sequential dependencies.

## ⚙️ Installation

```bash
git clone https://github.com/EarlyBloomer-byte/Speech_Emotion_Recognition.git
cd Speech_emotion_Recognition
pip install -r requirements.txt
````

Optional:

* Install FFmpeg for audio processing:

  ```bash
  sudo apt install ffmpeg
  ```

---

## 🚀 Usage

### Predict Emotion from Audio

```bash
python predict.py --file path/to/audio.wav
```

### Train the Model

```bash
python train.py --config config.yaml
```

### Jupyter Notebook

You can also explore the model and predictions via the notebook:

```bash
jupyter notebook SER-demo.ipynb
```

---

## 📊 Results

| Emotion | Precision | Recall | F1-Score |
| ------- | --------- | ------ | -------- |
| Happy   | x.xx      | x.xx   | x.xx     |
| Sad     | x.xx      | x.xx   | x.xx     |
| Angry   | x.xx      | x.xx   | x.xx     |
| Neutral | x.xx      | x.xx   | x.xx     |
| Fear    | x.xx      | x.xx   | x.xx     |

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

## Limitations & Future Work

* Model may not generalize well to noisy or real-world audio
* Limited support for overlapping or mixed emotions
* Future improvements:

  * Real-time emotion detection
  * Support for more languages
  * Larger and more diverse datasets

---

## 🤝 Contributing

Contributions are very welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

If you have any questions or feedback, feel free to reach out:

* GitHub Issues
* Email: [your.email@example.com](mailto:adebayoj383@gmail.com)
