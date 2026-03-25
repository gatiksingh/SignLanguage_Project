# Real-Time Sign Language Recognition System

This project implements a real-time sign language recognition system that translates dynamic hand gestures into text and speech using a webcam
Built entirely from scratch, the system avoids dependencies on skeletal landmark libraries like MediaPipe, relying instead on a custom deep learning pipeline to capture both spatial appearance and temporal motion.

---

##  Key Features

* **Custom Hand Extraction:** Uses skin-color HSV thresholding and morphological cleaning to isolate hands without external libraries.
* **Hybrid Architecture:** Combines a **TimeDistributed CNN** for spatial features with a **Bidirectional GRU** to track movement over time.
* **Real-Time Inference:** Features a sliding window buffer and temporal smoothing to ensure stable, high-confidence predictions.
* **Text-to-Speech (TTS):** Integrated `pyttsx3` support to announce recognized gestures aloud.
* **Robust Data Collection:** An interactive script for recording labeled training sequences with live bounding-box previews.

---

## 🛠 Technical Architecture

The system treats sign language as a temporal sequence: the meaning of a sign emerges from how the hand moves over a short window of time, rather than from a single frame.

### System Pipeline
1.  **Data Collection:** Captures labeled sequences (fixed number of frames) directly from a webcam.
2.  **Model Training:** Uses a custom Keras generator with data augmentation (temporal dropping, reversal, and jitter) to train the model.
3.  **Inference:** Runs frame-by-frame on live input using a deque buffer for continuous predictions.

### Neural Network Layers
| Component | Details |
| :--- | :--- |
| **Input** | Normalized grayscale video clips: `(batch, SEQUENCE_LENGTH, H, W, 1)`. |
| **Feature Extractor** | 4-block TimeDistributed CNN using ReLU activation and Max-Pooling. |
| **Temporal Encoder** | Two stacked Bidirectional GRU layers (256 and 128 total units). |
| **Classifier Head** | Dense layers with Dropout (0.5 and 0.3) and a Softmax output. |

---

## 📂 File Structure

| File | Role |
| :--- | :--- |
| `config.py` | Central configuration for paths, action labels, and image sizes. |
| `collect_data.py` | Interactive tool for recording training sequences. |
| `train.py` | Model definition, data augmentation, and training logi. |
| `inference.py` | Real-time recognition script with smoothing and TTS. |
| `scratch_sign_model.h5` | Trained Keras model weights. |

---

## ⚠️ Limitations & Future Work

* **Sensitivity:** The fixed HSV skin range can be sensitive to lighting variations and specific skin tones.
* **Single Hand:** The algorithm currently targets the largest detected contour, limiting support for two-handed signs.
* **Grammar:** The system recognizes individual gesture labels rather than constructing full grammatical sentences.
* **Future Improvements:** Planned updates include adaptive skin thresholding, MediaPipe integration for robustness, and mobile deployment via TensorFlow Lite.

---

