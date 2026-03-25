# Real-Time Sign Language Recognition System

[cite_start]This project implements a real-time sign language recognition system that translates dynamic hand gestures into text and speech using a webcam[cite: 5]. [cite_start]Built entirely from scratch, the system avoids dependencies on skeletal landmark libraries like MediaPipe, relying instead on a custom deep learning pipeline to capture both spatial appearance and temporal motion[cite: 6, 8, 115].

---

##  Key Features

* [cite_start]**Custom Hand Extraction:** Uses skin-color HSV thresholding and morphological cleaning to isolate hands without external libraries[cite: 64, 67, 116].
* [cite_start]**Hybrid Architecture:** Combines a **TimeDistributed CNN** for spatial features with a **Bidirectional GRU** to track movement over time[cite: 3, 34, 39].
* [cite_start]**Real-Time Inference:** Features a sliding window buffer and temporal smoothing to ensure stable, high-confidence predictions[cite: 95, 104, 107].
* [cite_start]**Text-to-Speech (TTS):** Integrated `pyttsx3` support to announce recognized gestures aloud[cite: 28, 109].
* [cite_start]**Robust Data Collection:** An interactive script for recording labeled training sequences with live bounding-box previews[cite: 13, 17].

---

## 🛠 Technical Architecture

[cite_start]The system treats sign language as a temporal sequence: the meaning of a sign emerges from how the hand moves over a short window of time, rather than from a single frame[cite: 7].

### System Pipeline
1.  [cite_start]**Data Collection:** Captures labeled sequences (fixed number of frames) directly from a webcam[cite: 13, 14].
2.  [cite_start]**Model Training:** Uses a custom Keras generator with data augmentation (temporal dropping, reversal, and jitter) to train the model[cite: 22, 84].
3.  [cite_start]**Inference:** Runs frame-by-frame on live input using a deque buffer for continuous predictions[cite: 25, 96].

### Neural Network Layers
| Component | Details |
| :--- | :--- |
| **Input** | [cite_start]Normalized grayscale video clips: `(batch, SEQUENCE_LENGTH, H, W, 1)`[cite: 31, 33]. |
| **Feature Extractor** | [cite_start]4-block TimeDistributed CNN using ReLU activation and Max-Pooling[cite: 35, 36]. |
| **Temporal Encoder** | [cite_start]Two stacked Bidirectional GRU layers (256 and 128 total units)[cite: 42, 43]. |
| **Classifier Head** | [cite_start]Dense layers with Dropout (0.5 and 0.3) and a Softmax output[cite: 45, 46]. |

---

## 📂 File Structure

| File | Role |
| :--- | :--- |
| `config.py` | [cite_start]Central configuration for paths, action labels, and image sizes[cite: 112]. |
| `collect_data.py` | [cite_start]Interactive tool for recording training sequences[cite: 112]. |
| `train.py` | [cite_start]Model definition, data augmentation, and training logic[cite: 112]. |
| `inference.py` | [cite_start]Real-time recognition script with smoothing and TTS[cite: 112]. |
| `scratch_sign_model.h5` | [cite_start]Trained Keras model weights[cite: 112]. |

---

## ⚠️ Limitations & Future Work

* [cite_start]**Sensitivity:** The fixed HSV skin range can be sensitive to lighting variations and specific skin tones[cite: 133, 135].
* [cite_start]**Single Hand:** The algorithm currently targets the largest detected contour, limiting support for two-handed signs[cite: 134].
* [cite_start]**Grammar:** The system recognizes individual gesture labels rather than constructing full grammatical sentences[cite: 136].
* [cite_start]**Future Improvements:** Planned updates include adaptive skin thresholding, MediaPipe integration for robustness, and mobile deployment via TensorFlow Lite[cite: 138, 139, 143].

---

**Would you like me to generate a detailed `requirements.txt` file based on the listed dependencies?**
