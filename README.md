# Two-Stream Real-Time Violence Detection

**A robust, privacy-preserving, and computationally efficient system for detecting violent activity in real-time video streams.**

## Overview

Conventional surveillance systems are often reactive and inefficient, recording hours of inactivity. This project implements a proactive violence detection system designed for real-time application.

The solution utilizes a Dual-Stream Deep Learning Architecture that processes:

1.  **Frame Difference (Change Detection):** Captures motion dynamics and eliminates static background noise.
2.  **Skeleton/Pose Data:** To capture specific human body movements using RTMO.

By integrating a motion-detection filtering module, the system achieves significant computational savings by processing data only when relevant activity is detected.

## Key Features

- **Dual-Stream Fusion:** Combines temporal motion features (from frame differences) with skeleton data for high-accuracy action recognition.
- **Smart Motion Filtering:** Automatically filters out static scenes to save GPU resources.
- **Real-Time Inference:** Optimized threading pipeline allows for continuous monitoring with minimal latency.
- **Privacy-Aware:** Heavily relies on skeleton data and frame differences, reducing reliance on raw visual details compared to pure RGB methods.

## Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Pose Estimation:** RTMO (via RTMLib)
- **Computer Vision:** OpenCV
- **Language:** Python 3.x

## Datasets

The model was trained and evaluated using the following datasets:

### Training Data: RWF-2000 (https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)

- **Source:** Large-scale video database collected from real-world surveillance cameras on YouTube.
- **Size:** 2,000 clips (1,000 Violent, 1,000 Non-Violent).
- **Format:** 5-second clips at 30 FPS.
- **Usage:** Used for training the core Deep Learning model to recognize real-world violence patterns.

### Testing Data: Surveillance Camera Fight Dataset (https://github.com/seymanurakti/fight-detection-surv-dataset)

- **Source:** A collection of YouTube videos containing specific fight instances and regular surveillance footage.
- **Size:** 300 videos total (150 Fight, 150 Non-Fight).
- **Format:** 2-second sequences.
- **Usage:** Used as an independent test set to validate the model's performance on unseen data.

## System Architecture

The model implements a custom **CNN-ConvLSTM** architecture with a specialized preprocessing stage:

1.  **Preprocessing (Frame Grouping):**

    - Both streams (Change Detection and Skeleton) undergo a grouping process.
    - Frames are first converted to **grayscale**.
    - Every **3 consecutive grayscale frames** are stacked to form a single 3-channel image (Pseudo-RGB).
    - This technique allows the spatial extractors to capture short-term temporal evolution before the data reaches the LSTM layers.

2.  **Inputs:** The model accepts two streams of these 16-frame sequences (where each "frame" is actually a group of 3 timestamps).

3.  **Spatial Feature Extraction:** Uses `SeparableConv2D` blocks to extract spatial features from the grouped inputs efficiently.

4.  **Temporal Modeling:** Utilizes `ConvLSTM2D` layers to capture the longer-term dependencies of actions and motion sequences.

5.  **Fusion:** The motion stream and skeleton stream are fused via an `Add` layer before final classification.

## Performance and Results

The system was tested on videos with static cameras and varying levels of activity.

| Metric          | Result      | Note                                                        |
| :-------------- | :---------- | :---------------------------------------------------------- |
| **F1-Score**    | 78.62%      | Balanced precision and recall.                              |
| **Latency**     | ~0.98s      | Maximum processing latency (end-to-end).                    |
| **GPU Savings** | Up to 90.8% | Compared to continuous processing without motion filtering. |


