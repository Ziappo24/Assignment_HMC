# Human Pose Estimation for Tai Chi: Active Ageing and Physical User Modelling (PhyUM)

This repository contains the benchmarking software and the LaTeX technical report source code for the **Human Motion Computing** assignment (Course: *Advanced Topics in Computer Science*, Prof.ssa Olga C. Santos, Roma Tre University).

The project performs a comparative evaluation of four Human Pose Estimation (HPE) algorithms applied to tracking elderly subjects during Tai Chi practice in home settings.

---

## 🚀 Repository Content & Software Architecture

The repository is structured to run a Python-based benchmarking pipeline (`benchmark_hpe.py`) that processes video sequences to extract biomechanical landmarks and compute performance metrics.

### Supported Algorithms
1. **MediaPipe Pose**: High-performance model (BlazePose topology, 33 landmarks) optimized for mobile and edge commercial devices.
2. **YOLO Pose (Ultralytics)**: State-of-the-art Top-Down model (17 COCO keypoints), highly robust against loose clothing and complex overlapping body postures.
3. **OpenCV DNN Pose**: Direct Caffe/DNN network wrapper running on CPU.
4. **OpenPose (Simulated)**: A Bottom-Up simulator mimicking the Part Affinity Fields (PAF) computational load to model heavy edge workloads.

---

## 📈 Tracked Metrics
For each video frame, the pipeline computes and logs:
- **Computational Cost**: Inference Latency (ms) and Instantaneous Frames Per Second (FPS).
- **Tracking Quality**: Mean Confidence Score of the detected landmarks.
- **Physical Coverage**: Keypoint Coverage Ratio subdivided into body regions (face, trunk, upper limbs, lower limbs) to map Tai Chi postures.

---

## 📁 Repository Structure
```text
├── benchmark_hpe.py          # Python benchmark orchestration script
├── requirements.txt          # Project dependencies (install via pip)
├── report.tex                # Complete LaTeX Source of the Technical Report
├── performance_comparison.png# Comparative performance visualization chart
├── hpe_benchmark_results.csv # Raw csv containing the compiled benchmark results
├── .gitignore                # Excludes large binaries, virtual envs, and datasets
└── README.md                 # Project Overview (this file)
```

> [!IMPORTANT]
> To comply with GitHub's file size limits, the video datasets and pre-trained model weights are hosted externally on Google Drive. 
> To run the benchmark locally, follow the download and directory structure instructions in the next section.

---

## ⚙️ Quick Start & Setup

### 1. Download Datasets and Model Weights
Please download the `dataset/` and model directories from the Google Drive link:
* **Google Drive Folder**: `[Insert your Google Drive link here]`

### 2. Setup the Workspace
Extract the downloaded files into the root of this project. Your workspace should look like this:
```text
├── benchmark_hpe.py
├── ...
├── dataset/                  # <-- DOWNLOADED FROM DRIVE
│   ├── Posa_1_Difficile/
│   ├── Posa_7_Facile/
│   └── ...
├── models_dnn/               # <-- DOWNLOADED FROM DRIVE
│   ├── pose_deploy_linevec.prototxt
│   └── pose_iter_440000.caffemodel
├── models_mediapipe/         # <-- DOWNLOADED FROM DRIVE
│   └── pose_landmarker_full.task
└── yolov8n-pose.pt           # <-- DOWNLOADED FROM DRIVE
```

### 3. Create Virtual Environment & Install Dependencies
```bash
# Create environment
python -m venv hmc_env

# Activate environment (Windows PowerShell)
.\hmc_env\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 4. Run the Benchmark
To run the automated benchmarking pipeline on your dataset:
```bash
python benchmark_hpe.py
```
This will run the inference engines over the clips, outputting `hpe_benchmark_results.csv` and updating the comparison plots.