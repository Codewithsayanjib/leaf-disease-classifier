# 🌿 Leaf Disease Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### 🚀 Deep Learning · Transfer Learning · Explainable AI · Real-time Inference

**A production-grade plant leaf disease classifier using EfficientNetV2 with GradCAM explainability — trained on 20,000+ images across 15 disease classes.**

[🌐 Live Demo](https://leaf-disease-classifier-zuzyrq9a7z5vc2xfzf8n23.streamlit.app) · [🤗 Model Weights](https://huggingface.co/Sayanjib/leaf-disease-classifier) · [📊 Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

</div>

---

## 📸 Demo

| Uploaded Leaf | GradCAM Heatmap | Prediction |
|:---:|:---:|:---:|
| ![leaf](https://placehold.co/200x200/2d6a4f/white?text=Leaf+Image) | ![cam](https://placehold.co/200x200/d62828/white?text=GradCAM) | `Tomato Early Blight` · `90.67%` |

> 🔥 **GradCAM** highlights exactly *which regions* of the leaf the model focuses on — making predictions transparent and trustworthy.

---

## ✨ Features

- 🧠 **EfficientNetV2-S** pretrained on ImageNet, fine-tuned on PlantVillage
- 🔥 **GradCAM** explainability — visualize model attention on any leaf image
- ⚡ **Apple MPS acceleration** — GPU-accelerated training on Apple Silicon (M1/M2/M3/M4)
- 📊 **Full evaluation suite** — accuracy, F1-score, per-class report, confusion matrix
- 🌐 **Streamlit web app** — upload any leaf image and get instant predictions
- 🤗 **Hugging Face** model hosting — weights auto-downloaded on app startup
- 🎯 **~98% validation accuracy** across 15 disease classes

---

## 🏗️ Project Structure

```
leaf_disease_classifier/
├── 📁 src/
│   ├── dataset.py          # Data loading, augmentation, train/val/test split
│   ├── model.py            # EfficientNetV2-S architecture + checkpoint loading
│   ├── train.py            # Training loop, LR scheduler, best model saving
│   ├── evaluate.py         # Confusion matrix, classification report
│   └── gradcam_viz.py      # Custom GradCAM implementation (no cv2 dependency)
├── 📁 app/
│   └── app.py              # Streamlit web application
├── 📁 data/                # PlantVillage dataset (gitignored)
├── 📁 outputs/
│   ├── plots/              # Training curves, confusion matrix, GradCAM outputs
│   └── checkpoints/        # Saved model weights (gitignored)
├── config.py               # Centralized configuration (paths, hyperparameters)
├── main.py                 # Entry point — trains and evaluates the model
├── requirements.txt
└── packages.txt
```

---

## 🧬 Model Architecture

```
Input (224×224×3)
       ↓
EfficientNetV2-S (pretrained ImageNet)
       ↓
Global Average Pooling
       ↓
Dropout (0.2)
       ↓
Linear (1280 → 15)
       ↓
Softmax → Disease Class
```

| Component | Detail |
|---|---|
| **Base Model** | EfficientNetV2-S (ImageNet pretrained) |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-4) |
| **Scheduler** | Cosine Annealing |
| **Loss** | CrossEntropy + Label Smoothing (0.1) |
| **Augmentation** | RandomCrop, Flip, Rotation, ColorJitter |
| **Device** | Apple MPS / CUDA / CPU |

---

## 🌱 Disease Classes

The model classifies **15 disease categories** across multiple plant species:

| Plant | Diseases |
|---|---|
| 🍅 **Tomato** | Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Bacterial Spot |
| 🥔 **Potato** | Early Blight, Late Blight, Healthy |
| 🌶️ **Pepper** | Bacterial Spot, Healthy |
| + more | Healthy variants across all species |

---

## 📈 Results

| Metric | Score |
|---|---|
| **Validation Accuracy** | ~98% |
| **Test Accuracy** | ~97–98% |
| **Macro F1-Score** | ~0.97 |
| **Training Epochs** | 15 |
| **Training Time (M4 MPS)** | ~45 min |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Codewithsayanjib/leaf-disease-classifier.git
cd leaf-disease-classifier
```

### 2. Set up environment
```bash
conda create -n leafdisease python=3.11 -y
conda activate leafdisease
python -m pip install -r requirements.txt
```

### 3. Download the dataset
```bash
kaggle datasets download -d emmarex/plantdisease -p data/
unzip data/plantdisease.zip -d data/
```

### 4. Train the model
```bash
python main.py
```

### 5. Run the Streamlit app
```bash
streamlit run app/app.py
```

> 💡 The app auto-downloads pretrained weights from Hugging Face — no manual setup needed!

---

## 🔍 GradCAM Explainability

GradCAM (Gradient-weighted Class Activation Mapping) visualizes **which parts of the leaf the model uses** to make its prediction. This makes the model interpretable and trustworthy for real-world agricultural use.

The implementation in this repo is **built from scratch** using only PyTorch — no external CAM libraries required.

```python
from src.gradcam_viz import get_gradcam
from src.model import load_model
import json

with open("models/class_names.json") as f:
    class_names = json.load(f)

model = load_model(len(class_names), "outputs/checkpoints/best_model.pth")
pred_class, confidence = get_gradcam(model, "your_leaf.jpg", class_names)
print(f"Predicted: {pred_class} ({confidence:.2%})")
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Model training & inference |
| ![timm](https://img.shields.io/badge/timm-EfficientNetV2-blue?style=flat) | Pretrained model zoo |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Web application |
| ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) | Model weight hosting |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Metrics & evaluation |
| ![Apple Silicon](https://img.shields.io/badge/Apple_MPS-000000?style=flat&logo=apple&logoColor=white) | GPU acceleration |

---

## 🤗 Pretrained Weights

Model weights are hosted on Hugging Face and auto-downloaded when you run the app:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Sayanjib/leaf-disease-classifier",
    filename="best_model.pth",
    local_dir="outputs/checkpoints"
)
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

<div align="center">

**Sayanjib Sur**

[![GitHub](https://img.shields.io/badge/GitHub-Codewithsayanjib-181717?style=for-the-badge&logo=github)](https://github.com/Codewithsayanjib)

*Built as part of a Computer Vision assignment — designed to be production-ready.*

</div>

---

<div align="center">

⭐ **If you found this useful, please star the repo!** ⭐

</div>
