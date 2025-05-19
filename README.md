# Skin Lesion Classification using Self-Supervised Vision Transformers (DINO)

This project explores a hybrid deep learning pipeline for classifying dermoscopic skin lesion images using self-supervised learning. We leverage the DINO (Self-Distillation with No Labels) framework built on a ViT-B/16 Vision Transformer to extract robust visual features without relying on labeled data. A MiniLM-based text encoder is also used to enable prompt-based semantic retrieval.

---

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Preprocessing and Augmentation](#preprocessing-and-augmentation)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Sustainable Development Goals (SDG) Impact](#sustainable-development-goals-sdg-impact)
- [Installation and Requirements](#installation-and-requirements)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Authors](#authors)

---

## Objective

To build a lightweight, accurate skin lesion classifier using features from a pretrained DINO model and a MiniLM text encoder. This dual approach supports both standard classification and text-prompt-based image retrieval, reducing the need for large-scale annotated medical datasets.

---

## Dataset

We use the HAM10000 ("Human Against Machine with 10000 training images") dataset:
- Contains 10,015 dermoscopic images across 7 skin lesion classes.
- Each image is associated with metadata such as age, gender, and lesion location.

Link to dataset: [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## Preprocessing and Augmentation

- All images are resized to 224x224 pixels to match DINO input requirements.
- The following augmentations are applied:
  - `RandomResizedCrop(224, scale=(0.2, 1.0))`
  - `RandomHorizontalFlip(p=0.5)`
  - `ColorJitter` (brightness, contrast, saturation, hue) with `p=0.8`
  - `RandomGrayscale(p=0.2)`
  - `GaussianBlur(p=0.5)`
  - Normalization to mean = 0.5, standard deviation = 0.5

---

## Model Architecture

### Image Encoder  
- DINO ViT-B/16 pretrained on ImageNet using self-supervised learning.  
- Downloaded from the official DINO GitHub repository:  
  [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)

### Text Encoder  
- A MiniLM-based encoder (`all-MiniLM-L6-v2`) from the SentenceTransformer library is used to convert medical text prompts into semantic embeddings.
- These embeddings are aligned with image features using cosine similarity or contrastive learning, enabling cross-modal retrieval.

### Classifier  
- A fully connected neural network (MLP) is trained on frozen DINO features for supervised classification.

---

## Training Strategy

- Dataset split: 80% training, 10% validation, 10% test
- Class imbalance addressed through:
  - Augmentation of minority classes
  - Stratified sampling to maintain class distribution
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Regularization techniques: Dropout and Early Stopping

---

## Results

- Achieved high classification performance using frozen DINO features.
- Enabled text-to-image retrieval via MiniLM embeddings, demonstrating early multimodal capabilities.
- Suitable for low-resource medical AI settings.

---

## Sustainable Development Goals (SDG) Impact

This project contributes to the following United Nations Sustainable Development Goals:

- **SDG 3 – Good Health and Well-being:**  
  Enhancing early detection of skin cancer using AI improves health outcomes and access to diagnostic tools.

- **SDG 4 – Quality Education:**  
  This project can serve as a learning tool in AI for healthcare, enabling students and researchers to explore real-world ML applications.

- **SDG 9 – Industry, Innovation and Infrastructure:**  
  Demonstrates how self-supervised learning can optimize medical imaging systems, especially in low-resource settings.

- **SDG 10 – Reduced Inequalities:**  
  By reducing the dependency on labeled data, this project helps create diagnostic tools accessible to underserved communities.

- **SDG 17 – Partnerships for the Goals:**  
  Encourages interdisciplinary collaboration between AI, healthcare, and policy experts to create meaningful impact.

---

## Installation and Requirements

To reproduce this project, install the following libraries:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install scikit-learn pandas matplotlib seaborn plotly
pip install sentence-transformers
pip install opencv-python
```

For Jupyter Notebook usage:

```bash
pip install notebook ipywidgets
```

---

## Future Work

- Fine-tune DINO and MiniLM jointly for more robust cross-modal understanding
- Expand text-prompt library to include metadata (e.g., location, patient history)
- Integrate contrastive loss or triplet loss for better alignment between modalities

---

## Acknowledgments

We gratefully acknowledge the following sources and communities:

- **DINO framework** from Facebook AI Research for enabling self-supervised learning:  
  [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)

- **HAM10000 Dataset** provided by Tschandl, Rosendahl, and Kittler for academic use:  
  [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

- **MiniLM** by Microsoft for efficient text encoding:  
  [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
