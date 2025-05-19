# AiProject
Project Overview
This project introduces a hybrid approach to skin cancer classification by combining DINO, a self-supervised Vision Transformer (ViT-B/16), with text-prompt-based learning. The model aligns visual features from skin lesion images with semantic textual prompts (e.g., “a photo of melanoma”) using cosine similarity. This enables accurate classification with minimal reliance on labeled data, making it ideal for deployment in resource-constrained medical environments.

We use the HAM10000 dataset, which contains over 10,000 dermoscopic images labeled across seven skin lesion types, including melanoma, basal cell carcinoma, and benign nevi. Metadata such as patient age, gender, and lesion location is available to support potential multimodal extensions.

Model Architecture
Image Encoder: DINO with ViT-B/16 extracts rich, label-free visual features.

Text Encoder: SentenceTransformer (MiniLM) converts medical prompts into embeddings.

Loss Functions: Trained using either contrastive loss or triplet loss to align image and prompt embeddings.

Optional Classifier: A lightweight MLP can also be trained on frozen DINO features for supervised classification.

Dual Use: Supports both direct classification and semantic prompt-based image retrieval, enabling flexible applications.


## Link to Dataset : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data
