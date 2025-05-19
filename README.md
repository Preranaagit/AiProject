# AiProject
Project Overview
This project introduces a hybrid approach to skin cancer classification by combining DINO, a self-supervised Vision Transformer (ViT-B/16), with text-prompt-based learning. The model aligns visual features from skin lesion images with semantic textual prompts (e.g., “a photo of melanoma”) using cosine similarity. This enables accurate classification with minimal reliance on labeled data, making it ideal for deployment in resource-constrained medical environments.
We use the HAM10000 dataset, which contains over 10,000 dermoscopic images labeled across seven skin lesion types, including melanoma, basal cell carcinoma, and benign nevi. Metadata such as patient age, gender, and lesion location is available to support potential multimodal extensions.

The model architecture consists of two main components: an image encoder and a text encoder. The image encoder uses DINO with a ViT-B/16 backbone to extract rich, label-free visual features from dermoscopic images. On the text side, medical prompts such as “a photo of melanoma” are converted into embeddings using a SentenceTransformer model (MiniLM). These image and text embeddings are aligned using either contrastive loss or triplet loss, allowing the model to learn meaningful associations between visual patterns and semantic descriptions. Additionally, a lightweight MLP classifier can be trained on the frozen DINO features to enable direct supervised classification. This dual design allows the model to function both as a prompt-based retrieval system and as a standard classifier, providing flexibility for different medical AI applications.


## Link to Dataset : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data
