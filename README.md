
# FAF: Focus-Aware Fusion Network 📸✨

Welcome to the **FAF (Fuse Any Focus)** project! This repository contains the code and model for **multi-focus image fusion (MFIF)**, where our goal is to synthesize a sharp, all-in-focus image from multiple source images with varying focal depths. With the **FAF model** and the newly developed **FAF-1M dataset**, we tackle two critical challenges in MFIF: the scarcity of training data and the **synthetic-to-real domain gap**. 📷🌐

The **FAF model** has shown to outperform current state-of-the-art methods in various benchmarks, both qualitatively and quantitatively. 🚀

---

## Introduction🚀 

**Multi-focus image fusion (MFIF)** aims to combine several source images, each with different focal points, into a single output image that is in-focus across all objects. Current MFIF approaches face two major challenges:

1. **Scarcity of Training Data**: There aren't enough real-world multi-focus image pairs to train deep learning models effectively.
2. **Synthetic-to-Real Domain Gap**: Models trained on synthetic data often struggle to generalize to real-world scenarios.

To address these challenges, we introduce the **Fuse Any Focus (FAF)** model, which leverages both **synthetic** and **real-world** multi-focus image pairs. The **FAF-1M dataset**, the largest MFIF dataset to date, includes over **1,000,000 synthetic** and **2,000 real-world** multi-focus image pairs with ground truth.

To bridge the domain gap, we propose a two-step approach: **pre-training** on synthetic data and **fine-tuning** on real-world data. This ensures the model achieves high performance in real-world MFIF applications. 

The **FAF model** works by predicting a **decision map** that allows for accurate image fusion, improving depth perception and detail. 💡



---

## Quick Start🚀 

### Prerequisites 🛠️

Before running the code, make sure you have the following installed:

- Python 3.9+
- PyTorch (with CUDA support for GPU acceleration) 🔥
- Other Python dependencies (see `requirements.txt`)

```bash
pip install -r requirements.txt
```
## FAF-1M Dataset📚 

**FAF-1M** is the largest **Multi-Focus Image Fusion (MFIF)** dataset, containing over **1,000,000 synthetic** and **2,000 real-world** multi-focus image pairs with ground truth. It was created to address the challenges of limited training data and the synthetic-to-real domain gap in MFIF. The dataset includes diverse scenes generated via optical simulation and captured using the proprietary **FAF data engine**, making it ideal for training and evaluating fusion models.

### Download the FAF-1M Dataset:
[Download FAF-1M Dataset]

## ⚡ Pretrained & Fine-tuned Weights

To help you get started quickly, we provide both **pretrained** and **fine-tuned** weights for the FAF model. 

- **Pretrained Weights**: These weights were trained on the synthetic **FAF-1M** dataset.
- **Fine-tuned Weights**: These weights were further fine-tuned on FAF-Real-2k real-world data to bridge the synthetic-to-real domain gap.

### Download Links:
- [Download FAF-Pretrained Weights]
- [Download FAF-Fine-tuned Weights]


## Training the Model 🏋️‍♀️

1. **Prepare your dataset**: Ensure your dataset is ready (e.g., **FAF-1M**).
   
2. **Train the model**:

```bash
python train.py --root_path ./path/to/FAF-1M --batch_size 2 --max_epochs 10 --base_lr 0.000001 --n_gpu 1
```

Feel free to modify the arguments to suit your needs. For example:
- **`--max_epochs`**: The number of training epochs.
- **`--base_lr`**: The learning rate for the optimizer.
- **`--batch_size`**: The batch size for training.

The model will be trained and saved to the specified directory for future use. 📈

---

## Testing the Model 🧪

After training your model, you can use it to test on new data. Here's how you can test your trained FAF model:

1. **Prepare your test dataset**: Make sure your test images are in the right format.

2. **Run the test script**:

```bash
python test.py --root_path ./path/to/FAF-1M-val
```

The results will be saved in a directory called `results/`, with output images showing the fused focus mask and the final fused image. 💥

---
## 🧑‍🤝‍🧑 Contributing

We welcome contributions! If you find a bug, have an idea for improvement, or want to enhance the documentation, feel free to open an issue or create a pull request. 🤗

---

## 💬 Contact

If you have any questions or need help, don't hesitate to reach out. You can contact us via the GitHub issues page or email at [your-email@example.com]. ✉️

---

## 📑 Citation

If you use this repository in your research, please cite the following paper:

```
@article{faftest,
  author = {xxx},
  title = {Fuse Any Focus: Multi-focus Image Fusion via Focal Plane Segmentation},
  journal = {xx},
  year = {2024},
  volume = {X},
  pages = {XX-XX}
}
```

---

## 📂 License

The code and dataset are released under the **MIT License**. Feel free to use, modify, and distribute the code with attribution. 🔓

