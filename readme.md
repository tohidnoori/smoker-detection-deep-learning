#  Smoker Detection using Deep Learning 🚬

This project builds a deep learning model to classify whether a person is a **smoker** or **non-smoker** from images.  
It compares different CNN architectures and transfer learning with **EfficientNetB0**.

---

## Dataset
- Source: [Cigarette Smoker Detection (Kaggle)](https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection)  
- Two classes:
  - **Smoking**
  - **Not Smoking**  
- Images resized to **256×256**  
- Split: **80% training / 20% testing**

---

## Preprocessing
- Mounted Google Drive for dataset storage.
- Resized all images to **256×256 pixels** using OpenCV.
- Normalized pixel values to `[0, 1]`.
- Converted to TensorFlow `tf.data.Dataset` pipeline.

---

## Models & Results

### Model 1: Basic CNN
- Architecture: Conv2D → LeakyReLU → MaxPooling → Dense
- Parameters: **11.36M (43.34 MB)**  
- Model size: **21.67 MB**  
- **Results:**
  - Accuracy: **0.8507**
  - F1-score: **0.8585**
  - Precision: **0.9262**
  - Recall: **0.8000**
- ⚠️ Overfitted

---

### Model 2: CNN with Dropout & BatchNormalization
- Added:
  - Batch Normalization
  - Dropout for regularization  
- Parameters: **6.72M (25.64 MB)**  
- Model size: **8.55 MB**  
- **Results:**
  - Accuracy: **0.8818**
  - F1-score: **0.8962**
  - Precision: **0.8913**
  - Recall: **0.9011**
- ✅ Reduced overfitting

---

### Model 3: Transfer Learning with EfficientNetB0
- Pretrained **EfficientNetB0 (ImageNet weights)**
- Added: GlobalAveragePooling + Dense + Dropout
- Parameters: **4.21M (16.07 MB)**  
- Model size: **16.07 MB**  
- **Results:**
  - Accuracy: **0.9192**
  - F1-score: **0.9294**
  - Precision: **0.9185**
  - Recall: **0.9407**
- ⭐ Best performing model

---

## Plot
Random 20 validation images were visualized:

- ✅ Correct predictions shown in **green**  
- ❌ Incorrect predictions shown in **red**

```markdown
![Sample Predictions]()
