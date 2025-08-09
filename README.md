---

## 📌 Project Overview
- **Goal:** Detect human activity in an image.
- **Approach:** Transfer learning with a pre-trained **ResNet18** model.
- **Dataset:**  
  - `Training_set.csv` — training images and labels.
  - `Testing_set.csv` — testing images and labels.
  - Folders:  
    - `data/train/` — training images.  
    - `data/test/` — testing images.
- **Why ResNet18?**  
  - Lightweight and fast for experimentation.
  - Already trained on ImageNet, so it understands general visual features.

---

## 🧠 Model Architecture
- Base: **ResNet18** from `torchvision.models` (pre-trained on ImageNet).
- Modified **final fully connected layer** to output 15 classes.
- All other layers frozen (fine-tuning only last layers).

---

## 🔄 Data Preprocessing
- **Resize** all images to `224x224`.
- **Augmentation** (training only):
  - Random horizontal flip
  - Random rotation
  - Color jitter (brightness & contrast)
- **Normalization**: ImageNet mean & std.

---

## 🏋️ Training
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (lr=0.001, weight decay=1e-4)
- Learning Rate Scheduler: `StepLR` (step=3, gamma=0.7)
- Best model saved to `best_activity_model.pth`.

---

## 📊 Evaluation
- Metrics:
  - Accuracy
  - Precision, Recall, F1-score per class
- Example challenges:
  - Similar actions (e.g., *drinking* vs *laughing*) can be confused.
  - Imbalanced class distribution.

---

## 🚀 How to Run

### 1 Unzip data.zip into the data/ folder before running.
### 2 Install requirements
```bash
pip install -r requirements.txt