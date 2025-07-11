# **COMSYS Hackathon5 **

## **Execution Guide**

### **Task A – Gender Classification from Facial Images**

* **Script**: `task_A_submission.py`

* **Steps to Run**:

  1. Ensure your dataset is arranged in the expected folder layout.
  2. Update the `train_dir` and `val_dir` variables at the top of the script if your paths differ.
  3. Launch the script using the following command:

     ```bash
     python task_A_submission.py
     ```

* **Expected Outputs**:

  * Final evaluation statistics:

    * Accuracy
    * Precision
    * Recall
    * F1 Score
  * The best-performing model will be saved as `best_gender_model.pt`

---

### **Task B – Identity Verification Using Face Images**

* **Script**: `task_B_submission.py`

* **Steps to Run**:

  1. Ensure the face images are correctly organized under `train/` and `val/` directories.
  2. Modify the directory paths (`train_dir`, `val_dir`) at the top of the script if needed.
  3. Run the script using:

     ```bash
     python task_B_submission.py
     ```

* **Expected Outputs**:

  * Evaluation metrics:

    * Accuracy
    * Precision
    * Recall
    * F1 Score

> Note: This task doesn’t involve model training. Instead, it uses embeddings from a pre-trained FaceNet-based model to perform face pair verification based on cosine similarity and an ROC-determined threshold.

---

## **Model Design Overview**

### **Task A – Gender Classification**

* **Base Model**: EfficientNet-B0 (ImageNet pretrained)
* **Classification Layer**: Replaced `_fc` with `nn.Linear(in_features=1280, out_features=2)`
* **Training Configuration**:

  * Loss: CrossEntropyLoss
  * Optimizer: Adam (`lr=2e-4`)
  * Scheduler: CosineAnnealingLR
* **Model Saving**: The best version based on validation performance is saved as `best_gender_model.pt`

---

### **Task B – Face Matching**

* **Face Cropping**: MTCNN is used for face detection and alignment
* **Embedding Generation**: InceptionResnetV1 (pre-trained)
* **Verification Strategy**:

  * Cosine similarity is computed for each embedding pair
  * Threshold for similarity is chosen using ROC curve analysis on training pairs
  * Final classification is based on whether similarity exceeds this threshold

## **Environment Setup**

Install all required packages with:

```bash
pip install torch torchvision facenet-pytorch efficientnet-pytorch scikit-learn numpy pillow tqdm matplotlib
```

---

Let me know if you'd like this turned into a PDF report or added to a GitHub README with formatting enhancements.
