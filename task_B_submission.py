import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Device and model setup ---
# Automatically choose GPU if available, else default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MTCNN for face detection (detects and aligns a single face per image)
mtcnn = MTCNN(keep_all=False, device=device)

# Load FaceNet model (InceptionResnetV1) pretrained on VGGFace2 for embedding generation
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --- Function to extract face embedding from image ---
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")  # Load image and ensure RGB format
        face = mtcnn(img)                          # Detect and align face
        if face is None:
            return None                            # Skip if no face is detected
        with torch.no_grad():                      # Disable gradient tracking for inference
            emb = facenet(face.unsqueeze(0).to(device))  # Generate embedding
        return emb.cpu().numpy().flatten()         # Return as a flat NumPy array
    except:
        return None                                # Return None if any error occurs

# --- Function to load all face embeddings from a directory ---
def load_embeddings(folder):
    embeddings, labels, img_paths = [], [], []
    persons = os.listdir(folder)  # Each subfolder is a separate identity
    for person in tqdm(persons, desc=f"Processing {folder}"):
        person_dir = os.path.join(folder, person)
        for root, _, files in os.walk(person_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, file)
                    emb = get_embedding(path)
                    if emb is not None:
                        embeddings.append(emb)
                        labels.append(person)
                        img_paths.append(path)
    return np.array(embeddings), np.array(labels), img_paths

# --- Generate positive (same person) and negative (different person) pairs ---
def generate_pairs(embeddings, labels):
    positives, negatives = [], []
    unique_labels = np.unique(labels)
    label_to_idx = {label: np.where(labels == label)[0] for label in unique_labels}

    # Create all possible positive pairs (same identity)
    for label, idxs in label_to_idx.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    positives.append((idxs[i], idxs[j]))

    # Randomly create negative pairs (different identities)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            idx1 = np.random.choice(label_to_idx[unique_labels[i]])
            idx2 = np.random.choice(label_to_idx[unique_labels[j]])
            negatives.append((idx1, idx2))

    return positives, negatives

# --- Cosine similarity function between two embeddings ---
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Define dataset paths (adjust these if running locally)
    train_dir = "/content/Comsys_Hackathon5/Task_B/train"
    val_dir = "/content/Comsys_Hackathon5/Task_B/val"

    # --- Extract embeddings from training and validation datasets ---
    X_train, y_train, train_paths = load_embeddings(train_dir)
    X_val, y_val, val_paths = load_embeddings(val_dir)

    # Combine all embeddings and labels
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    paths_all = train_paths + val_paths

    # --- Create verification pairs ---
    pos_pairs, neg_pairs = generate_pairs(X_all, y_all)
    n_pairs = min(len(pos_pairs), len(neg_pairs))  # Balance number of positive and negative pairs
    pairs = pos_pairs[:n_pairs] + neg_pairs[:n_pairs]
    pair_labels = [1] * n_pairs + [0] * n_pairs     # 1 for match, 0 for non-match

    # --- Compute cosine similarity for each pair ---
    sims = np.array([cosine_sim(X_all[i], X_all[j]) for i, j in pairs])

    # Split data into training and testing sets (80/20 split)
    X_train_sim, X_test_sim, y_train_sim, y_test_sim = train_test_split(
        sims, pair_labels, test_size=0.2, random_state=42
    )

    # --- Determine the best threshold using ROC curve on training data ---
    fpr, tpr, thresholds = roc_curve(y_train_sim, X_train_sim)
    optimal_idx = np.argmax(tpr - fpr)         # Find threshold that maximizes TPR - FPR
    optimal_thresh = thresholds[optimal_idx]   # Select optimal similarity threshold

    # Apply threshold to predict classes
    y_train_pred = (X_train_sim >= optimal_thresh).astype(int)
    y_test_pred = (X_test_sim >= optimal_thresh).astype(int)

    # --- Compute metrics for training set ---
    train_acc = accuracy_score(y_train_sim, y_train_pred)
    train_prec = precision_score(y_train_sim, y_train_pred)
    train_rec = recall_score(y_train_sim, y_train_pred)
    train_f1 = f1_score(y_train_sim, y_train_pred)

    # --- Compute metrics for validation set ---
    test_acc = accuracy_score(y_test_sim, y_test_pred)
    test_prec = precision_score(y_test_sim, y_test_pred)
    test_rec = recall_score(y_test_sim, y_test_pred)
    test_f1 = f1_score(y_test_sim, y_test_pred)

    # --- Display final evaluation results ---
    print("\n=== Final Evaluation (Task B) ===")

    print("\n-- Training Set Metrics --")
    print(f"Accuracy : {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall   : {train_rec:.4f}")
    print(f"F1 Score : {train_f1:.4f}")

    print("\n-- Validation Set Metrics --")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
