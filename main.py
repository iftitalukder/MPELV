import numpy as np
import pandas as pd
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from scipy.stats import kurtosis, skew, entropy


# ===============================
# MPELV++ Accuracy-Optimized
# ===============================
class MPELVpp:
    def __init__(self, W=200, H=400, num_classes=6, reg_lambda=0.25):
        self.W = W
        self.H = H
        self.num_classes = num_classes
        self.reg_lambda = reg_lambda
        # now 10 features → H hidden neurons
        self.hidden_weights = np.random.randn(10, H) * 0.5
        self.beta = None
        self.scaler = None
        self.feature_importances_global = np.zeros(10)
        self.feature_importances_class = np.zeros((num_classes, 10))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def extract_features(self, x, y, z):
        if len(x) != self.W or len(y) != self.W or len(z) != self.W:
            raise ValueError(f"Window size must be {self.W}.")
        m = np.sqrt(x**2 + y**2 + z**2)

        # Core features
        var_m = np.log1p(np.var(m))
        mean_m = np.log1p(np.mean(m))
        max_m = np.max(m)
        mad_m = np.median(np.abs(m - np.median(m)))

        # Frequency features
        fft_m = np.abs(np.fft.fft(m))[:self.W//2]
        freqs = np.fft.fftfreq(self.W, d=0.01)[:self.W//2]
        spectral_centroid = np.sum(freqs * fft_m) / np.sum(fft_m) if np.sum(fft_m) > 0 else 0
        spectral_centroid = np.clip(spectral_centroid, 0, 50)
        spectral_energy = np.log1p(np.sum(fft_m**2) / self.W)

        # Additional discriminative features
        rms = np.sqrt(np.mean(m**2))
        sk = skew(m)
        hist, _ = np.histogram(m, bins=30, density=True)
        ent = entropy(hist + 1e-12)  # avoid log(0)
        kurto = kurtosis(m)

        return np.array([var_m, mean_m, max_m, mad_m,
                         spectral_centroid, spectral_energy,
                         rms, sk, ent, kurto])

    def _hidden_layer(self, features):
        return self.sigmoid(np.dot(features, self.hidden_weights))

    def train(self, features, labels, class_weights=None):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        enc = OneHotEncoder(sparse_output=False, categories=[range(self.num_classes)])
        Y_onehot = enc.fit_transform(labels.reshape(-1, 1))

        if class_weights is not None:
            Y_onehot = Y_onehot * class_weights[labels].reshape(-1, 1)

        H_matrix = self._hidden_layer(X_scaled)
        I = np.eye(H_matrix.shape[1])
        self.beta = np.linalg.inv(H_matrix.T @ H_matrix + self.reg_lambda * I) @ (H_matrix.T @ Y_onehot)

    def classify(self, features):
        f_scaled = self.scaler.transform([features])[0]
        h = self._hidden_layer(f_scaled)
        o = np.dot(h, self.beta)
        return np.argmax(o), o

    def explain(self, features, true_label=None):
        f_scaled = self.scaler.transform([features])[0]
        contributions = np.sum(np.abs(self.hidden_weights) * np.abs(f_scaled[:, None]), axis=1)
        self.feature_importances_global += contributions
        if true_label is not None:
            self.feature_importances_class[true_label] += contributions
        return contributions


# ===============================
# Supporting Functions
# ===============================
def preprocess_data(df, vib_cols=['X', 'Y', 'Z']):
    df = df.copy()
    for col in vib_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def create_non_overlapping_windows(df, W=200, class_col='class', vib_cols=['X', 'Y', 'Z']):
    features, labels = [], []
    n = len(df)
    for start in range(0, n, W):
        end = start + W
        if end > n: break
        window = df.iloc[start:end]
        if len(window) != W: continue
        x, y, z = window[vib_cols[0]].values, window[vib_cols[1]].values, window[vib_cols[2]].values
        window_classes = window[class_col].values
        counts = np.bincount(window_classes.astype(int), minlength=6)
        class_mode = np.argmax(counts)
        if counts[class_mode] < W * 0.8:
            continue
        f = MPELVpp(W=W).extract_features(x, y, z)
        features.append(f)
        labels.append(class_mode)
    return np.array(features), np.array(labels)

def save_plot(fig, filename):
    os.makedirs('results/figures', exist_ok=True)
    fig.savefig(f'results/figures/{filename}.png', dpi=300, bbox_inches='tight')
    pickle.dump(fig, open(f'results/figures/{filename}.fig', 'wb'))


# ===============================
# Evaluation + Plots
# ===============================
def evaluate_model(y_true, y_pred, y_proba, class_names, set_name="Test"):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    f1_m = f1_score(y_true, y_pred, average='macro')
    print(f"\n=== {set_name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_w:.4f}")
    print(f"Macro F1: {f1_m:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{set_name} Confusion Matrix")
    save_plot(fig, f"{set_name.lower()}_confusion_matrix")
    plt.close(fig)

    return acc, f1_w, f1_m


# ===============================
# Pipeline
# ===============================
def run_mpelvpp_pipeline(file_path='balanced_dataset.csv', W=200, test_size=0.2, val_size=0.2):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)

    features, labels = create_non_overlapping_windows(df, W=W)
    class_names = ['Normal'] + [f'Fault {i}' for i in range(1, 6)]
    feature_names = ['Var','Mean','Max','MAD','Centroid','Energy','RMS','Skew','Entropy','Kurtosis']

    unique, counts = np.unique(labels, return_counts=True)
    class_weights = {i: 1.0/count for i, count in zip(unique, counts)}
    class_weights[3] *= 2.0   # slightly stronger boost for Fault 3
    class_weights = np.array([class_weights.get(i, 1.0) for i in labels])

    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size,
                                                      stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size),
                                                      stratify=y_temp, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = MPELVpp(W=W, H=400, reg_lambda=0.25)

    start_train = time.time()
    model.train(X_train, y_train, class_weights=class_weights)
    train_time = time.time() - start_train

    # Validation
    y_val_pred, y_val_proba = [], []
    for f, y in zip(X_val, y_val):
        pred, raw = model.classify(f)
        y_val_pred.append(pred)
        y_val_proba.append(np.exp(raw)/np.sum(np.exp(raw)))
        model.explain(f, true_label=y)
    val_acc, val_f1_w, val_f1_m = evaluate_model(y_val, np.array(y_val_pred), np.array(y_val_proba), class_names, "Validation")

    # Test
    start_infer = time.time()
    y_test_pred, y_test_proba = [], []
    for f, y in zip(X_test, y_test):
        pred, raw = model.classify(f)
        y_test_pred.append(pred)
        y_test_proba.append(np.exp(raw)/np.sum(np.exp(raw)))
        model.explain(f, true_label=y)
    infer_time = (time.time() - start_infer) / len(X_test)
    test_acc, test_f1_w, test_f1_m = evaluate_model(y_test, np.array(y_test_pred), np.array(y_test_proba), class_names, "Test")

    # Model size
    param_size = (model.hidden_weights.nbytes +
                  model.beta.nbytes +
                  model.scaler.mean_.nbytes +
                  model.scaler.scale_.nbytes) / (1024 * 1024)

    # Explainability normalization
    global_sum = np.sum(model.feature_importances_global)
    global_imp = (model.feature_importances_global / global_sum) if global_sum > 0 else np.zeros_like(model.feature_importances_global)
    class_imp = model.feature_importances_class

    os.makedirs('results/metrics', exist_ok=True)
    explain_path = 'results/metrics/feature_importances.json'
    with open(explain_path, 'w') as f:
        json.dump({
            "global": dict(zip(feature_names, global_imp.round(5).tolist())),
            "per_class": {class_names[i]: dict(zip(feature_names,
                                                  ((class_imp[i] / np.sum(class_imp[i])) if np.sum(class_imp[i]) > 0 else np.zeros_like(class_imp[i])).round(5).tolist()))
                          for i in range(len(class_names))}
        }, f, indent=4)

    os.makedirs('results/model', exist_ok=True)
    model_path = "results/model/mpelvpp_model_final.npz"
    np.savez(model_path, beta=model.beta, hidden_weights=model.hidden_weights,
             scaler_mean=model.scaler.mean_, scaler_scale=model.scaler.scale_,
             W=model.W, H=model.H, class_names=class_names)

    overall_results = {
        "validation": {"accuracy": float(val_acc), "weighted_f1": float(val_f1_w), "macro_f1": float(val_f1_m)},
        "test": {"accuracy": float(test_acc), "weighted_f1": float(test_f1_w), "macro_f1": float(test_f1_m),
                 "classification_report": classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)},
        "efficiency": {"training_time_sec": round(train_time, 5),
                       "avg_inference_time_sec": round(infer_time, 8),
                       "model_size_MB": round(param_size, 5)},
        "explainability_file": explain_path,
        "plots": {"global_importances": "results/figures/global_feature_importances.png",
                  "heatmap": "results/figures/feature_importances_heatmap.png",
                  "val_confusion": "results/figures/validation_confusion_matrix.png",
                  "test_confusion": "results/figures/test_confusion_matrix.png"},
        "model_file": model_path
    }
    results_json_path = "results/metrics/overall_results.json"
    with open(results_json_path, "w") as f:
        json.dump(overall_results, f, indent=4)

    return model, test_acc, test_f1_w


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    try:
        model, acc, f1 = run_mpelvpp_pipeline('balanced_dataset.csv', W=200)
        print(f"\nFinal Test Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")
    except Exception as e:
        print("Error:", e)
