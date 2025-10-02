import numpy as np
import pandas as pd
import json
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


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


def extract_features(x, y, z, W=200):
    m = np.sqrt(x**2 + y**2 + z**2)
    var_m = np.log1p(np.var(m))
    mean_m = np.log1p(np.mean(m))
    max_m = np.max(m)
    mad_m = np.median(np.abs(m - np.median(m)))
    fft_m = np.abs(np.fft.fft(m))[:W//2]
    freqs = np.fft.fftfreq(W, d=0.01)[:W//2]
    spectral_centroid = np.sum(freqs * fft_m) / np.sum(fft_m) if np.sum(fft_m) > 0 else 0
    spectral_centroid = np.clip(spectral_centroid, 0, 50)
    spectral_energy = np.log1p(np.sum(fft_m**2) / W)
    return np.array([var_m, mean_m, max_m, mad_m, spectral_centroid, spectral_energy])


def create_non_overlapping_windows(df, W=200, class_col='class', vib_cols=['X', 'Y', 'Z']):
    features, labels = [], []
    n = len(df)
    for start in range(0, n, W):
        end = start + W
        if end > n:
            break
        window = df.iloc[start:end]
        if len(window) != W:
            continue
        x, y, z = window[vib_cols[0]].values, window[vib_cols[1]].values, window[vib_cols[2]].values
        window_classes = window[class_col].values
        counts = np.bincount(window_classes.astype(int), minlength=6)
        class_mode = np.argmax(counts)
        if counts[class_mode] < W * 0.8:
            continue
        f = extract_features(x, y, z, W=W)
        features.append(f)
        labels.append(class_mode)
    return np.array(features), np.array(labels)


# ===============================
# Run Baselines
# ===============================
def run_baselines(file_path='balanced_dataset.csv', W=200, test_size=0.2, val_size=0.2):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)

    features, labels = create_non_overlapping_windows(df, W=W)
    class_names = ['Normal'] + [f'Fault {i}' for i in range(1, 6)]

    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size,
                                                      stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size),
                                                      stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    baselines = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM_RBF": SVC(kernel="rbf", probability=False),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for name, model in baselines.items():
        # Train time
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Inference time
        start = time.time()
        y_pred = model.predict(X_test)
        infer_time = (time.time() - start) / len(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted")
        f1_m = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Model size via pickle
        model_bytes = pickle.dumps(model)
        model_size_MB = len(model_bytes) / (1024 * 1024)

        # Store results
        results[name] = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "classification_report": report,
            "train_time_ms": round(train_time * 1000, 3),
            "avg_infer_time_ms": round(infer_time * 1000, 6),
            "model_size_MB": round(model_size_MB, 5)
        }

        print(f"{name} → Acc: {acc:.4f}, Weighted F1: {f1_w:.4f}, Macro F1: {f1_m:.4f}, "
              f"TrainTime: {train_time*1000:.2f} ms, Inference: {infer_time*1000:.4f} ms, "
              f"Model Size: {model_size_MB:.4f} MB")

    # Save results
    os.makedirs("baseline_results", exist_ok=True)
    baseline_json = "baseline_results/baseline_results.json"
    with open(baseline_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Baseline results saved to {baseline_json}")
    return results


# ===============================
# Run Entry
# ===============================
if __name__ == "__main__":
    try:
        run_baselines("balanced_dataset.csv", W=200)
    except Exception as e:
        print("Error:", e)
