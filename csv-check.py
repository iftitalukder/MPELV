import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (replace 'pipe_vibration_data.csv' with your actual file path)
file_path = 'balanced_dataset.csv'
df = pd.read_csv(file_path)

# 1. Basic Information
print("=== Basic Dataset Info ===")
print(f"Shape: {df.shape} (rows, columns)")
print(f"Columns: {list(df.columns)}")
print(f"Data Types:\n{df.dtypes}")
print("\nHead (first 5 rows):\n", df.head())
print("\nTail (last 5 rows):\n", df.tail())

# 2. Summary Statistics
print("\n=== Summary Statistics ===")
print(df.describe(include='all'))  # Includes numerical and categorical

# 3. Missing Values
print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])  # Only show columns with missing values
if missing.sum() == 0:
    print("No missing values.")

# 4. Class Distribution (assuming column named 'label' or 'class'; adjust if different)
label_col = 'label'  # Change to your actual label column name, e.g., 'class'
if label_col in df.columns:
    print("\n=== Class Distribution ===")
    class_counts = df[label_col].value_counts()
    print(class_counts)
    print(f"Unique Classes: {df[label_col].unique()}")
    print(f"Number of Classes: {len(class_counts)}")
    
    # Visualize class distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(x=label_col, data=df)
    plt.title('Class Distribution')
    plt.show()
else:
    print(f"\nNo '{label_col}' column found. Check for class/label column.")

# 5. Vibration Data Specific Checks (assuming columns like 'x', 'y', 'z'; adjust names)
vib_cols = ['x', 'y', 'z']  # Change to your actual vibration column names
if all(col in df.columns for col in vib_cols):
    print("\n=== Vibration Data Checks ===")
    
    # Check for non-numeric values
    for col in vib_cols:
        non_numeric = df[col].apply(lambda v: not isinstance(v, (int, float)))
        if non_numeric.any():
            print(f"Non-numeric values in '{col}': {df[non_numeric][col].head()}")
    
    # Basic stats per class
    if label_col in df.columns:
        print("\nMean Vibration per Class:")
        print(df.groupby(label_col)[vib_cols].mean())
    
    # Sample length check (if data is flat; for time-series, assume rows are samples)
    print(f"\nTotal Samples: {len(df)}")
    
    # If timestamp column exists (e.g., 'timestamp'), check sampling rate
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        sampling_intervals = df['timestamp'].diff().dropna()
        avg_sampling_rate = sampling_intervals.mean().total_seconds()
        print(f"\nAverage Sampling Interval: {avg_sampling_rate:.2f} seconds")
        if avg_sampling_rate > 0:
            print(f"Estimated Sampling Frequency: {1 / avg_sampling_rate:.2f} Hz")
else:
    print("\nVibration columns (x, y, z) not all found. Adjust 'vib_cols' list.")

# 6. Outliers Detection (basic, using IQR for vibration columns)
if all(col in df.columns for col in vib_cols):
    print("\n=== Outlier Detection (IQR Method) ===")
    for col in vib_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        print(f"Outliers in '{col}': {outliers.sum()} ({outliers.sum() / len(df) * 100:.2f}%)")

# 7. Correlation (for numerical columns)
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    print("\n=== Correlation Matrix ===")
    corr = df[numerical_cols].corr()
    print(corr)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# 8. Recommendations for Pipeline
print("\n=== Pipeline Recommendations ===")
print("- Ensure data is windowed (e.g., 50 samples per window) for MPELV.")
print("- If classes are imbalanced, consider oversampling or class weights.")
print("- Handle missing values (e.g., interpolate for time-series).")
print("- Normalize if needed, but MPELV features (var, ZCR) are scale-invariant.")
print("- For training: Group by windows, extract features per class.")