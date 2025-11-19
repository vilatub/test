#!/usr/bin/env python3
"""
Script to create temporal anomalies notebook for Phase 5.3
Includes: Change Point Detection (CUSUM, PELT), LSTM-Autoencoder
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Title and Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Phase 5.3: Temporal Anomaly Detection

## Advanced Time Series Anomaly Methods

This notebook covers sophisticated temporal anomaly detection techniques:

1. **Change Point Detection**
   - CUSUM (Cumulative Sum Control Chart)
   - PELT (Pruned Exact Linear Time) Algorithm

2. **Deep Learning for Temporal Anomalies**
   - LSTM-Autoencoder for sequence reconstruction
   - Anomaly scoring based on reconstruction error

### Dataset
We'll create synthetic sensor data with various anomaly types:
- Point anomalies (sudden spikes)
- Contextual anomalies (unusual patterns)
- Collective anomalies (regime changes)"""))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# For Change Point Detection
from scipy.signal import find_peaks

# For LSTM-Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

print("Libraries loaded successfully")
print(f"TensorFlow version: {tf.__version__}")"""))

    # Cell 3: Create temporal dataset
    cells.append(nbf.v4.new_code_cell("""def create_sensor_data(n_samples=5000, anomaly_rate=0.05):
    \"\"\"
    Create synthetic sensor time series with various anomaly types.

    Parameters:
    -----------
    n_samples : int
        Number of time points
    anomaly_rate : float
        Proportion of anomalies

    Returns:
    --------
    data : DataFrame with sensor readings and labels
    \"\"\"
    t = np.arange(n_samples)

    # Base signal: combination of trends and seasonality
    trend = 0.001 * t
    daily_pattern = 5 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    weekly_pattern = 3 * np.sin(2 * np.pi * t / 168)  # 168-hour (weekly) cycle
    noise = np.random.normal(0, 0.5, n_samples)

    signal = 50 + trend + daily_pattern + weekly_pattern + noise

    # Initialize labels
    labels = np.zeros(n_samples, dtype=int)

    # Add different types of anomalies
    n_anomalies = int(n_samples * anomaly_rate)

    # Type 1: Point anomalies (sudden spikes) - 40% of anomalies
    n_point = int(n_anomalies * 0.4)
    point_indices = np.random.choice(range(100, n_samples-100), n_point, replace=False)
    for idx in point_indices:
        signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(10, 20)
        labels[idx] = 1

    # Type 2: Level shifts (regime changes) - 30% of anomalies
    n_shifts = 3
    shift_starts = np.random.choice(range(500, n_samples-500), n_shifts, replace=False)
    for start in shift_starts:
        duration = np.random.randint(50, 150)
        shift_amount = np.random.choice([-1, 1]) * np.random.uniform(5, 10)
        signal[start:start+duration] += shift_amount
        labels[start:start+duration] = 1

    # Type 3: Gradual drifts - 30% of anomalies
    n_drifts = 2
    drift_starts = np.random.choice(range(200, n_samples-300), n_drifts, replace=False)
    for start in drift_starts:
        duration = np.random.randint(100, 200)
        drift = np.linspace(0, np.random.uniform(8, 15), duration)
        signal[start:start+duration] += drift
        labels[start:start+duration] = 1

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'sensor_value': signal,
        'is_anomaly': labels
    })

    return data

# Generate data
sensor_data = create_sensor_data(n_samples=5000, anomaly_rate=0.05)

print(f"Dataset shape: {sensor_data.shape}")
print(f"Total anomalies: {sensor_data['is_anomaly'].sum()} ({sensor_data['is_anomaly'].mean()*100:.1f}%)")
print(f"\\nSample data:")
print(sensor_data.head(10))"""))

    # Cell 4: Visualize data
    cells.append(nbf.v4.new_code_cell("""# Visualize the time series with anomalies
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full time series
ax1 = axes[0]
ax1.plot(sensor_data['timestamp'], sensor_data['sensor_value'],
         linewidth=0.5, alpha=0.7, label='Sensor Value')

# Highlight anomalies
anomaly_mask = sensor_data['is_anomaly'] == 1
ax1.scatter(sensor_data.loc[anomaly_mask, 'timestamp'],
            sensor_data.loc[anomaly_mask, 'sensor_value'],
            c='red', s=10, alpha=0.5, label='Anomalies')

ax1.set_xlabel('Time')
ax1.set_ylabel('Sensor Value')
ax1.set_title('Complete Time Series with Ground Truth Anomalies')
ax1.legend()

# Zoomed view (first 500 points)
ax2 = axes[1]
zoom_data = sensor_data.iloc[:500]
ax2.plot(zoom_data['timestamp'], zoom_data['sensor_value'],
         linewidth=1, alpha=0.8, label='Sensor Value')

zoom_anomalies = zoom_data[zoom_data['is_anomaly'] == 1]
ax2.scatter(zoom_anomalies['timestamp'], zoom_anomalies['sensor_value'],
            c='red', s=30, alpha=0.7, label='Anomalies')

ax2.set_xlabel('Time')
ax2.set_ylabel('Sensor Value')
ax2.set_title('Zoomed View: First 500 Time Points')
ax2.legend()

plt.tight_layout()
plt.show()"""))

    # Cell 5: CUSUM implementation
    cells.append(nbf.v4.new_markdown_cell("""## 1. Change Point Detection

### 1.1 CUSUM (Cumulative Sum Control Chart)

CUSUM detects changes in the mean of a time series by accumulating deviations from a target value. It's particularly effective for detecting small, persistent shifts.

**Algorithm:**
- $S_t^+ = \\max(0, S_{t-1}^+ + (x_t - \\mu_0 - k))$ (upward CUSUM)
- $S_t^- = \\max(0, S_{t-1}^- + (-x_t + \\mu_0 - k))$ (downward CUSUM)

Where $k$ is the allowable slack and $h$ is the decision threshold."""))

    # Cell 6: CUSUM code
    cells.append(nbf.v4.new_code_cell("""def cusum_detector(data, threshold=5, drift=0.5):
    \"\"\"
    CUSUM (Cumulative Sum) change point detection.

    Parameters:
    -----------
    data : array-like
        Time series data
    threshold : float
        Decision threshold (h)
    drift : float
        Allowable slack (k)

    Returns:
    --------
    change_points : list of indices where changes detected
    cusum_pos : positive CUSUM values
    cusum_neg : negative CUSUM values
    \"\"\"
    n = len(data)
    mean = np.mean(data[:100])  # Use first 100 points as baseline
    std = np.std(data[:100])

    # Normalize by std
    normalized = (data - mean) / std

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    change_points = []

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - drift)
        cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - drift)

        # Check for change point
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            change_points.append(i)
            # Reset after detection
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    return change_points, cusum_pos, cusum_neg

# Apply CUSUM
values = sensor_data['sensor_value'].values
change_points, cusum_pos, cusum_neg = cusum_detector(values, threshold=5, drift=0.5)

print(f"CUSUM detected {len(change_points)} change points")
print(f"First 10 change points at indices: {change_points[:10]}")"""))

    # Cell 7: Visualize CUSUM
    cells.append(nbf.v4.new_code_cell("""# Visualize CUSUM results
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original signal with change points
ax1 = axes[0]
ax1.plot(values, linewidth=0.5, alpha=0.7)
for cp in change_points:
    ax1.axvline(x=cp, color='red', alpha=0.3, linewidth=1)
ax1.set_ylabel('Sensor Value')
ax1.set_title('Time Series with CUSUM Change Points')

# CUSUM statistics
ax2 = axes[1]
ax2.plot(cusum_pos, label='CUSUM+', alpha=0.7)
ax2.plot(cusum_neg, label='CUSUM-', alpha=0.7)
ax2.axhline(y=5, color='red', linestyle='--', label='Threshold')
ax2.set_ylabel('CUSUM Value')
ax2.set_title('CUSUM Statistics')
ax2.legend()

# Zoomed view
ax3 = axes[2]
zoom_range = slice(0, 1000)
ax3.plot(values[zoom_range], linewidth=1, alpha=0.8)
zoom_cps = [cp for cp in change_points if cp < 1000]
for cp in zoom_cps:
    ax3.axvline(x=cp, color='red', alpha=0.5, linewidth=2)
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Sensor Value')
ax3.set_title('Zoomed View: First 1000 Points')

plt.tight_layout()
plt.show()"""))

    # Cell 8: PELT introduction
    cells.append(nbf.v4.new_markdown_cell("""### 1.2 PELT (Pruned Exact Linear Time) Algorithm

PELT is an efficient algorithm for exact segmentation of time series. It finds the optimal number and location of change points by minimizing a cost function with a penalty for the number of changes.

We'll implement a simplified version using dynamic programming with pruning."""))

    # Cell 9: PELT implementation
    cells.append(nbf.v4.new_code_cell("""def pelt_detector(data, penalty=10, min_segment=10):
    \"\"\"
    Simplified PELT-like change point detection using dynamic programming.

    Parameters:
    -----------
    data : array-like
        Time series data
    penalty : float
        Penalty for adding a change point (BIC-like)
    min_segment : int
        Minimum segment length

    Returns:
    --------
    change_points : list of change point indices
    \"\"\"
    n = len(data)

    # Cost function: negative log-likelihood for Gaussian
    def segment_cost(start, end):
        if end - start < 2:
            return np.inf
        segment = data[start:end]
        var = np.var(segment)
        if var == 0:
            var = 1e-10
        return (end - start) * np.log(var)

    # Dynamic programming
    # F[t] = minimum cost of segmenting data[0:t]
    F = np.full(n + 1, np.inf)
    F[0] = -penalty  # Will add penalty for first segment

    # Store the last change point for each position
    last_cp = np.zeros(n + 1, dtype=int)

    for t in range(min_segment, n + 1):
        # Find best previous change point
        candidates = range(max(0, t - 500), t - min_segment + 1)  # Limit search for efficiency

        for s in candidates:
            cost = F[s] + segment_cost(s, t) + penalty
            if cost < F[t]:
                F[t] = cost
                last_cp[t] = s

    # Backtrack to find change points
    change_points = []
    t = n
    while t > 0:
        if last_cp[t] > 0:
            change_points.append(last_cp[t])
        t = last_cp[t]

    change_points = sorted(change_points)
    return change_points

# Apply PELT
pelt_change_points = pelt_detector(values, penalty=15, min_segment=20)

print(f"PELT detected {len(pelt_change_points)} change points")
print(f"Change points at indices: {pelt_change_points[:15]}...")"""))

    # Cell 10: Visualize PELT
    cells.append(nbf.v4.new_code_cell("""# Visualize PELT results
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full series with PELT segments
ax1 = axes[0]
ax1.plot(values, linewidth=0.5, alpha=0.7)

# Color each segment differently
all_cps = [0] + pelt_change_points + [len(values)]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_cps)-1))

for i in range(len(all_cps)-1):
    start, end = all_cps[i], all_cps[i+1]
    segment_mean = np.mean(values[start:end])
    ax1.hlines(y=segment_mean, xmin=start, xmax=end,
               colors=colors[i], linewidths=2, alpha=0.7)
    ax1.axvline(x=start, color='red', alpha=0.3, linewidth=1)

ax1.set_ylabel('Sensor Value')
ax1.set_title('PELT Segmentation with Segment Means')

# Compare with ground truth
ax2 = axes[1]
ax2.plot(values, linewidth=0.5, alpha=0.5, label='Signal')

# PELT change points
for cp in pelt_change_points:
    ax2.axvline(x=cp, color='blue', alpha=0.5, linewidth=1, label='PELT' if cp == pelt_change_points[0] else '')

# Ground truth anomaly regions
anomaly_indices = np.where(sensor_data['is_anomaly'].values == 1)[0]
ax2.scatter(anomaly_indices, values[anomaly_indices], c='red', s=5, alpha=0.3, label='True Anomalies')

ax2.set_xlabel('Time Index')
ax2.set_ylabel('Sensor Value')
ax2.set_title('PELT Change Points vs Ground Truth Anomalies')
ax2.legend()

plt.tight_layout()
plt.show()"""))

    # Cell 11: LSTM-Autoencoder introduction
    cells.append(nbf.v4.new_markdown_cell("""## 2. LSTM-Autoencoder for Temporal Anomaly Detection

LSTM-Autoencoders learn to reconstruct normal time series patterns. Anomalies are detected when the reconstruction error exceeds a threshold.

### Architecture:
1. **Encoder**: LSTM layers compress the input sequence
2. **Bottleneck**: Compressed representation
3. **Decoder**: LSTM layers reconstruct the sequence

### Anomaly Score:
- Compute reconstruction error for each sequence
- Higher error indicates more anomalous behavior"""))

    # Cell 12: Prepare sequences
    cells.append(nbf.v4.new_code_cell("""def create_sequences(data, seq_length=50):
    \"\"\"
    Create sequences for LSTM-Autoencoder.

    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Length of each sequence

    Returns:
    --------
    sequences : array of shape (n_sequences, seq_length, 1)
    \"\"\"
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences).reshape(-1, seq_length, 1)

# Normalize data
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

# Create sequences
SEQ_LENGTH = 50
sequences = create_sequences(values_scaled, SEQ_LENGTH)

print(f"Number of sequences: {sequences.shape[0]}")
print(f"Sequence shape: {sequences.shape}")

# Split into train/test (use first 70% for training - assumed normal)
train_size = int(len(sequences) * 0.7)
X_train = sequences[:train_size]
X_test = sequences[train_size:]

# Get corresponding labels for test set
test_labels = sensor_data['is_anomaly'].values[train_size + SEQ_LENGTH - 1:]

print(f"\\nTraining sequences: {X_train.shape[0]}")
print(f"Test sequences: {X_test.shape[0]}")
print(f"Test labels: {len(test_labels)}")"""))

    # Cell 13: Build LSTM-Autoencoder
    cells.append(nbf.v4.new_code_cell("""def build_lstm_autoencoder(seq_length, n_features=1):
    \"\"\"
    Build LSTM-Autoencoder model.

    Architecture:
    - Encoder: LSTM(64) -> LSTM(32)
    - Decoder: LSTM(32) -> LSTM(64) -> TimeDistributed(Dense)
    \"\"\"
    # Encoder
    inputs = Input(shape=(seq_length, n_features))

    # Encoder
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)

    # Bottleneck
    bottleneck = RepeatVector(seq_length)(encoded)

    # Decoder
    decoded = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)

    # Output
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

# Build model
model = build_lstm_autoencoder(SEQ_LENGTH)
model.summary()"""))

    # Cell 14: Train model
    cells.append(nbf.v4.new_code_cell("""# Train the model
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('LSTM-Autoencoder Training History')
ax.legend()
plt.show()

print(f"\\nFinal training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")"""))

    # Cell 15: Calculate reconstruction error
    cells.append(nbf.v4.new_code_cell("""# Calculate reconstruction error
train_pred = model.predict(X_train, verbose=0)
test_pred = model.predict(X_test, verbose=0)

# Mean squared error for each sequence
train_mse = np.mean(np.square(X_train - train_pred), axis=(1, 2))
test_mse = np.mean(np.square(X_test - test_pred), axis=(1, 2))

# Set threshold based on training data (mean + 2*std)
threshold = np.mean(train_mse) + 2 * np.std(train_mse)

print(f"Training MSE - Mean: {np.mean(train_mse):.6f}, Std: {np.std(train_mse):.6f}")
print(f"Anomaly threshold: {threshold:.6f}")
print(f"Test MSE - Mean: {np.mean(test_mse):.6f}, Std: {np.std(test_mse):.6f}")

# Detect anomalies
predictions = (test_mse > threshold).astype(int)
print(f"\\nPredicted anomalies in test set: {predictions.sum()} ({predictions.mean()*100:.1f}%)")"""))

    # Cell 16: Visualize LSTM results
    cells.append(nbf.v4.new_code_cell("""# Visualize reconstruction error and anomalies
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Reconstruction error distribution
ax1 = axes[0]
ax1.hist(train_mse, bins=50, alpha=0.5, label='Train', density=True)
ax1.hist(test_mse, bins=50, alpha=0.5, label='Test', density=True)
ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
ax1.set_xlabel('Reconstruction Error (MSE)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Reconstruction Errors')
ax1.legend()

# Time series with anomaly scores
ax2 = axes[1]
test_indices = np.arange(train_size + SEQ_LENGTH - 1, len(values))
ax2.plot(test_indices, values[train_size + SEQ_LENGTH - 1:], linewidth=0.5, alpha=0.5)

# Normalize MSE for visualization
mse_normalized = (test_mse - test_mse.min()) / (test_mse.max() - test_mse.min())
colors = plt.cm.Reds(mse_normalized)

for i, (idx, mse_val) in enumerate(zip(test_indices[:len(test_mse)], test_mse)):
    if mse_val > threshold:
        ax2.axvline(x=idx, color='red', alpha=0.3, linewidth=1)

ax2.set_ylabel('Sensor Value')
ax2.set_title('Test Data with LSTM-AE Detected Anomalies (red lines)')

# Comparison with ground truth
ax3 = axes[2]
ax3.plot(test_mse, label='Reconstruction Error', alpha=0.7)
ax3.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

# Mark true anomalies
if len(test_labels) == len(test_mse):
    true_anomaly_idx = np.where(test_labels == 1)[0]
    ax3.scatter(true_anomaly_idx, test_mse[true_anomaly_idx],
                c='green', s=20, alpha=0.5, label='True Anomalies')

ax3.set_xlabel('Sequence Index')
ax3.set_ylabel('MSE')
ax3.set_title('Reconstruction Error with True Anomaly Locations')
ax3.legend()

plt.tight_layout()
plt.show()"""))

    # Cell 17: Evaluate LSTM-AE
    cells.append(nbf.v4.new_code_cell("""# Evaluate LSTM-Autoencoder performance
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Align predictions with labels
min_len = min(len(predictions), len(test_labels))
pred_aligned = predictions[:min_len]
labels_aligned = test_labels[:min_len]

# Calculate metrics
precision = precision_score(labels_aligned, pred_aligned, zero_division=0)
recall = recall_score(labels_aligned, pred_aligned, zero_division=0)
f1 = f1_score(labels_aligned, pred_aligned, zero_division=0)

print("LSTM-Autoencoder Performance:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(labels_aligned, pred_aligned)
print(f"\\nConfusion Matrix:")
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels_aligned, test_mse[:min_len])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('LSTM-Autoencoder ROC Curve')
ax.legend(loc='lower right')
plt.show()"""))

    # Cell 18: Method comparison
    cells.append(nbf.v4.new_code_cell("""# Compare all methods
print("=" * 60)
print("TEMPORAL ANOMALY DETECTION - METHOD COMPARISON")
print("=" * 60)

print("\\n1. CUSUM (Cumulative Sum Control Chart)")
print(f"   - Change points detected: {len(change_points)}")
print("   - Best for: Small, persistent shifts in mean")
print("   - Advantages: Simple, interpretable, real-time capable")

print("\\n2. PELT (Pruned Exact Linear Time)")
print(f"   - Segments detected: {len(pelt_change_points) + 1}")
print("   - Best for: Optimal segmentation with unknown number of changes")
print("   - Advantages: Exact solution, handles multiple change types")

print("\\n3. LSTM-Autoencoder")
print(f"   - AUC-ROC: {roc_auc:.3f}")
print(f"   - Precision: {precision:.3f}, Recall: {recall:.3f}")
print("   - Best for: Complex temporal patterns, sequence anomalies")
print("   - Advantages: Learns complex dependencies, no assumptions on data")

print("\\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("- Use CUSUM for real-time monitoring with quick response")
print("- Use PELT for offline analysis and optimal segmentation")
print("- Use LSTM-AE for complex patterns when labels available for tuning")"""))

    # Cell 19: Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## Summary

### Key Takeaways

1. **Change Point Detection Methods**
   - CUSUM: Effective for detecting shifts in mean, runs in linear time
   - PELT: Optimal segmentation with penalty-based change point detection

2. **Deep Learning Approaches**
   - LSTM-Autoencoder learns temporal patterns from normal data
   - Anomalies detected via reconstruction error threshold
   - Requires careful threshold tuning for best results

3. **Dataset Characteristics**
   - 5,000 time points with synthetic sensor data
   - Multiple anomaly types: point, shift, drift
   - 5% anomaly rate

### Next Steps
- Combine methods for ensemble anomaly detection
- Experiment with different LSTM architectures (attention, bidirectional)
- Apply to real-world sensor data from manufacturing or IoT"""))

    nb['cells'] = cells

    # Save notebook
    output_path = '/home/user/test/notebooks/phase5_anomaly_patterns/04_temporal_anomalies.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Temporal Anomalies notebook created: {output_path}")
    print(f"Total cells: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
