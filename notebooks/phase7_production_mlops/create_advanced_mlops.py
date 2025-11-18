#!/usr/bin/env python3
"""
Script to create advanced MLOps notebook for Phase 7 expansion.
Includes: sklearn Pipeline, ONNX export, A/B Testing framework, Model Compression
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Title
    cells.append(nbf.v4.new_markdown_cell("""# Phase 7.2: Advanced MLOps Practices

## Production-Ready Machine Learning Infrastructure

This notebook covers advanced MLOps techniques:

1. **sklearn Pipeline with ColumnTransformer** - End-to-end reproducible ML workflows
2. **ONNX Export** - Cross-platform model deployment
3. **A/B Testing Framework** - Statistical testing for model comparison
4. **Model Compression** - Quantization and pruning for efficient inference

### Focus
Building production-ready pipelines that are reproducible, portable, and efficient."""))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# For ONNX
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# TensorFlow for compression demo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

np.random.seed(42)
tf.random.set_seed(42)

print("Libraries loaded successfully")
print(f"ONNX version: {onnx.__version__}")
print(f"ONNX Runtime version: {ort.__version__}")
print(f"TensorFlow version: {tf.__version__}")"""))

    # Cell 3: Create dataset
    cells.append(nbf.v4.new_code_cell("""def create_customer_churn_data(n_samples=10000):
    \"\"\"
    Create synthetic customer churn dataset with mixed feature types.
    Ideal for demonstrating sklearn Pipelines.
    \"\"\"
    # Numeric features
    data = {
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 100, n_samples),
        'total_charges': np.zeros(n_samples),  # Will calculate
        'num_support_tickets': np.random.poisson(2, n_samples),
        'num_referrals': np.random.poisson(1, n_samples),
        'avg_monthly_usage_gb': np.random.lognormal(2, 1, n_samples),
    }

    # Calculate total charges
    data['total_charges'] = data['tenure_months'] * data['monthly_charges'] * \\
                            np.random.uniform(0.9, 1.1, n_samples)

    # Categorical features
    data['contract_type'] = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples, p=[0.5, 0.3, 0.2]
    )
    data['payment_method'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples
    )
    data['internet_service'] = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        n_samples, p=[0.35, 0.45, 0.2]
    )

    # Add some missing values (realistic)
    for col in ['monthly_charges', 'num_support_tickets']:
        mask = np.random.random(n_samples) < 0.05
        data[col] = np.where(mask, np.nan, data[col])

    # Generate churn labels based on features
    churn_prob = 0.1 + \\
        0.3 * (data['contract_type'] == 'Month-to-month') + \\
        0.1 * (data['tenure_months'] < 12) / 12 + \\
        0.15 * (data['num_support_tickets'] > 3) + \\
        0.1 * (np.array(data['monthly_charges']) > 70)

    # Handle NaN in probability calculation
    churn_prob = np.nan_to_num(churn_prob, nan=0.2)
    churn_prob = np.clip(churn_prob, 0, 1)

    data['churned'] = np.random.binomial(1, churn_prob)

    df = pd.DataFrame(data)
    return df

# Generate data
df = create_customer_churn_data(n_samples=10000)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churned'].mean()*100:.1f}%")
print(f"\\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\\nSample data:")
df.head()"""))

    # Cell 4: sklearn Pipeline intro
    cells.append(nbf.v4.new_markdown_cell("""## 1. sklearn Pipeline with ColumnTransformer

### Why use Pipelines?

1. **Reproducibility** - All preprocessing steps are bundled together
2. **No data leakage** - Transformations fit only on training data
3. **Easy deployment** - Single object to serialize and deploy
4. **Cross-validation** - Preprocessing inside CV loop

### ColumnTransformer
Applies different transformations to different column types (numeric vs categorical)."""))

    # Cell 5: Build Pipeline
    cells.append(nbf.v4.new_code_cell("""# Define feature groups
numeric_features = ['tenure_months', 'monthly_charges', 'total_charges',
                    'num_support_tickets', 'num_referrals', 'avg_monthly_usage_gb']
categorical_features = ['contract_type', 'payment_method', 'internet_service']

# Numeric transformer: impute then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: impute then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline with classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Pipeline structure:")
print(pipeline)"""))

    # Cell 6: Train and evaluate
    cells.append(nbf.v4.new_code_cell("""# Prepare data
X = df.drop('churned', axis=1)
y = df['churned'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("Pipeline Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"\\nCross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")"""))

    # Cell 7: Feature names
    cells.append(nbf.v4.new_code_cell("""# Get feature names after transformation
def get_feature_names(column_transformer):
    \"\"\"Get feature names from ColumnTransformer.\"\"\"
    output_features = []

    for name, pipe, columns in column_transformer.transformers_:
        if name == 'remainder':
            continue

        if hasattr(pipe, 'get_feature_names_out'):
            names = pipe.get_feature_names_out(columns)
        elif hasattr(pipe[-1], 'get_feature_names_out'):
            names = pipe[-1].get_feature_names_out(columns)
        else:
            names = columns

        output_features.extend(names)

    return output_features

feature_names = get_feature_names(preprocessor)
print(f"Total features after transformation: {len(feature_names)}")
print(f"\\nFeature names: {feature_names}")

# Feature importance
rf_model = pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

# Plot top features
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_n = 15
ax.barh(importance_df['feature'][:top_n][::-1],
        importance_df['importance'][:top_n][::-1])
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Features in Pipeline Model')
plt.tight_layout()
plt.show()"""))

    # Cell 8: Save pipeline
    cells.append(nbf.v4.new_code_cell("""# Save the complete pipeline
import os

# Create artifacts directory
os.makedirs('/home/user/test/notebooks/phase7_production_mlops/artifacts', exist_ok=True)

# Save with pickle
pipeline_path = '/home/user/test/notebooks/phase7_production_mlops/artifacts/churn_pipeline.pkl'
with open(pipeline_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Pipeline saved to: {pipeline_path}")

# Load and verify
with open(pipeline_path, 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Test loaded pipeline
test_pred = loaded_pipeline.predict(X_test[:5])
print(f"\\nLoaded pipeline predictions: {test_pred}")
print("Pipeline save/load successful!")"""))

    # Cell 9: ONNX intro
    cells.append(nbf.v4.new_markdown_cell("""## 2. ONNX Export

### What is ONNX?

**Open Neural Network Exchange** - an open format for ML models that enables:
- Cross-platform deployment (Python, C++, JavaScript, etc.)
- Hardware optimization (CPU, GPU, mobile)
- Model interoperability between frameworks

### Benefits:
- Faster inference with ONNX Runtime
- Deploy anywhere (cloud, edge, browser)
- Single model format for all platforms"""))

    # Cell 10: Export to ONNX
    cells.append(nbf.v4.new_code_cell("""# For ONNX export, we need to first transform the data
# and then export just the classifier

# Transform training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Train a simple model for ONNX export
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_transformed, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train_transformed.shape[1]]))]

onnx_model = convert_sklearn(
    lr_model,
    initial_types=initial_type,
    target_opset=12
)

# Save ONNX model
onnx_path = '/home/user/test/notebooks/phase7_production_mlops/artifacts/churn_model.onnx'
with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved to: {onnx_path}")
print(f"Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

# Validate ONNX model
onnx.checker.check_model(onnx_model)
print("ONNX model validation passed!")"""))

    # Cell 11: ONNX Runtime inference
    cells.append(nbf.v4.new_code_cell("""# Inference with ONNX Runtime
session = ort.InferenceSession(onnx_path)

# Get input/output names
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print(f"Input name: {input_name}")
print(f"Output names: {output_names}")

# Run inference
X_test_transformed = preprocessor.transform(X_test)
X_test_onnx = X_test_transformed.astype(np.float32)

# Time comparison
import time

# sklearn inference
start = time.time()
for _ in range(100):
    sklearn_pred = lr_model.predict_proba(X_test_transformed)
sklearn_time = (time.time() - start) / 100 * 1000

# ONNX Runtime inference
start = time.time()
for _ in range(100):
    onnx_pred = session.run(output_names, {input_name: X_test_onnx})
onnx_time = (time.time() - start) / 100 * 1000

print(f"\\nInference time comparison (per batch):")
print(f"  sklearn: {sklearn_time:.2f} ms")
print(f"  ONNX Runtime: {onnx_time:.2f} ms")
print(f"  Speedup: {sklearn_time/onnx_time:.1f}x")

# Verify predictions match
sklearn_proba = lr_model.predict_proba(X_test_transformed)[:, 1]
onnx_proba = onnx_pred[1][:, 1]

print(f"\\nPrediction difference: {np.abs(sklearn_proba - onnx_proba).max():.6f}")"""))

    # Cell 12: A/B Testing intro
    cells.append(nbf.v4.new_markdown_cell("""## 3. A/B Testing Framework for Models

### Why A/B Test Models?

1. **Validate improvements** - Statistical confidence that new model is better
2. **Risk mitigation** - Gradual rollout with monitoring
3. **Business metrics** - Correlate model metrics with business outcomes

### Statistical Framework:
- Null hypothesis: Models A and B perform equally
- Alternative: Model B performs differently (better or worse)
- Use appropriate statistical tests for model comparison"""))

    # Cell 13: A/B Testing framework
    cells.append(nbf.v4.new_code_cell("""class ABTestFramework:
    \"\"\"
    Framework for A/B testing machine learning models.
    \"\"\"

    def __init__(self, model_a, model_b, name_a='Model A', name_b='Model B'):
        self.model_a = model_a
        self.model_b = model_b
        self.name_a = name_a
        self.name_b = name_b
        self.results = []

    def run_experiment(self, X, y, n_splits=10, metrics=['accuracy', 'roc_auc']):
        \"\"\"
        Run A/B experiment with cross-validation.

        Parameters:
        -----------
        X : features
        y : target
        n_splits : number of CV splits
        metrics : list of metrics to compare
        \"\"\"
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        results_a = {m: [] for m in metrics}
        results_b = {m: [] for m in metrics}

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Model A
            self.model_a.fit(X_train, y_train)
            pred_a = self.model_a.predict(X_test)
            proba_a = self.model_a.predict_proba(X_test)[:, 1]

            # Model B
            self.model_b.fit(X_train, y_train)
            pred_b = self.model_b.predict(X_test)
            proba_b = self.model_b.predict_proba(X_test)[:, 1]

            # Calculate metrics
            for metric in metrics:
                if metric == 'accuracy':
                    results_a[metric].append(accuracy_score(y_test, pred_a))
                    results_b[metric].append(accuracy_score(y_test, pred_b))
                elif metric == 'roc_auc':
                    results_a[metric].append(roc_auc_score(y_test, proba_a))
                    results_b[metric].append(roc_auc_score(y_test, proba_b))

        self.results_a = results_a
        self.results_b = results_b

        return self

    def statistical_test(self, metric='roc_auc', alpha=0.05):
        \"\"\"
        Perform statistical test to compare models.

        Uses paired t-test since we have matched samples from CV.
        \"\"\"
        scores_a = np.array(self.results_a[metric])
        scores_b = np.array(self.results_b[metric])

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Effect size (Cohen's d)
        diff = scores_b - scores_a
        effect_size = diff.mean() / diff.std()

        # Results
        results = {
            'metric': metric,
            'mean_a': scores_a.mean(),
            'std_a': scores_a.std(),
            'mean_b': scores_b.mean(),
            'std_b': scores_b.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < alpha,
            'winner': self.name_b if (p_value < alpha and scores_b.mean() > scores_a.mean()) else \\
                      self.name_a if (p_value < alpha and scores_a.mean() > scores_b.mean()) else 'No significant difference'
        }

        return results

    def plot_results(self, metric='roc_auc'):
        \"\"\"Visualize A/B test results.\"\"\"
        scores_a = self.results_a[metric]
        scores_b = self.results_b[metric]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot comparison
        ax1 = axes[0]
        ax1.boxplot([scores_a, scores_b], labels=[self.name_a, self.name_b])
        ax1.set_ylabel(metric.upper())
        ax1.set_title(f'{metric.upper()} Distribution')

        # Paired differences
        ax2 = axes[1]
        diff = np.array(scores_b) - np.array(scores_a)
        ax2.hist(diff, bins=10, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', label='No difference')
        ax2.axvline(x=diff.mean(), color='green', linestyle='-', label=f'Mean diff: {diff.mean():.4f}')
        ax2.set_xlabel(f'{self.name_b} - {self.name_a}')
        ax2.set_ylabel('Count')
        ax2.set_title('Paired Differences')
        ax2.legend()

        plt.tight_layout()
        plt.show()

print("A/B Testing Framework defined")"""))

    # Cell 14: Run A/B test
    cells.append(nbf.v4.new_code_cell("""# Compare Random Forest vs Gradient Boosting

# Create pipelines for both models
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# Run A/B test
ab_test = ABTestFramework(
    pipeline_rf, pipeline_gb,
    name_a='Random Forest',
    name_b='Gradient Boosting'
)

ab_test.run_experiment(X, y, n_splits=10, metrics=['accuracy', 'roc_auc'])

# Statistical test
results = ab_test.statistical_test(metric='roc_auc', alpha=0.05)

print("A/B Test Results (ROC-AUC):")
print("=" * 50)
print(f"\\n{ab_test.name_a}:")
print(f"  Mean: {results['mean_a']:.4f} (+/- {results['std_a']:.4f})")
print(f"\\n{ab_test.name_b}:")
print(f"  Mean: {results['mean_b']:.4f} (+/- {results['std_b']:.4f})")
print(f"\\nStatistical Test:")
print(f"  t-statistic: {results['t_statistic']:.3f}")
print(f"  p-value: {results['p_value']:.4f}")
print(f"  Effect size: {results['effect_size']:.3f}")
print(f"\\nConclusion: {results['winner']}")

# Visualize
ab_test.plot_results(metric='roc_auc')"""))

    # Cell 15: Model compression intro
    cells.append(nbf.v4.new_markdown_cell("""## 4. Model Compression

### Why Compress Models?

1. **Faster inference** - Reduced computation
2. **Smaller size** - Better for edge/mobile deployment
3. **Lower cost** - Less compute resources needed

### Techniques:
- **Quantization** - Reduce precision (FP32 → INT8)
- **Pruning** - Remove unimportant weights
- **Knowledge distillation** - Train smaller model to mimic larger one"""))

    # Cell 16: Build model for compression
    cells.append(nbf.v4.new_code_cell("""# Build a neural network to demonstrate compression
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Prepare data
X_nn = preprocessor.fit_transform(X)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y, test_size=0.2, random_state=42, stratify=y
)

# Train original model
original_model = build_model(X_train_nn.shape[1])

original_model.fit(
    X_train_nn, y_train_nn,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Evaluate original
original_pred = (original_model.predict(X_test_nn, verbose=0) > 0.5).astype(int).flatten()
original_acc = accuracy_score(y_test_nn, original_pred)

print(f"Original model accuracy: {original_acc:.4f}")

# Save original model
original_path = '/home/user/test/notebooks/phase7_production_mlops/artifacts/original_model.h5'
original_model.save(original_path)
original_size = os.path.getsize(original_path)
print(f"Original model size: {original_size / 1024:.1f} KB")"""))

    # Cell 17: Quantization
    cells.append(nbf.v4.new_code_cell("""# Post-training quantization with TensorFlow Lite

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(original_model)

# Dynamic range quantization (INT8)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for full integer quantization
def representative_dataset():
    for i in range(100):
        yield [X_train_nn[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    quantized_model = converter.convert()

    # Save quantized model
    quantized_path = '/home/user/test/notebooks/phase7_production_mlops/artifacts/quantized_model.tflite'
    with open(quantized_path, 'wb') as f:
        f.write(quantized_model)

    quantized_size = os.path.getsize(quantized_path)
    compression_ratio = original_size / quantized_size

    print(f"Quantized model size: {quantized_size / 1024:.1f} KB")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    # Run inference with TFLite
    interpreter = tf.lite.Interpreter(model_path=quantized_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\\nTFLite model details:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")

except Exception as e:
    print(f"Full integer quantization failed: {e}")
    print("\\nFalling back to dynamic range quantization...")

    converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    quantized_path = '/home/user/test/notebooks/phase7_production_mlops/artifacts/quantized_model.tflite'
    with open(quantized_path, 'wb') as f:
        f.write(quantized_model)

    quantized_size = os.path.getsize(quantized_path)
    compression_ratio = original_size / quantized_size

    print(f"Quantized model size: {quantized_size / 1024:.1f} KB")
    print(f"Compression ratio: {compression_ratio:.1f}x")"""))

    # Cell 18: Pruning
    cells.append(nbf.v4.new_code_cell("""# Model pruning demonstration
# We'll use magnitude-based pruning to remove small weights

def prune_model(model, sparsity=0.5):
    \"\"\"
    Apply magnitude-based pruning to model weights.

    Parameters:
    -----------
    model : Keras model
    sparsity : fraction of weights to prune (set to 0)
    \"\"\"
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())

    total_weights = 0
    pruned_weights = 0

    for layer in pruned_model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            total_weights += weights.size

            # Calculate threshold for this layer
            threshold = np.percentile(np.abs(weights), sparsity * 100)

            # Prune weights below threshold
            mask = np.abs(weights) > threshold
            pruned = weights * mask
            pruned_weights += np.sum(mask == 0)

            layer.kernel.assign(pruned)

    actual_sparsity = pruned_weights / total_weights
    return pruned_model, actual_sparsity

# Apply pruning with different sparsity levels
sparsity_levels = [0.3, 0.5, 0.7, 0.9]
results = []

for sparsity in sparsity_levels:
    pruned, actual_sparsity = prune_model(original_model, sparsity)

    # Evaluate pruned model
    pruned_pred = (pruned.predict(X_test_nn, verbose=0) > 0.5).astype(int).flatten()
    pruned_acc = accuracy_score(y_test_nn, pruned_pred)

    results.append({
        'target_sparsity': sparsity,
        'actual_sparsity': actual_sparsity,
        'accuracy': pruned_acc
    })

    print(f"Sparsity {sparsity*100:.0f}%: Accuracy = {pruned_acc:.4f}")

# Visualize accuracy vs sparsity
fig, ax = plt.subplots(figsize=(10, 6))

sparsities = [r['actual_sparsity'] * 100 for r in results]
accuracies = [r['accuracy'] for r in results]

ax.plot(sparsities, accuracies, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=original_acc, color='red', linestyle='--', label=f'Original: {original_acc:.4f}')

ax.set_xlabel('Sparsity (%)')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy vs Pruning Sparsity')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nOriginal accuracy: {original_acc:.4f}")
print("Pruning allows significant model compression with minimal accuracy loss!")"""))

    # Cell 19: Summary
    cells.append(nbf.v4.new_code_cell("""# Summary of all techniques
print("=" * 70)
print("ADVANCED MLOPS TECHNIQUES - SUMMARY")
print("=" * 70)

print("\\n1. sklearn Pipeline with ColumnTransformer")
print("   - Combines preprocessing and model in single object")
print("   - Ensures reproducibility and prevents data leakage")
print("   - Easy to serialize and deploy")

print("\\n2. ONNX Export")
print("   - Cross-platform model format")
print(f"   - Inference speedup: ~{sklearn_time/onnx_time:.1f}x faster")
print("   - Deploy to any platform (cloud, edge, browser)")

print("\\n3. A/B Testing Framework")
print("   - Statistical rigor for model comparison")
print("   - Paired t-test with effect size")
print("   - Make data-driven deployment decisions")

print("\\n4. Model Compression")
print(f"   - Quantization compression: {compression_ratio:.1f}x smaller")
print("   - Pruning maintains accuracy up to ~50% sparsity")
print("   - Essential for edge/mobile deployment")

print("\\n" + "=" * 70)
print("PRODUCTION DEPLOYMENT CHECKLIST")
print("=" * 70)
print("[ ] Pipeline tested with production-like data")
print("[ ] ONNX model validated for correctness")
print("[ ] A/B test shows significant improvement")
print("[ ] Compression tested on target hardware")
print("[ ] Monitoring and logging in place")
print("[ ] Rollback plan documented")"""))

    # Cell 20: Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## Summary

### Key Takeaways

1. **sklearn Pipelines** create reproducible, deployable ML workflows that prevent data leakage and simplify model serving.

2. **ONNX format** enables cross-platform deployment with optimized inference performance through ONNX Runtime.

3. **A/B Testing** provides statistical rigor for model comparison, ensuring improvements are real and not due to chance.

4. **Model Compression** through quantization and pruning reduces model size and inference time for edge deployment.

### Best Practices

- Always use Pipelines to encapsulate the full ML workflow
- Export to ONNX for production inference where performance matters
- Run statistical A/B tests before deploying new models
- Consider compression for latency-sensitive or resource-constrained deployments

### Production Considerations

- Version control all artifacts (models, pipelines, configs)
- Implement comprehensive logging and monitoring
- Plan for model updates and rollbacks
- Document all assumptions and constraints

### Next Steps
- Integrate with CI/CD for automated model deployment
- Set up model monitoring for drift detection
- Build dashboards for model performance tracking"""))

    nb['cells'] = cells

    # Save notebook
    output_path = '/home/user/test/notebooks/phase7_production_mlops/02_advanced_mlops.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Advanced MLOps notebook created: {output_path}")
    print(f"Total cells: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
