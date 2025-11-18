#!/usr/bin/env python3
"""
Script to create advanced XAI notebook for Phase 6 expansion.
Includes: DeepSHAP, Attention Visualization, Counterfactual Explanations, Model Distillation
"""

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Title
    cells.append(nbf.v4.new_markdown_cell("""# Phase 6.2: Advanced Explainable AI Methods

## Deep Learning Interpretability & Counterfactual Analysis

This notebook covers advanced XAI techniques:

1. **DeepSHAP** - SHAP values for deep neural networks
2. **Attention Visualization** - Interpreting attention weights
3. **Counterfactual Explanations** - "What-if" scenarios
4. **Model Distillation** - Extracting interpretable rules from black-box models

### Dataset
Credit card fraud detection with class imbalance - ideal for demonstrating
explainability in high-stakes decisions."""))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Flatten

# SHAP
import shap

np.random.seed(42)
tf.random.set_seed(42)

print("Libraries loaded successfully")
print(f"TensorFlow: {tf.__version__}")
print(f"SHAP: {shap.__version__}")"""))

    # Cell 3: Create dataset
    cells.append(nbf.v4.new_code_cell("""def create_fraud_dataset(n_samples=10000, fraud_rate=0.05):
    \"\"\"
    Create synthetic credit card fraud dataset.

    Features represent transaction characteristics that
    are meaningful for explainability demonstrations.
    \"\"\"
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # Normal transactions
    normal = {
        'amount': np.random.lognormal(4, 1, n_normal),  # $50-100 typical
        'hour': np.random.choice(range(8, 22), n_normal),  # Daytime
        'day_of_week': np.random.choice(range(7), n_normal),
        'merchant_category': np.random.choice(range(10), n_normal),
        'distance_from_home': np.abs(np.random.normal(5, 3, n_normal)),
        'time_since_last_txn': np.random.exponential(24, n_normal),  # Hours
        'avg_txn_amount': np.random.lognormal(4, 0.5, n_normal),
        'txn_frequency': np.random.poisson(10, n_normal),  # Per month
        'is_online': np.random.binomial(1, 0.3, n_normal),
        'is_international': np.random.binomial(1, 0.05, n_normal),
    }

    # Fraudulent transactions - different patterns
    fraud = {
        'amount': np.random.lognormal(6, 1.5, n_fraud),  # Higher amounts
        'hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraud),  # Night
        'day_of_week': np.random.choice(range(7), n_fraud),
        'merchant_category': np.random.choice([0, 1, 2], n_fraud),  # Specific categories
        'distance_from_home': np.abs(np.random.normal(50, 30, n_fraud)),  # Far from home
        'time_since_last_txn': np.random.exponential(1, n_fraud),  # Quick succession
        'avg_txn_amount': np.random.lognormal(4, 0.5, n_fraud),
        'txn_frequency': np.random.poisson(10, n_fraud),
        'is_online': np.random.binomial(1, 0.7, n_fraud),  # More online
        'is_international': np.random.binomial(1, 0.4, n_fraud),  # More international
    }

    # Combine
    df_normal = pd.DataFrame(normal)
    df_normal['is_fraud'] = 0

    df_fraud = pd.DataFrame(fraud)
    df_fraud['is_fraud'] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

# Generate data
df = create_fraud_dataset(n_samples=10000, fraud_rate=0.05)

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
print(f"\\nFeatures: {list(df.columns[:-1])}")
print(f"\\nSample data:")
df.head()"""))

    # Cell 4: Prepare data
    cells.append(nbf.v4.new_code_cell("""# Prepare features and target
feature_names = ['amount', 'hour', 'day_of_week', 'merchant_category',
                 'distance_from_home', 'time_since_last_txn', 'avg_txn_amount',
                 'txn_frequency', 'is_online', 'is_international']

X = df[feature_names].values
y = df['is_fraud'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training fraud rate: {y_train.mean()*100:.1f}%")"""))

    # Cell 5: DeepSHAP intro
    cells.append(nbf.v4.new_markdown_cell("""## 1. DeepSHAP - SHAP Values for Deep Neural Networks

DeepSHAP combines SHAP with DeepLIFT to efficiently compute SHAP values for deep learning models. It's faster than KernelSHAP for neural networks while maintaining theoretical guarantees.

### How it works:
- Uses backpropagation-based attribution
- Computes contributions relative to a reference (background) distribution
- Satisfies SHAP's consistency and local accuracy properties"""))

    # Cell 6: Build neural network
    cells.append(nbf.v4.new_code_cell("""# Build a neural network for fraud detection
def build_fraud_detector(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train model
model = build_fraud_detector(X_train_scaled.shape[1])

# Use class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("\\nModel Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")"""))

    # Cell 7: DeepSHAP analysis
    cells.append(nbf.v4.new_code_cell("""# Apply DeepSHAP
# Use a subset of training data as background
background = X_train_scaled[np.random.choice(len(X_train_scaled), 100, replace=False)]

# Create DeepSHAP explainer
explainer = shap.DeepExplainer(model, background)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test_scaled[:500])

# For binary classification, shap_values might be a list
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print(f"SHAP values shape: {shap_values.shape}")
print(f"Feature importance calculated for {shap_values.shape[0]} samples")"""))

    # Cell 8: Visualize DeepSHAP
    cells.append(nbf.v4.new_code_cell("""# SHAP Summary Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Beeswarm plot
plt.sca(axes[0])
shap.summary_plot(shap_values, X_test_scaled[:500],
                  feature_names=feature_names, show=False)
axes[0].set_title('DeepSHAP: Feature Impact on Fraud Prediction')

# Bar plot for mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)

axes[1].barh([feature_names[i] for i in sorted_idx], mean_abs_shap[sorted_idx])
axes[1].set_xlabel('Mean |SHAP value|')
axes[1].set_title('DeepSHAP: Feature Importance')

plt.tight_layout()
plt.show()"""))

    # Cell 9: Individual explanation
    cells.append(nbf.v4.new_code_cell("""# Explain individual fraud predictions
fraud_indices = np.where(y_test[:500] == 1)[0]

if len(fraud_indices) > 0:
    # Take first fraud case
    idx = fraud_indices[0]

    print(f"Explaining prediction for sample {idx} (True fraud)")
    print(f"Predicted probability: {y_pred_proba[idx]:.3f}")
    print(f"\\nFeature contributions:")

    # Show feature values and their SHAP contributions
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': X_test[idx],
        'SHAP': shap_values[idx]
    }).sort_values('SHAP', key=abs, ascending=False)

    print(contributions.to_string(index=False))

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_test_scaled[idx],
        feature_names=feature_names
    ), show=False)
    plt.title(f'DeepSHAP Waterfall: Fraud Case {idx}')
    plt.tight_layout()
    plt.show()"""))

    # Cell 10: Attention intro
    cells.append(nbf.v4.new_markdown_cell("""## 2. Attention Visualization

Attention mechanisms provide built-in interpretability by showing which parts of the input the model focuses on. We'll build a simple attention-based model and visualize the attention weights.

### Architecture:
- Multi-head self-attention over features
- Attention weights show feature interactions
- More interpretable than standard feed-forward networks"""))

    # Cell 11: Attention model
    cells.append(nbf.v4.new_code_cell("""class AttentionFraudDetector(Model):
    \"\"\"
    Fraud detector with attention mechanism for interpretability.
    \"\"\"
    def __init__(self, n_features, n_heads=2, d_model=32):
        super().__init__()

        # Project features to d_model dimensions
        self.feature_embedding = Dense(d_model)

        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=0.1
        )

        self.norm = LayerNormalization()
        self.flatten = Flatten()

        # Classification head
        self.classifier = Sequential([
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        self.n_features = n_features
        self.d_model = d_model

    def call(self, inputs, training=False, return_attention=False):
        # Reshape to (batch, n_features, 1)
        x = tf.expand_dims(inputs, -1)

        # Embed features
        x = self.feature_embedding(x)  # (batch, n_features, d_model)

        # Self-attention with attention weights
        attn_output, attn_weights = self.attention(
            x, x, return_attention_scores=True, training=training
        )

        # Residual + Norm
        x = self.norm(x + attn_output)

        # Flatten and classify
        x = self.flatten(x)
        output = self.classifier(x, training=training)

        if return_attention:
            return output, attn_weights
        return output

# Build and train attention model
attn_model = AttentionFraudDetector(n_features=len(feature_names))

attn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
attn_history = attn_model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
y_pred_attn = attn_model.predict(X_test_scaled).flatten()
y_pred_attn_class = (y_pred_attn > 0.5).astype(int)

print("\\nAttention Model Performance:")
print(classification_report(y_test, y_pred_attn_class))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_attn):.3f}")"""))

    # Cell 12: Visualize attention
    cells.append(nbf.v4.new_code_cell("""# Get attention weights
_, attention_weights = attn_model(X_test_scaled[:100], return_attention=True)

# Average attention across heads and samples
# attention_weights shape: (batch, n_heads, n_features, n_features)
avg_attention = attention_weights.numpy().mean(axis=(0, 1))  # (n_features, n_features)

# Visualize attention heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full attention matrix
im = axes[0].imshow(avg_attention, cmap='Blues')
axes[0].set_xticks(range(len(feature_names)))
axes[0].set_yticks(range(len(feature_names)))
axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
axes[0].set_yticklabels(feature_names)
axes[0].set_title('Average Attention Weights (Feature Interactions)')
plt.colorbar(im, ax=axes[0])

# Feature importance from attention (sum of attention received)
feature_importance = avg_attention.sum(axis=0)
sorted_idx = np.argsort(feature_importance)

axes[1].barh([feature_names[i] for i in sorted_idx], feature_importance[sorted_idx])
axes[1].set_xlabel('Total Attention Received')
axes[1].set_title('Feature Importance from Attention')

plt.tight_layout()
plt.show()

print("\\nTop feature interactions (high attention):")
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if avg_attention[i, j] > 0.1:
            print(f"  {feature_names[i]} <-> {feature_names[j]}: {avg_attention[i,j]:.3f}")"""))

    # Cell 13: Counterfactual intro
    cells.append(nbf.v4.new_markdown_cell("""## 3. Counterfactual Explanations

Counterfactuals answer: "What minimal changes would flip the prediction?"

For fraud detection: "What would need to change for this transaction to be classified as legitimate?"

### Algorithm:
1. Start with original instance
2. Optimize to find nearest instance with different prediction
3. Constraints ensure realistic changes (e.g., can't change past transactions)"""))

    # Cell 14: Counterfactual generator
    cells.append(nbf.v4.new_code_cell("""def generate_counterfactual(model, instance, target_class, feature_names,
                                feature_ranges, immutable_features=None,
                                learning_rate=0.1, max_iterations=1000):
    \"\"\"
    Generate counterfactual explanation using gradient descent.

    Parameters:
    -----------
    model : Keras model
    instance : array, original instance
    target_class : int, desired prediction (0 or 1)
    feature_names : list of feature names
    feature_ranges : dict of (min, max) for each feature
    immutable_features : list of features that cannot change

    Returns:
    --------
    counterfactual : array, modified instance
    changes : dict of feature changes
    \"\"\"
    if immutable_features is None:
        immutable_features = []

    # Create trainable variable
    cf = tf.Variable(instance.reshape(1, -1), dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for iteration in range(max_iterations):
        with tf.GradientTape() as tape:
            pred = model(cf)

            # Loss: prediction loss + distance loss
            if target_class == 1:
                pred_loss = -tf.math.log(pred + 1e-10)
            else:
                pred_loss = -tf.math.log(1 - pred + 1e-10)

            # L1 distance for sparsity
            distance_loss = 0.1 * tf.reduce_sum(tf.abs(cf - instance))

            total_loss = pred_loss + distance_loss

        gradients = tape.gradient(total_loss, cf)
        optimizer.apply_gradients([(gradients, cf)])

        # Project to feasible region
        cf_numpy = cf.numpy().flatten()
        for i, fname in enumerate(feature_names):
            # Immutable features
            if fname in immutable_features:
                cf_numpy[i] = instance[i]
            # Clip to range
            elif fname in feature_ranges:
                cf_numpy[i] = np.clip(cf_numpy[i],
                                      feature_ranges[fname][0],
                                      feature_ranges[fname][1])

        cf.assign(cf_numpy.reshape(1, -1))

        # Check if target reached
        current_pred = model(cf).numpy().flatten()[0]
        if (target_class == 1 and current_pred > 0.5) or \\
           (target_class == 0 and current_pred < 0.5):
            break

    # Calculate changes
    cf_final = cf.numpy().flatten()
    changes = {}
    for i, fname in enumerate(feature_names):
        if abs(cf_final[i] - instance[i]) > 0.01:
            changes[fname] = {
                'original': instance[i],
                'counterfactual': cf_final[i],
                'change': cf_final[i] - instance[i]
            }

    return cf_final, changes

# Define feature constraints
feature_ranges = {
    'amount': (-3, 3),  # Scaled values
    'hour': (-3, 3),
    'day_of_week': (-3, 3),
    'merchant_category': (-3, 3),
    'distance_from_home': (-3, 3),
    'time_since_last_txn': (-3, 3),
    'avg_txn_amount': (-3, 3),
    'txn_frequency': (-3, 3),
    'is_online': (-3, 3),
    'is_international': (-3, 3)
}

# Immutable features (can't change historical data)
immutable = ['avg_txn_amount', 'txn_frequency']

print("Counterfactual generator defined")
print(f"Immutable features: {immutable}")"""))

    # Cell 15: Generate counterfactuals
    cells.append(nbf.v4.new_code_cell("""# Find a fraud prediction to explain
fraud_preds = np.where((y_pred > 0.5) & (y_test == 1))[0]

if len(fraud_preds) > 0:
    idx = fraud_preds[0]
    original = X_test_scaled[idx]
    original_pred = y_pred_proba[idx]

    print(f"Original transaction (predicted fraud with p={original_pred:.3f})")
    print("\\nGenerating counterfactual (what would make it legitimate?)...")

    # Generate counterfactual
    cf, changes = generate_counterfactual(
        model, original, target_class=0,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        immutable_features=immutable
    )

    cf_pred = model.predict(cf.reshape(1, -1), verbose=0).flatten()[0]

    print(f"\\nCounterfactual prediction: {cf_pred:.3f}")
    print(f"\\nRequired changes to flip prediction:")

    if changes:
        for fname, vals in changes.items():
            # Convert back from scaled values for interpretability
            orig_unscaled = scaler.inverse_transform(original.reshape(1, -1))[0]
            cf_unscaled = scaler.inverse_transform(cf.reshape(1, -1))[0]

            fidx = feature_names.index(fname)
            print(f"  {fname}:")
            print(f"    Original: {orig_unscaled[fidx]:.2f}")
            print(f"    Counterfactual: {cf_unscaled[fidx]:.2f}")
    else:
        print("  No significant changes needed")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(feature_names))
    width = 0.35

    ax.bar(x - width/2, original, width, label='Original (Fraud)')
    ax.bar(x + width/2, cf, width, label='Counterfactual (Legitimate)')

    ax.set_ylabel('Scaled Feature Value')
    ax.set_title('Original vs Counterfactual Feature Values')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()"""))

    # Cell 16: Model distillation intro
    cells.append(nbf.v4.new_markdown_cell("""## 4. Model Distillation

Model distillation transfers knowledge from a complex "teacher" model to a simpler, more interpretable "student" model.

### Benefits:
- Interpretable approximation of black-box model
- Preserves most of the teacher's accuracy
- Can extract decision rules

### Our approach:
Train a decision tree to mimic the neural network's predictions."""))

    # Cell 17: Distillation
    cells.append(nbf.v4.new_code_cell("""# Get soft labels from neural network
soft_labels = model.predict(X_train_scaled, verbose=0).flatten()

# Train decision tree on soft labels
# Use probability as regression target for soft distillation
dt_student = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)

# Convert to hard labels for decision tree
hard_labels = (soft_labels > 0.5).astype(int)
dt_student.fit(X_train_scaled, hard_labels)

# Evaluate student
y_pred_student = dt_student.predict(X_test_scaled)
y_pred_student_proba = dt_student.predict_proba(X_test_scaled)[:, 1]

print("Model Distillation Results:")
print("\\nTeacher (Neural Network):")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"  Accuracy: {(y_pred == y_test).mean():.3f}")

print("\\nStudent (Decision Tree):")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_student_proba):.3f}")
print(f"  Accuracy: {(y_pred_student == y_test).mean():.3f}")

# Agreement between teacher and student
agreement = (y_pred == y_pred_student).mean()
print(f"\\nTeacher-Student Agreement: {agreement*100:.1f}%")"""))

    # Cell 18: Visualize decision tree
    cells.append(nbf.v4.new_code_cell("""# Visualize the distilled decision tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    dt_student,
    feature_names=feature_names,
    class_names=['Legitimate', 'Fraud'],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=8
)

ax.set_title('Distilled Decision Tree (Approximating Neural Network)', fontsize=14)
plt.tight_layout()
plt.show()

# Extract and print rules
def get_rules(tree, feature_names, class_names):
    \"\"\"Extract rules from decision tree.\"\"\"
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, depth, rule):
        if tree_.feature[node] != -2:  # Not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left branch
            recurse(tree_.children_left[node], depth + 1,
                   rule + [f"{name} <= {threshold:.2f}"])
            # Right branch
            recurse(tree_.children_right[node], depth + 1,
                   rule + [f"{name} > {threshold:.2f}"])
        else:  # Leaf
            class_idx = np.argmax(tree_.value[node])
            class_name = class_names[class_idx]
            samples = tree_.n_node_samples[node]
            if class_name == 'Fraud' and samples > 10:
                rules.append((rule, class_name, samples))

    recurse(0, 0, [])
    return rules

rules = get_rules(dt_student, feature_names, ['Legitimate', 'Fraud'])

print("\\nExtracted Fraud Detection Rules:")
print("=" * 60)
for i, (conditions, cls, samples) in enumerate(rules[:5], 1):
    print(f"\\nRule {i} ({samples} samples):")
    for cond in conditions:
        print(f"  - {cond}")
    print(f"  => {cls}")"""))

    # Cell 19: Summary comparison
    cells.append(nbf.v4.new_code_cell("""# Compare all XAI methods
print("=" * 70)
print("ADVANCED XAI METHODS - COMPARISON")
print("=" * 70)

print("\\n1. DeepSHAP")
print("   - Provides feature attributions for neural networks")
print("   - Fast computation using backpropagation")
print("   - Best for: Understanding individual predictions")

print("\\n2. Attention Visualization")
print("   - Built-in interpretability from attention weights")
print("   - Shows feature interactions")
print("   - Best for: Understanding which features interact")

print("\\n3. Counterfactual Explanations")
print("   - Actionable 'what-if' scenarios")
print("   - Minimal changes to flip prediction")
print("   - Best for: User-facing explanations, recourse")

print("\\n4. Model Distillation")
print("   - Interpretable approximation of complex model")
print("   - Extracts decision rules")
print("   - Best for: Global model understanding, compliance")

print("\\n" + "=" * 70)
print("RECOMMENDATIONS FOR FRAUD DETECTION")
print("=" * 70)
print("- Use DeepSHAP for individual transaction review")
print("- Use attention to identify important feature interactions")
print("- Use counterfactuals for customer explanations")
print("- Use distillation for regulatory compliance and audits")"""))

    # Cell 20: Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## Summary

### Key Takeaways

1. **DeepSHAP** efficiently computes SHAP values for neural networks, revealing which features drive individual predictions.

2. **Attention mechanisms** provide built-in interpretability by showing feature interactions and importance directly from model architecture.

3. **Counterfactual explanations** offer actionable insights by showing minimal changes needed to flip a prediction - crucial for customer communication.

4. **Model distillation** creates interpretable approximations of complex models, enabling rule extraction for compliance and auditing.

### Best Practices

- Combine multiple XAI methods for comprehensive understanding
- Consider the audience: technical (SHAP) vs. non-technical (counterfactuals)
- Validate explanations against domain knowledge
- Document and version explanations for regulatory purposes

### Next Steps
- Integrate XAI into production pipelines
- Build interactive dashboards for model monitoring
- Establish explanation baselines for drift detection"""))

    nb['cells'] = cells

    # Save notebook
    output_path = '/home/user/test/notebooks/phase6_explainable_ai/02_advanced_xai_methods.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Advanced XAI notebook created: {output_path}")
    print(f"Total cells: {len(cells)}")
    return output_path

if __name__ == "__main__":
    create_notebook()
