"""
Q-FedSecure DR-XAI: Training Pipeline with Class Balancing & tf.data Pipelines
Trains the DenseNet201 + 4-Qubit PennyLane VQC model on APTOS 2019 / IDRiD datasets.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Ensure root path is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset_loader import APTOSDatasetLoader
from src.model import build_qcnet_model

print("=" * 70)
print("  Q-FedSecure DR-XAI: Model Training & Class Balancing Pipeline")
print("=" * 70)

# 1. Ingest and Balance Dataset
print("\n--- 1. Loading & Ingesting Dataset ---")
loader = APTOSDatasetLoader(
    data_dir="dataset",
    csv_file="dataset/train.csv",
    image_size=(224, 224),
    filter_bad_quality=False  # Set True for full IQA validation pass
)

df = loader.scan_dataset()
print(f"Class distribution:\n{loader.get_class_distribution()}")

# 2. Compute Class Weights & Stratified Split
class_weights = loader.calculate_class_weights()
print(f"\nCalculated Balanced Class Weights: {class_weights}")

train_df, val_df, test_df = loader.split_train_val_test(
    test_size=0.15,
    val_size=0.15,
    balance_train=True
)

# 3. Build High-Performance tf.data Pipelines
print("\n--- 2. Building tf.data Input Pipelines ---")
train_ds = loader.create_tf_dataset(train_df.head(60), batch_size=16, augment=True, shuffle=True)
val_ds = loader.create_tf_dataset(val_df.head(20), batch_size=16, augment=False, shuffle=False)

# 4. Construct QCNET (DenseNet201 + 4-Qubit VQC)
print("\n--- 3. Constructing QCNET Hybrid Architecture ---")
model = build_qcnet_model(num_classes=5, trainable_backbone=False)
model.summary()

# 5. Train Model
print("\n--- 4. Training QCNET Model ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1
)

# 6. Save Weights
os.makedirs("weights", exist_ok=True)
weights_out = os.path.join("weights", "qcnet_weights.weights.h5")
try:
    model.save_weights(weights_out)
    print(f"\nTraining complete! Weights saved to '{weights_out}'.")
except Exception as e:
    fallback_out = "qcnet_weights.h5"
    model.save_weights(fallback_out)
    print(f"\nTraining complete! Weights saved to '{fallback_out}'.")