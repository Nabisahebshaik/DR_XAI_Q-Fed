"""
Q-FedSecure DR-XAI: Advanced Fine-Tuning & Evaluation Pipeline
Two-Phase Quantum-Classical Transfer Learning:
  Phase 1: Quantum Layer & Classifier Head Warmup (DenseNet201 frozen)
  Phase 2: Fine-Tuning Deep Convolutional Feature Maps (Top Conv Blocks unfrozen, lr=1e-5)
Includes Quadratic Weighted Kappa (QWK) & Confusion Matrix Diagnostics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import tensorflow as tf

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset_loader import APTOSDatasetLoader, GRADE_MAPPINGS
from src.model import build_qcnet_model, DR_CLASSES


def train_and_fine_tune(
    data_dir: str = "dataset",
    csv_file: str = "dataset/train.csv",
    batch_size: int = 16,
    phase1_epochs: int = 4,
    phase2_epochs: int = 4,
    sample_limit: int = 300,  # Limits sample size for fast iteration; set to None for full dataset
    weights_dir: str = "weights"
):
    print("\n" + "=" * 75)
    print("  Q-FedSecure DR-XAI: Two-Stage Quantum-Classical Fine-Tuning")
    print("=" * 75)

    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "qcnet_weights.weights.h5")

    # -------------------------------------------------------------------------
    # 1. Dataset Loading, Quality Check & Stratified Partitioning
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Ingesting and Partitioning APTOS 2019 / IDRiD Dataset...")
    loader = APTOSDatasetLoader(data_dir=data_dir, csv_file=csv_file, image_size=(224, 224))
    df = loader.scan_dataset()

    if sample_limit and len(df) > sample_limit:
        print(f"[Info] Sampling {sample_limit} balanced records for optimized training loop...")
        balanced_sample = []
        for diag_val in np.unique(df["diagnosis"]):
            sub = df[df["diagnosis"] == diag_val]
            n_samples = min(len(sub), max(1, sample_limit // 5))
            balanced_sample.append(sub.sample(n_samples, random_state=42))
        df = pd.concat(balanced_sample, ignore_index=True)
        loader.dataframe = df

    class_dist = loader.get_class_distribution()
    class_weights = loader.calculate_class_weights()
    print("Class Distribution:\n", class_dist)
    print("Computed Loss Class Weights:", class_weights)

    train_df, val_df, test_df = loader.split_train_val_test(
        test_size=0.15,
        val_size=0.15,
        balance_train=True
    )

    print(f"Dataset Cohorts: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # -------------------------------------------------------------------------
    # 2. Asynchronous tf.data Input Pipeline Creation
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Constructing tf.data Asynchronous Input Pipelines...")
    train_ds = loader.create_tf_dataset(train_df, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = loader.create_tf_dataset(val_df, batch_size=batch_size, augment=False, shuffle=False)
    test_ds = loader.create_tf_dataset(test_df, batch_size=batch_size, augment=False, shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Phase 1: Quantum Layer & Classifier Head Warmup
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Phase 1: Training Quantum VQC & Bottleneck (DenseNet201 Frozen)...")
    model = build_qcnet_model(num_classes=5, trainable_backbone=False)
    
    # Warmup optimizer with slightly higher learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 4. Phase 2: Unfreezing Deep Convolutional Blocks & Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Phase 2: Fine-Tuning Deep Feature Extractor (lr=1e-5)...")
    
    # Unfreeze the DenseNet201 backbone for fine-tuning
    densenet_layer = None
    for l in model.layers:
        if "densenet" in l.name.lower():
            densenet_layer = l
            break

    if densenet_layer is not None:
        densenet_layer.trainable = True
        # Freeze initial 500 layers, fine-tune the final dense block and transition layer
        for sublayer in densenet_layer.layers[:-30]:
            sublayer.trainable = False
        print(f"Unfrozen last 30 layers of {densenet_layer.name} for DR lesion feature adaptation.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_weights_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=[checkpoint, tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)],
        verbose=1
    )

    # Save final model weights
    try:
        model.save_weights(best_weights_path)
        print(f"\n[Saved] Best fine-tuned QCNET weights saved to: {best_weights_path}")
    except Exception as e:
        print(f"[Warning] Error saving weights: {e}")

    # -------------------------------------------------------------------------
    # 5. Model Evaluation & Clinical Metrics (Quadratic Weighted Kappa)
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Comprehensive Clinical Evaluation on Held-Out Test Cohort...")
    
    y_true = []
    y_pred = []
    
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(batch_y.numpy())
        y_pred.extend(pred_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = np.mean(y_true == y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print("\n" + "=" * 50)
    print(f"  Test Multiclass Accuracy:  {acc * 100:.2f}%")
    print(f"  Quadratic Weighted Kappa:  {qwk:.4f} (QWK)")
    print("=" * 50)

    # Classification Report
    target_names = [f"Grade {k} ({GRADE_MAPPINGS[k]})" for k in range(5)]
    present_classes = np.unique(np.concatenate([y_true, y_pred]))
    filtered_names = [target_names[i] for i in present_classes]
    
    print("\nDetailed Clinical Classification Report:")
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=filtered_names, zero_division=0))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"G{i}" for i in range(5)], 
                yticklabels=[f"G{i}" for i in range(5)])
    plt.title(f"QCNET Confusion Matrix (QWK: {qwk:.3f}, Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted DR Grade")
    plt.ylabel("Ground Truth Grade")
    plt.tight_layout()
    
    cm_path = os.path.join(weights_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Diagnostic] Confusion matrix plot saved to '{cm_path}'.")

    print("\n[SUCCESS] Fine-Tuning & Evaluation Pipeline Finished Successfully!")
    return model, history_p2, qwk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune QCNET hybrid model on APTOS/IDRiD")
    parser.add_argument("--epochs_p1", type=int, default=3, help="Phase 1 Warmup Epochs")
    parser.add_argument("--epochs_p2", type=int, default=3, help="Phase 2 Fine-Tuning Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--samples", type=int, default=200, help="Subset sample limit for faster training")
    args = parser.parse_args()

    train_and_fine_tune(
        phase1_epochs=args.epochs_p1,
        phase2_epochs=args.epochs_p2,
        batch_size=args.batch_size,
        sample_limit=args.samples
    )
