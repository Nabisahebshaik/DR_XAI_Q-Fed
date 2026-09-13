"""
Q-FedSecure DR-XAI: Data Ingestion, Class Balancing & Augmentation Pipeline
Supports APTOS 2019 Blindness Detection and IDRiD (Indian Diabetic Retinopathy Image Dataset).
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Generator, Union
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.preprocessing import assess_image_quality, apply_clahe_enhancement

# Standard ICDR 5-class DR Grade Labels
GRADE_MAPPINGS = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

STRING_TO_GRADE = {
    "no_dr": 0, "0": 0, "normal": 0, "nondr": 0,
    "mild": 1, "1": 1, "mild_npdr": 1,
    "moderate": 2, "2": 2, "mod": 2, "moderate_npdr": 2,
    "severe": 3, "3": 3, "severe_npdr": 3,
    "proliferate_dr": 4, "proliferative_dr": 4, "4": 4, "pdr": 4
}


class APTOSDatasetLoader:
    """
    Data ingestion pipeline for the APTOS 2019 Blindness Detection benchmark.
    Handles raw image indexing, quality assessment filtering, class balancing,
    stratified splitting, and tf.data generation.
    """
    def __init__(
        self,
        data_dir: str = "dataset",
        csv_file: Optional[str] = "dataset/train.csv",
        image_size: Tuple[int, int] = (224, 224),
        filter_bad_quality: bool = False
    ):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.image_size = image_size
        self.filter_bad_quality = filter_bad_quality
        self.dataframe: Optional[pd.DataFrame] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def scan_dataset(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Scans directory structure or CSV metadata to index all valid fundus images.
        Supports both CSV-indexed format and class-folder directory structures.
        """
        records = []

        # 1. Strategy A: Check for subfolder-based structure (dataset/colored_images/<Grade>/<image>.png)
        subfolder_dir = os.path.join(self.data_dir, "colored_images")
        if os.path.exists(subfolder_dir):
            for class_name in os.listdir(subfolder_dir):
                c_dir = os.path.join(subfolder_dir, class_name)
                if os.path.isdir(c_dir):
                    grade_val = STRING_TO_GRADE.get(class_name.lower(), None)
                    if grade_val is not None:
                        files = os.listdir(c_dir)
                        if max_samples:
                            files = files[:max_samples]
                        for img_f in files:
                            if img_f.lower().endswith((".png", ".jpg", ".jpeg")):
                                full_p = os.path.join(c_dir, img_f)
                                records.append({"image_path": full_p, "diagnosis": grade_val, "class_name": GRADE_MAPPINGS[grade_val]})

        # 2. Strategy B: Check CSV metadata if available
        if len(records) == 0 and self.csv_file and os.path.exists(self.csv_file):
            df_csv = pd.read_csv(self.csv_file)
            img_col = "id_code" if "id_code" in df_csv.columns else df_csv.columns[0]
            diag_col = "diagnosis" if "diagnosis" in df_csv.columns else df_csv.columns[1]

            if max_samples:
                df_csv = df_csv.head(max_samples)

            for _, row in df_csv.iterrows():
                fname = str(row[img_col])
                diag = int(row[diag_col])
                for ext in [".png", ".jpg", ".jpeg", ""]:
                    test_p = os.path.join(self.data_dir, fname + ext)
                    if os.path.exists(test_p):
                        records.append({"image_path": test_p, "diagnosis": diag, "class_name": GRADE_MAPPINGS.get(diag, str(diag))})
                        break

        df = pd.DataFrame(records)
        print(f"[DatasetLoader] Indexed {len(df)} fundus images across {df['diagnosis'].nunique() if len(df) > 0 else 0} classes.")

        # 3. Optional Quality Assessment (IQA) Filtering
        if self.filter_bad_quality and len(df) > 0:
            valid_paths = []
            print("[DatasetLoader] Running real-time Laplacian & Luminance Quality Assessment filter...")
            for idx, row in df.iterrows():
                is_valid, _ = assess_image_quality(row["image_path"])
                valid_paths.append(is_valid)
            
            initial_count = len(df)
            df = df[valid_paths].reset_index(drop=True)
            print(f"[DatasetLoader] Retained {len(df)}/{initial_count} high-quality gradeable images.")

        self.dataframe = df
        return df

    def get_class_distribution(self) -> pd.Series:
        """Returns the class frequency count."""
        if self.dataframe is None or len(self.dataframe) == 0:
            self.scan_dataset()
        return self.dataframe["diagnosis"].value_counts().sort_index()

    def calculate_class_weights(self) -> Dict[int, float]:
        """
        Computes balanced class weights to counteract severe imbalance
        in Grade 3 (Severe) and Grade 4 (Proliferative DR).
        """
        if self.dataframe is None or len(self.dataframe) == 0:
            self.scan_dataset()

        y = self.dataframe["diagnosis"].values
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        return self.class_weights

    def balance_classes_oversample(self, max_samples_per_class: Optional[int] = None) -> pd.DataFrame:
        """
        Performs random oversampling on minority classes (e.g. Grade 1, 3, 4)
        so each class has an equal representation during training.
        """
        if self.dataframe is None or len(self.dataframe) == 0:
            self.scan_dataset()

        df = self.dataframe.copy()
        class_counts = df["diagnosis"].value_counts()
        target_count = max_samples_per_class or class_counts.max()

        balanced_dfs = []
        for cls in np.unique(df["diagnosis"].values):
            cls_df = df[df["diagnosis"] == cls]
            count = len(cls_df)
            if count < target_count:
                cls_oversampled = cls_df.sample(target_count, replace=True, random_state=42)
                balanced_dfs.append(cls_oversampled)
            else:
                balanced_dfs.append(cls_df.sample(target_count, replace=False, random_state=42))

        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[DatasetLoader] Balanced dataset via oversampling to {len(balanced_df)} samples ({target_count} per class).")
        return balanced_df

    def split_train_val_test(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        balance_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs stratified partitioning into Train, Validation, and Test sets.
        """
        if self.dataframe is None or len(self.dataframe) == 0:
            self.scan_dataset()

        df = self.dataframe.copy()
        y = df["diagnosis"].values

        # Split off Test set
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, stratify=y, random_state=42
        )

        # Split remaining into Train and Validation
        val_ratio_adjusted = val_size / (1.0 - test_size)
        y_train_val = df_train_val["diagnosis"].values
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_ratio_adjusted, stratify=y_train_val, random_state=42
        )

        if balance_train:
            # Balance only the training cohort
            class_counts = df_train["diagnosis"].value_counts()
            target_count = class_counts.max()
            balanced_train_dfs = []
            for cls in np.unique(df_train["diagnosis"].values):
                cls_df = df_train[df_train["diagnosis"] == cls]
                if len(cls_df) < target_count:
                    cls_oversampled = cls_df.sample(target_count, replace=True, random_state=42)
                    balanced_train_dfs.append(cls_oversampled)
                else:
                    balanced_train_dfs.append(cls_df)
            df_train = pd.concat(balanced_train_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

        print(f"[DatasetLoader] Split Summary: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        return df_train, df_val, df_test

    def create_tf_dataset(
        self,
        df_split: pd.DataFrame,
        batch_size: int = 16,
        augment: bool = False,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Constructs a high-performance tf.data pipeline with CLAHE preprocessing
        and domain-specific retinal data augmentation.
        """
        image_paths = df_split["image_path"].values
        labels = df_split["diagnosis"].values.astype(np.int32)

        def _load_and_preprocess(path_tensor, label):
            path_str = path_tensor.numpy().decode("utf-8")
            img_processed = apply_clahe_enhancement(path_str, target_size=self.image_size)
            if img_processed is None:
                img_processed = np.zeros((*self.image_size, 3), dtype=np.float32)
            return img_processed, label

        def _tf_map(path, label):
            img, lbl = tf.py_function(
                _load_and_preprocess,
                inp=[path, label],
                Tout=[tf.float32, tf.int32]
            )
            img.set_shape([*self.image_size, 3])
            lbl.set_shape([])
            return img, lbl

        def _augment_img(img, label):
            # Retinal-safe augmentations
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.08)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            dataset = dataset.map(_augment_img, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(500, len(df_split)))

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


class IDRiDDatasetLoader:
    """
    Data ingestion pipeline for the Indian Diabetic Retinopathy Image Dataset (IDRiD).
    Handles pixel-level lesion segmentation masks (Microaneurysms, Hemorrhages,
    Hard Exudates, Soft Exudates, and Optic Disc).
    """
    def __init__(self, idrid_root: str = "data/IDRiD"):
        self.idrid_root = idrid_root

    def list_available_modalities(self) -> Dict[str, List[str]]:
        """Scans for IDRiD ground-truth lesion masks."""
        lesion_types = ["Microaneurysms", "Hemorrhages", "Hard_Exudates", "Soft_Exudates", "Optic_Disc"]
        found_files = {}

        for l_type in lesion_types:
            pattern = os.path.join(self.idrid_root, "**", f"*{l_type}*", "*.tif*")
            matches = glob.glob(pattern, recursive=True)
            found_files[l_type] = matches

        return found_files

    def load_image_and_lesion_masks(
        self,
        image_name: str,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Loads the fundus photograph along with all matching IDRiD binary lesion masks.
        """
        raw_img_pattern = os.path.join(self.idrid_root, "**", f"{image_name}.jpg")
        matches = glob.glob(raw_img_pattern, recursive=True)
        
        if not matches:
            # Synthetic canvas when physical IDRiD folder is offline/not downloaded
            fundus = np.zeros((*target_size, 3), dtype=np.uint8)
            cv2.circle(fundus, (target_size[0]//2, target_size[1]//2), target_size[0]//2 - 10, (180, 80, 50), -1)
            masks = {
                "Microaneurysms": np.zeros(target_size, dtype=np.uint8),
                "Hemorrhages": np.zeros(target_size, dtype=np.uint8),
                "Hard_Exudates": np.zeros(target_size, dtype=np.uint8),
                "Optic_Disc": np.zeros(target_size, dtype=np.uint8)
            }
            return fundus, masks

        fundus_raw = cv2.imread(matches[0])
        fundus = cv2.cvtColor(fundus_raw, cv2.COLOR_BGR2RGB)
        fundus = cv2.resize(fundus, target_size)

        masks = {}
        for l_type in ["Microaneurysms", "Hemorrhages", "Hard_Exudates", "Soft_Exudates", "Optic_Disc"]:
            mask_pattern = os.path.join(self.idrid_root, "**", f"{image_name}_{l_type}*.tif")
            mask_matches = glob.glob(mask_pattern, recursive=True)
            if mask_matches:
                m = cv2.imread(mask_matches[0], cv2.IMREAD_GRAYSCALE)
                masks[l_type] = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
            else:
                masks[l_type] = np.zeros(target_size, dtype=np.uint8)

        return fundus, masks
