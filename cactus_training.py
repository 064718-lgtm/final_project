"""
Training script for the Aerial Cactus Identification dataset.

This follows the workflow of the Kaggle notebook
https://www.kaggle.com/code/shahules/getting-started-with-cnn-and-vgg16
and provides two models:
  1) A compact CNN trained from scratch.
  2) A VGG16 transfer-learning model (ImageNet weights, top frozen).

Data loading is resilient to partially extracted folders: it will read
from an existing image directory when available and fall back to the
train.zip archive when a file is missing.

Requirements (install before running):
  pip install tensorflow pandas scikit-learn pillow
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


# -------- Data loading ------------------------------------------------------

def _default_train_dir() -> Optional[Path]:
    candidates = [
        Path("train"),
        Path("train_full/train"),
        Path("train/train"),
        Path("data/train"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _default_test_dir() -> Optional[Path]:
    candidates = [
        Path("test"),
        Path("test/test"),
        Path("data/test"),
        Path("test_full/test"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class ZipOrDirSequence(tf.keras.utils.Sequence):
    """Keras Sequence that loads images from a directory or a zip archive."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        target_size: Tuple[int, int],
        shuffle: bool = True,
        base_dir: Optional[Path] = None,
        zip_path: Optional[Path] = None,
        zip_prefix: str = "train",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.zip_path = zip_path
        self.zip_prefix = zip_prefix
        self.on_epoch_end()
        self._zip_ref: Optional[zipfile.ZipFile] = None
        if self.zip_path and self.zip_path.exists():
            self._zip_ref = zipfile.ZipFile(self.zip_path)

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle and not self.df.empty:
            self.df = self.df.sample(frac=1.0, random_state=None).reset_index(drop=True)

    def _open_from_zip(self, filename: str) -> Optional[Image.Image]:
        if not self._zip_ref:
            return None
        member_name = f"{self.zip_prefix}/{filename}"
        try:
            with self._zip_ref.open(member_name) as f:
                return Image.open(f).convert("RGB")
        except KeyError:
            return None

    def _open_from_dir(self, filename: str) -> Optional[Image.Image]:
        if not self.base_dir:
            return None
        path = self.base_dir / filename
        if not path.exists():
            return None
        with path.open("rb") as f:
            return Image.open(f).convert("RGB")

    def _load_image(self, filename: str) -> np.ndarray:
        img: Optional[Image.Image] = self._open_from_dir(filename)
        if img is None:
            img = self._open_from_zip(filename)
        if img is None:
            raise FileNotFoundError(f"Unable to locate image {filename}")
        img = img.resize(self.target_size)
        return np.asarray(img, dtype=np.float32)

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch = self.df.iloc[start:end]

        images = [self._load_image(fname) for fname in batch["id"].tolist()]
        x = np.stack(images, axis=0)
        if "has_cactus" in batch.columns:
            y = batch["has_cactus"].to_numpy(dtype=np.float32)
            return x, y
        return x

    def close(self) -> None:
        if self._zip_ref:
            self._zip_ref.close()

    def __del__(self) -> None:
        self.close()


# -------- Model definitions -------------------------------------------------

def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def build_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    aug = build_augmentation_layer()
    inputs = layers.Input(shape=input_shape)
    x = aug(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="simple_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_vgg16(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    aug = build_augmentation_layer()
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False  # follow the notebook: start with frozen base

    inputs = layers.Input(shape=input_shape)
    x = aug(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="vgg16_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# -------- Training / inference ---------------------------------------------

def train_and_evaluate(
    model: tf.keras.Model,
    train_seq: ZipOrDirSequence,
    val_seq: ZipOrDirSequence,
    epochs: int,
    save_path: Path,
) -> tf.keras.callbacks.History:
    callbacks: Iterable[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=list(callbacks),
        verbose=2,
    )
    return history


def predict_test(
    model: tf.keras.Model, test_seq: ZipOrDirSequence, ids: pd.Series
) -> pd.DataFrame:
    preds = model.predict(test_seq, verbose=1).reshape(-1)
    return pd.DataFrame({"id": ids, "has_cactus": preds})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN and VGG16 models for cactus classification."
    )
    parser.add_argument("--train-csv", default="train.csv", help="Path to train.csv")
    parser.add_argument("--train-zip", default="train.zip", help="Path to train.zip")
    parser.add_argument("--train-dir", default=None, help="Directory with training images")
    parser.add_argument("--test-dir", default=None, help="Directory with test images")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--image-size", type=int, default=96, help="Square image size")
    parser.add_argument("--epochs-cnn", type=int, default=8, help="Epochs for CNN")
    parser.add_argument("--epochs-vgg", type=int, default=6, help="Epochs for VGG16")
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to save models and submission"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_dir = Path(args.train_dir) if args.train_dir else _default_train_dir()
    test_dir = Path(args.test_dir) if args.test_dir else _default_test_dir()
    train_zip = Path(args.train_zip)

    print(f"Using train_dir={train_dir}, test_dir={test_dir}, train_zip={train_zip}")

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["has_cactus"],
        random_state=42,
    )

    train_seq = ZipOrDirSequence(
        train_df,
        batch_size=args.batch_size,
        target_size=image_size,
        shuffle=True,
        base_dir=train_dir,
        zip_path=train_zip,
        zip_prefix="train",
    )
    val_seq = ZipOrDirSequence(
        val_df,
        batch_size=args.batch_size,
        target_size=image_size,
        shuffle=False,
        base_dir=train_dir,
        zip_path=train_zip,
        zip_prefix="train",
    )

    cnn_model = build_cnn((*image_size, 3))
    print("Training compact CNN...")
    train_and_evaluate(
        cnn_model,
        train_seq,
        val_seq,
        epochs=args.epochs_cnn,
        save_path=output_dir / "cnn.keras",
    )

    vgg_model = build_vgg16((*image_size, 3))
    print("Training VGG16 transfer model...")
    train_and_evaluate(
        vgg_model,
        train_seq,
        val_seq,
        epochs=args.epochs_vgg,
        save_path=output_dir / "vgg16.keras",
    )

    # Prepare test predictions using the submission template
    sample_sub = pd.read_csv("sample_submission.csv")
    test_ids = sample_sub["id"]
    test_seq = ZipOrDirSequence(
        sample_sub[["id"]],
        batch_size=args.batch_size,
        target_size=image_size,
        shuffle=False,
        base_dir=test_dir,
        zip_path=Path("test.zip") if Path("test.zip").exists() else None,
        zip_prefix="test",
    )

    submission = predict_test(vgg_model, test_seq, test_ids)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
