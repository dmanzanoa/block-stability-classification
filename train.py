"""
train.py
========

This script provides a command‑line interface for training and
evaluating block stability prediction models.  It supports three model
variants (ResNet50 baseline, InceptionV3 with squeeze‑and‑excitation,
and VGG16) defined in the ``models`` module.  The script loads
training and validation data from CSV files and corresponding image
directories, constructs the chosen model, fits it to the training data
and reports validation accuracy.  Optionally, it can generate
predictions on a test set and write them to a CSV file in the format
expected by Kaggle competitions.

Example usage::

    python train.py \
        --train-csv data/train.csv --train-dir data/train_images \
        --val-csv data/val.csv --val-dir data/val_images \
        --model inceptionv3_se \
        --epochs 10 --batch-size 32 \
        --output-csv results/predictions.csv

"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from stability_dataset import load_dataset
from models import build_resnet50, build_inceptionv3_se, build_vgg16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train block stability classifiers")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--train-dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory with validation images")
    parser.add_argument("--test-csv", type=str, help="Path to test CSV file (optional)")
    parser.add_argument("--test-dir", type=str, help="Directory with test images (optional)")
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "inceptionv3_se", "vgg16"],
        default="resnet50",
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini‑batch size")
    parser.add_argument("--output-csv", type=str, help="Path to write test predictions (optional)")
    return parser.parse_args()


def build_model(name: str, input_shape: tuple, num_classes: int) -> tf.keras.Model:
    if name == "resnet50":
        return build_resnet50(input_shape, num_classes)
    if name == "inceptionv3_se":
        return build_inceptionv3_se(input_shape, num_classes)
    if name == "vgg16":
        return build_vgg16(input_shape, num_classes)
    raise ValueError(f"Unknown model name '{name}'")


def main() -> None:
    args = parse_args()
    # Load training and validation data
    X_train, y_train = load_dataset(args.train_csv, args.train_dir)
    X_val, y_val = load_dataset(args.val_csv, args.val_dir)

    # Determine input shape and number of classes
    input_shape = X_train.shape[1:]  # (H, W, C)
    num_classes = int(np.max(y_train)) + 1

    # Build and train model
    model = build_model(args.model, input_shape, num_classes)
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    # If test data and output path provided, generate predictions
    if args.test_csv and args.test_dir and args.output_csv:
        X_test, _ = load_dataset(args.test_csv, args.test_dir)
        preds = model.predict(X_test, batch_size=args.batch_size)
        pred_labels = np.argmax(preds, axis=1)
        # Write CSV with id and stable_height columns
        import pandas as pd

        df_test = pd.read_csv(args.test_csv)
        df_out = pd.DataFrame({"id": df_test["id"], "stable_height": pred_labels})
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
