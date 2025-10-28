# Block Stability Classification

This project implements several deep learning models to predict the **stable height** of a stack of blocks from a single image. In the ShapeStacks dataset, each image depicts a vertical stack of blocks rendered in a 3D environment; the blocks vary in colour, shape and material (cubes, rectangular solids, spheres, cylinders) and the height of the stack ranges from 2–6 blocks. A stable stack has all blocks correctly placed and its stable height equals the total number of blocks. In an unstable stack, only the bottom portion of the stack is stable and the stable height is the number of blocks that would remain upright. The goal is to build models that can estimate this stable height directly from image pixels.

Three convolutional neural network architectures are provided:

* **ResNet50 baseline** – A standard residual network pretrained on ImageNet and fine‑tuned on the ShapeStacks images to predict the stable height. The convolutional base is frozen initially and only the classifier head is trained.
* **InceptionV3 with Squeeze‑and‑Excitation (SE)** – An InceptionV3 backbone augmented with a squeeze‑and‑excitation block to model channelwise dependencies and improve representational power.
* **VGG16 classifier** – A VGG16 backbone that can be paired with foreground segmentation (e.g. using GrabCut) to isolate the stack from the background before classification.

## Repository Structure

| File | Purpose |
| --- | --- |
| `stability_dataset.py` | Functions for loading images and labels from CSV/ directory format, applying GrabCut foreground segmentation and masking images. |
| `models.py` | Builders for ResNet50, InceptionV3+SE and VGG16 models using TensorFlow/Keras. |
| `train.py` | Command‑line script for training and evaluating a selected model, with options to generate test predictions. |
| `requirements.txt` | Minimal list of Python dependencies. |
| `README.md` | Project description and usage instructions. |

## Installation

Install the required Python packages (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

## Data Preparation

The code assumes the ShapeStacks data is organised as follows:

```
data/
  train.csv        # Contains columns id, stable_height
  train_images/    # JPEG images with filenames `<id>.jpg`
  val.csv
  val_images/
  test.csv         # (optional) only id column, no labels
  test_images/
```

Each CSV must include an `id` column identifying the image and, for training/validation data, a `stable_height` column with the ground‑truth labels. You can apply GrabCut or other segmentation methods before feeding images into the VGG16 model to suppress background clutter.

## Training and Evaluation

Use the `train.py` script to train and evaluate a model. For example, to train the InceptionV3+SE model for 10 epochs on a GPU:

```bash
python train.py \
  --train-csv data/train.csv --train-dir data/train_images \
  --val-csv data/val.csv --val-dir data/val_images \
  --model inceptionv3_se \
  --epochs 10 --batch-size 32
```

The script will print a summary of the network, train the model and report validation accuracy. To generate predictions for the test set, specify `--test-csv`, `--test-dir` and `--output-csv`; the script will write a CSV file with the required `id` and `stable_height` columns.

