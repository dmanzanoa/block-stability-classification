"""
models.py
=========

This module defines a collection of convolutional neural network
architectures for the block stability prediction task.  Three model
families are provided:

* A **baseline ResNet50** classifier that fine‑tunes a pretrained
  ResNet50 on the training data.
* An **InceptionV3** model augmented with a **squeeze‑and‑excitation
  (SE) block** to recalibrate channel responses.  The SE block is
  inserted after the final convolutional features to enhance
  representational power.
* A **VGG16** classifier that can optionally operate on
  segmented inputs (e.g. using GrabCut) to focus on the stack of
  blocks.

All models use an ImageNet‑pretrained backbone and add a global
average pooling layer followed by a softmax classifier for the
``num_classes`` output categories (stable heights).  The builder
functions return compiled Keras models ready for training.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def se_block(input_tensor: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    """Construct a squeeze‑and‑excitation (SE) block.

    The SE block performs global average pooling followed by two fully
    connected layers to model channelwise dependencies.  The output is
    multiplied back onto the input tensor to scale each channel.

    Parameters
    ----------
    input_tensor : Tensor
        The input feature map of shape ``(H, W, C)``.
    ratio : int, optional
        Reduction ratio for the bottleneck layer (default 16).

    Returns
    -------
    Tensor
        The rescaled feature map with the same shape as ``input_tensor``.
    """
    channels = int(input_tensor.shape[-1])
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    excitation = layers.Dense(channels // ratio, activation="relu")(squeeze)
    excitation = layers.Dense(channels, activation="sigmoid")(excitation)
    excitation = layers.Reshape((1, 1, channels))(excitation)
    return layers.multiply([input_tensor, excitation])


def build_resnet50(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build a baseline ResNet50 classifier.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input images, e.g. ``(224, 224, 3)``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    Model
        A compiled Keras model.
    """
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False  # freeze convolutional base initially
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="resnet50_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_inceptionv3_se(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build an InceptionV3 classifier with a squeeze‑and‑excitation block.

    The SE block is applied to the final convolutional output of the
    InceptionV3 backbone prior to global pooling.  This encourages the
    network to recalibrate channel importance and can lead to improved
    performance with only a modest increase in parameters.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input images.
    num_classes : int
        Number of output classes.

    Returns
    -------
    Model
        A compiled Keras model.
    """
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = se_block(x, ratio=16)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="inceptionv3_se")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_vgg16(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Build a VGG16 classifier.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input images.
    num_classes : int
        Number of output classes.

    Returns
    -------
    Model
        A compiled Keras model.
    """
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="vgg16_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
