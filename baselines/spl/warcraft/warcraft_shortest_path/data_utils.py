import os
import numpy as np
import torch
import os

from comb_modules.dijkstra import dijkstra
from decorators import input_to_numpy
from utils import TrainingIterator

def _load_labels(data_dir, prefix):
  """Edge labels: upstream names them *_shortest_paths.npy, this repo *_edge_labels.npy (identical arrays)."""
  path = os.path.join(data_dir, prefix + "_shortest_paths.npy")
  if not os.path.exists(path):
    path = os.path.join(data_dir, prefix + "_edge_labels.npy")
  return np.load(path)


def load_dataset(data_dir, use_test_set, evaluate_with_extra, normalize, use_local_path, train_limit=None):
  train_prefix = "train"
  data_suffix = "maps"
  true_weights_suffix = ""

  val_prefix = ("test" if use_test_set else "val") + ("_extra" if evaluate_with_extra else "")
  train_data_path = os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")

  if os.path.exists(train_data_path):
    train_inputs = np.load(os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")).astype(np.float32)
    # noinspection PyTypeChecker
    train_inputs = train_inputs.transpose(0, 3, 1, 2)  # channel first

    train_labels = _load_labels(data_dir, train_prefix)
    train_true_weights = np.load(os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))
    if train_limit:
      train_inputs, train_labels, train_true_weights = (
        train_inputs[:train_limit], train_labels[:train_limit], train_true_weights[:train_limit])
    if normalize:
      mean, std = (
        np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
        np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
      )
      train_inputs -= mean
      train_inputs /= std
    train_iterator = TrainingIterator(dict(images=train_inputs, labels=train_labels, true_weights=train_true_weights))
  else:
      raise Exception(f"Cannot find {train_data_path}")

  val_inputs = np.load(os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")).astype(np.float32)
  # noinspection PyTypeChecker
  val_inputs = val_inputs.transpose(0, 3, 1, 2)  # channel first

  if normalize:
    # noinspection PyUnboundLocalVariable
    val_inputs -= mean
    # noinspection PyUnboundLocalVariable
    val_inputs /= std

  val_labels = _load_labels(data_dir, val_prefix)
  val_true_weights = np.load(os.path.join(data_dir, val_prefix + "_vertex_weights.npy"))
  val_full_images = np.load(os.path.join(data_dir, val_prefix + "_maps.npy"))
  eval_iterator = TrainingIterator(
    dict(images=val_inputs, labels=val_labels, true_weights=val_true_weights)
  )

  @input_to_numpy
  def denormalize(x):
    return (x * std) + mean

  metadata = {
    "input_image_size": val_full_images[0].shape[1],
    "output_features": val_true_weights[0].shape[0] * val_true_weights[0].shape[1],
    "num_channels": val_full_images[0].shape[-1],
    "denormalize": denormalize
  }

  return train_iterator, eval_iterator, metadata
