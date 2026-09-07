from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from node_categories import edge_labels_to_node_labels


class WarcraftDataset(Dataset):
    """Returns (image, node_label, edge_label) per sample.

    Args:
        data_dir: directory containing {split}_maps.npy and
                  {split}_edge_labels.npy.
        split:    'train' | 'val' | 'test'.
        node_info, node_order: from node_categories.build_node_categories().
    """

    def __init__(self, data_dir, split, node_info, node_order):
        data_dir = Path(data_dir)
        self.maps = np.load(data_dir / f"{split}_maps.npy")
        edge_labels = np.load(data_dir / f"{split}_edge_labels.npy")
        self.node_labels = edge_labels_to_node_labels(edge_labels, node_info, node_order)
        self.edge_labels = edge_labels

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.maps[idx]).float().div_(255.0).permute(2, 0, 1)
        node_label = torch.from_numpy(self.node_labels[idx]).long()
        edge_label = torch.from_numpy(self.edge_labels[idx]).float()
        return img, node_label, edge_label
