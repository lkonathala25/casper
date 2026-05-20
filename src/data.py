import math
import numpy as np
import torch
from torch.utils.data import Dataset


def digits_to_number(digits):
    """Convert an MSB-first sequence of digits to a Python int."""
    number = 0
    for d in digits:
        number = number * 10 + d
    return number


def carry_chain_output_digits(a_digits, b_digits):
    """Compute MSB-first output digits of a + b via the carry chain.

    Both inputs are MSB-first lists of digits, same length N. Returns a list
    of N+1 digits (MSB-first); index 0 is the carry-out.
    """
    n = len(a_digits)
    s = [0] * (n + 1)
    carry = 0
    for j in range(n):
        i = n - 1 - j
        raw = a_digits[i] + b_digits[i] + carry
        s[i + 1] = raw % 10
        carry = raw // 10
    s[0] = carry
    return s


def mnist_sum_num_samples(split):
    if split == 'train':
        return 5000
    if split == 'test':
        return 1000
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


def mnist_add_num_samples(n_digits, split):
    if split == 'train':
        return 60000 // (2 * n_digits)
    if split == 'test':
        return 10000 // (2 * n_digits)
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


class MNISTSumDataset(Dataset):
    """N-digit MNIST-Sum: each sample is N MNIST images, label is sum of digits."""

    def __init__(self, mnist_dataset, n_digits, num_samples, data_seed=0):
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        self.num_samples = num_samples

        rng = np.random.RandomState(data_seed)
        self.sample_indices = rng.randint(
            0, len(mnist_dataset), size=(num_samples, n_digits))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chosen = self.sample_indices[idx].tolist()
        images, labels = [], []
        for i in chosen:
            img, label = self.mnist[i]
            images.append(img)
            labels.append(label)

        stacked = torch.cat(images, dim=0)
        digit_labels = torch.tensor(labels, dtype=torch.long)
        sum_label = torch.tensor(sum(labels), dtype=torch.long)
        return stacked, digit_labels, sum_label


class MNISTAddDataset(Dataset):
    """N-digit MNIST-Add: 2N MNIST images grouped into two N-digit operands."""

    def __init__(self, mnist_dataset, n_digits, num_samples, data_seed=0):
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        self.n_images = 2 * n_digits
        self.num_samples = num_samples

        rng = np.random.RandomState(data_seed)
        self.sample_indices = rng.randint(
            0, len(mnist_dataset), size=(num_samples, self.n_images))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chosen = self.sample_indices[idx].tolist()
        images, labels = [], []
        for i in chosen:
            img, label = self.mnist[i]
            images.append(img)
            labels.append(label)

        stacked = torch.cat(images, dim=0)
        digit_labels = torch.tensor(labels, dtype=torch.long)

        a_digits = labels[:self.n_digits]
        b_digits = labels[self.n_digits:]

        output_digits = carry_chain_output_digits(a_digits, b_digits)
        output_digit_labels = torch.tensor(output_digits, dtype=torch.long)

        if self.n_digits <= 18:
            sum_label = torch.tensor(
                digits_to_number(a_digits) + digits_to_number(b_digits),
                dtype=torch.long,
            )
        else:
            # Place-value sum overflows int64 for N > 18; sum_label is only
            # used for diagnostic logging, so emit a sentinel.
            sum_label = torch.tensor(-1, dtype=torch.long)

        return stacked, digit_labels, output_digit_labels, sum_label
