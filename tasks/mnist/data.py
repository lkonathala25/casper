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


_IMAGE_BANKS = {}


def image_bank(mnist_dataset):
    """All images of an MNIST split as one (len, 28, 28) tensor, plus labels.

    Built once per dataset object by running its own transform on every image,
    so indexing the bank returns exactly what `mnist_dataset[i]` would, without
    per-image PIL conversion on every access.
    """
    key = id(mnist_dataset)
    if key not in _IMAGE_BANKS:
        images, labels = zip(*(mnist_dataset[i] for i in range(len(mnist_dataset))))
        _IMAGE_BANKS[key] = (mnist_dataset, torch.cat(images, dim=0),
                             torch.tensor(labels, dtype=torch.long))
    return _IMAGE_BANKS[key][1:]


class MNISTSumDataset(Dataset):
    """N-digit MNIST-Sum: each sample is N MNIST images, label is sum of digits."""

    def __init__(self, mnist_dataset, n_digits, num_samples, data_seed=0):
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        self.num_samples = num_samples

        rng = np.random.RandomState(data_seed)
        self.sample_indices = rng.randint(
            0, len(mnist_dataset), size=(num_samples, n_digits))
        self.images, self.labels = image_bank(mnist_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chosen = torch.from_numpy(self.sample_indices[idx])
        stacked = self.images[chosen]
        labels = self.labels[chosen].tolist()
        digit_labels = torch.tensor(labels, dtype=torch.long)
        sum_label = torch.tensor(sum(labels), dtype=torch.long)
        return stacked, digit_labels, sum_label


def ctsketch_add_copies(n_digits):
    """How many copies of the MNIST split CTSketch's Add loader deals from at this N."""
    for threshold, copies in ((200, 5), (100, 4), (50, 3), (25, 2)):
        if n_digits >= threshold:
            return copies
    return 1


def ctsketch_add_indices(split_size, n_digits, data_seed):
    """Deal a shuffled pool of MNIST indices into non-overlapping Add examples.

    Mirrors CTSketch's MNISTMultiDigitSum2Dataset: the split is repeated
    ctsketch_add_copies(N) times, shuffled, and cut into consecutive groups of
    2N images, so every image appears once per copy and the number of examples
    is (copies * split_size) // 2N. Returns an (examples, 2N) index array.
    """
    n_images = 2 * n_digits
    pool = np.tile(np.arange(split_size), ctsketch_add_copies(n_digits))
    np.random.RandomState(data_seed).shuffle(pool)
    n_examples = len(pool) // n_images
    return pool[: n_examples * n_images].reshape(n_examples, n_images)


class MNISTAddDataset(Dataset):
    """N-digit MNIST-Add: 2N MNIST images grouped into two N-digit operands."""

    def __init__(self, mnist_dataset, n_digits, num_samples, data_seed=0,
                 protocol='resample'):
        """protocol 'resample' draws num_samples examples with replacement (images
        reused); 'ctsketch' deals the split into non-overlapping examples as
        CTSketch does (ctsketch_add_indices) and ignores num_samples."""
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        self.n_images = 2 * n_digits

        if protocol == 'ctsketch':
            self.sample_indices = ctsketch_add_indices(len(mnist_dataset), n_digits, data_seed)
        elif protocol == 'resample':
            rng = np.random.RandomState(data_seed)
            self.sample_indices = rng.randint(
                0, len(mnist_dataset), size=(num_samples, self.n_images))
        else:
            raise ValueError(f"protocol must be 'resample' or 'ctsketch', got {protocol!r}")
        self.num_samples = len(self.sample_indices)
        self.images, self.labels = image_bank(mnist_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chosen = torch.from_numpy(self.sample_indices[idx])
        stacked = self.images[chosen]
        labels = self.labels[chosen].tolist()
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
