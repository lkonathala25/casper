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


# CTSketch's MNIST-Sum sizes (Choi et al., 2025, mnist_sum/mnist_config.py): 6000
# examples dealt from the training split, split 5000 train / 1000 held out, and
# 1000 test examples dealt from the test split.
SUM_DEALT_TRAIN, SUM_TRAIN, SUM_TEST = 6000, 5000, 1000


def ctsketch_sum_indices(split_size, n_digits, n_examples, data_seed):
    """Deal MNIST indices into n_examples non-overlapping N-image Sum examples, as CTSketch does.

    Mirrors CTSketch's MNISTSumDataset: the pool is the split repeated q times
    plus its first r images, where q, r = divmod(n_examples * N, split_size);
    it is shuffled and cut into consecutive groups of N. Returns (n_examples, N).
    """
    q, r = divmod(n_examples * n_digits, split_size)
    pool = np.concatenate([np.tile(np.arange(split_size), q), np.arange(r)])
    np.random.RandomState(data_seed).shuffle(pool)
    return pool.reshape(n_examples, n_digits)


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
    """N-digit MNIST-Sum: each sample is N MNIST images, label is sum of digits.

    Built exactly as CTSketch's MNISTSumDataset (ctsketch_sum_indices). Training:
    SUM_DEALT_TRAIN examples dealt from the training split, of which a random
    SUM_TRAIN are kept (CTSketch holds the other 1000 out for validation). Test:
    SUM_TEST examples dealt from the test split.
    """

    def __init__(self, mnist_dataset, n_digits, split, data_seed=0):
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        if split == 'train':
            dealt = ctsketch_sum_indices(len(mnist_dataset), n_digits, SUM_DEALT_TRAIN, data_seed)
            keep = np.random.RandomState(data_seed + 1).permutation(SUM_DEALT_TRAIN)[:SUM_TRAIN]
            self.sample_indices = dealt[np.sort(keep)]
        elif split == 'test':
            self.sample_indices = ctsketch_sum_indices(len(mnist_dataset), n_digits, SUM_TEST, data_seed)
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.num_samples = len(self.sample_indices)
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
    """N-digit MNIST-Add: 2N MNIST images grouped into two N-digit operands.

    Built exactly as CTSketch's MNISTMultiDigitSum2Dataset (ctsketch_add_indices):
    the split, repeated ctsketch_add_copies(N) times, is shuffled and dealt into
    non-overlapping examples, so a split of S images gives
    copies·S // 2N examples (15000/2500 at N=2 down to 585/97 at N=256).
    """

    def __init__(self, mnist_dataset, n_digits, data_seed=0):
        self.mnist = mnist_dataset
        self.n_digits = n_digits
        self.n_images = 2 * n_digits
        self.sample_indices = ctsketch_add_indices(len(mnist_dataset), n_digits, data_seed)
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
