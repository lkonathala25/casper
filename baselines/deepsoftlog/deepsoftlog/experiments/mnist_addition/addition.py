from pathlib import Path
import torch

from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.embeddings.initialize_vector import SPECIAL_MODELS
from deepsoftlog.embeddings.nn_models import AdditionFunctor, CarryFunctor
from deepsoftlog.experiments.mnist_addition.dataset import mnist_addition_dataset, AdditionDataset, MnistQueryDataset
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

_EXP_ROOT = Path(__file__).parent


def compute_raw_task_accuracy(program, cfg):
    """Raw task accuracy: argmax each digit, perform addition with carry, compare to ground truth."""
    store = program.get_store()
    store.eval()
    device = cfg['device']
    n_digits = cfg['digits']

    lenet = store.functor_embeddings["('lenet5', 1)"]

    test_dataset = mnist_addition_dataset(n_digits, "test")

    correct = 0
    total = 0
    digit_correct = 0
    digit_total = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            mnist_indices = test_dataset.data_indices[i]
            # mnist_indices: ((idx1,...,idxN), (idxN+1,...,idx2N))
            # First chunk = first number (MSB first), second chunk = second number
            all_pred_digits = []
            for chunk in mnist_indices:
                chunk_preds = []
                for idx in chunk:
                    img, label = test_dataset.dataset[idx]
                    img = img.unsqueeze(0).to(device)
                    output = lenet(img)
                    pred = output.argmax().item()
                    chunk_preds.append(pred)
                    digit_correct += (pred == label)
                    digit_total += 1
                all_pred_digits.append(chunk_preds)

            # Digit-wise addition with carry (LSB first)
            d1_lsb = list(reversed(all_pred_digits[0]))
            d2_lsb = list(reversed(all_pred_digits[1]))
            carry = 0
            result_digits = []
            for pos in range(n_digits):
                s = d1_lsb[pos] + d2_lsb[pos] + carry
                result_digits.append(s % 10)
                carry = s // 10
            result_digits.append(carry)

            # Ground truth (LSB first, padded to n_digits+1)
            gt_label = test_dataset._get_label(i)
            gt_str = str(gt_label)
            while len(gt_str) < n_digits + 1:
                gt_str = '0' + gt_str
            gt_lsb = [int(c) for c in reversed(gt_str)]

            if result_digits == gt_lsb:
                correct += 1
            total += 1

    store.train()
    task_acc = correct / total if total > 0 else 0
    digit_acc = digit_correct / digit_total if digit_total > 0 else 0
    return task_acc, digit_acc


def get_pretrain_dataloader(cfg):
    pretrain_dataset = AdditionDataset(1, 2500).randomly_mutate_output()
    return DataLoader(pretrain_dataset, batch_size=cfg['batch_size'])


def get_train_dataloader(cfg):
    train_dataset = mnist_addition_dataset(cfg['digits'], "training")
    train_dataset = train_dataset.random_subset(cfg['data_subset'])
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'])
    return dataloader


def get_test_dataloader(cfg):
    eval_dataset = mnist_addition_dataset(cfg['digits'], "test")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def get_val_dataloader(cfg):
    eval_dataset = mnist_addition_dataset(cfg['digits'], "val").randomly_mutate_output()
    return DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False)


def train(config_name):
    cfg = load_config(config_name)
    if cfg['add_symb']:
        SPECIAL_MODELS[('mod_ten_add', 3)] = AdditionFunctor
        SPECIAL_MODELS[('carry', 3)] = CarryFunctor

    # Training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_dataloader = get_val_dataloader(cfg)
    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, val_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)

    supervised_dataloader = None
    if cfg.get('supervised', False):
        print(">>> FULLY SUPERVISED MODE ACTIVE <<<")
        digit_dataset = MnistQueryDataset("training")
        supervised_dataloader = DataLoader(digit_dataset, batch_size=cfg['batch_size'])

    raw_eval_fn = lambda: compute_raw_task_accuracy(program, cfg)
    trainer = Trainer(program, get_train_dataloader, val_dataloader, test_dataloader, nll_loss, optimizer, logger=logger, supervised_dataloader=supervised_dataloader, raw_eval_fn=raw_eval_fn)
    trainer.train(cfg, nb_workers=cfg['nb_workers'])
    avg_epoch_time = sum(trainer.epoch_times) / len(trainer.epoch_times) if trainer.epoch_times else 0.0

    # Raw task accuracy (consistent with CTSketch/A-NeSI/Scallop)
    task_acc, digit_acc = compute_raw_task_accuracy(program, cfg)

    # Peak GPU memory
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory_mb = -1

    print(f"RESULT: accuracy={task_acc * 100:.4f} violation_rate=N/A peak_memory_mb={peak_memory_mb:.1f} digit_accuracy={digit_acc * 100:.4f} epoch_time_s={avg_epoch_time:.2f}")


def eval(folder: str, digits=None):
    cfg = load_config(f"results/{folder}/config.yaml")
    if digits is not None:
        cfg['digits'] = digits
    if cfg['add_symb']:
        SPECIAL_MODELS[('mod_ten_add', 3)] = AdditionFunctor
        SPECIAL_MODELS[('carry', 3)] = CarryFunctor
    print("EVALING", cfg['name'], cfg['digits'])

    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, test_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)
    trainer = Trainer(program, None, None, None)

    print("Digit accuracy:")
    digit_dataset = MnistQueryDataset("test")
    trainer.eval(DataLoader(digit_dataset, batch_size=4, shuffle=False))
    print("Predicate accuracy:")
    trainer.eval(test_dataloader)


if __name__ == "__main__":
    # eval("1d run", digits=1)
    train("deepsoftlog/experiments/mnist_addition/config.yaml")
