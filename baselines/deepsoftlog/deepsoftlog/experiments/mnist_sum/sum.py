from pathlib import Path
import torch
from tqdm import tqdm

from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.embeddings.initialize_vector import SPECIAL_MODELS
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

from deepsoftlog.embeddings.nn_models import SumFunctor, SumFunctorAll
from deepsoftlog.experiments.mnist_sum.dataset import mnist_sum_dataset, generate_data, SumDataset, MnistQueryDataset, MnistDataset

_EXP_ROOT = Path(__file__).parent


def compute_raw_task_accuracy(program, cfg):
    """Raw task accuracy: argmax each digit prediction, sum them, compare to ground truth."""
    store = program.get_store()
    store.eval()
    device = cfg['device']
    n_digits = cfg['digits']

    lenet = store.functor_embeddings["('lenet5', 1)"]

    test_dataset = mnist_sum_dataset(n_digits, "test")

    correct = 0
    total = 0
    digit_correct = 0
    digit_total = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            mnist_indices = test_dataset.data_indices[i]
            pred_sum = 0
            for chunk in mnist_indices:
                for idx in chunk:
                    img, label = test_dataset.dataset[idx]
                    img = img.unsqueeze(0).to(device)
                    output = lenet(img)
                    pred = output.argmax().item()
                    pred_sum += pred
                    digit_correct += (pred == label)
                    digit_total += 1

            gt_sum = test_dataset._get_label(i)
            if pred_sum == gt_sum:
                correct += 1
            total += 1

    store.train()
    task_acc = correct / total if total > 0 else 0
    digit_acc = digit_correct / digit_total if digit_total > 0 else 0
    return task_acc, digit_acc


def get_pretrain_dataloader(cfg):
    pretrain_dataset = SumDataset(1, 2500).randomly_mutate_output()
    return DataLoader(pretrain_dataset, batch_size=cfg['batch_size'])


def get_train_dataloader(cfg):
    train_dataset = mnist_sum_dataset(cfg['digits'], "training")
    train_dataset = train_dataset.random_subset(cfg['data_subset'])
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'])
    return dataloader


def get_test_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "test")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)

def get_validation_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "validation")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def get_val_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "val").randomly_mutate_output()
    return DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False)


def train(config_name):
    cfg = load_config(config_name)
    if cfg['sum_symb']:
        if cfg['digits'] == 2:
            SPECIAL_MODELS[('plus', 2)] = SumFunctor
        elif cfg['digits'] == 4:
            SPECIAL_MODELS[('plus4', 4)] = SumFunctor
        elif cfg['digits'] == 16:
            SPECIAL_MODELS[('plus16', 16)] = SumFunctorAll
        elif cfg['digits'] == 64:
            SPECIAL_MODELS[('plus64', 64)] = SumFunctorAll
        elif cfg['digits'] == 256:
            SPECIAL_MODELS[('plus256', 256)] = SumFunctorAll
        elif cfg['digits'] == 1024:
            SPECIAL_MODELS[('plus1024', 1024)] = SumFunctorAll

    # Training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    generate_data(cfg['digits'])
    val_dataloader = get_val_dataloader(cfg)
    vlidation_dataloader = get_validation_dataloader(cfg)
    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, val_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)

    # Supervised digit-level training dataloader
    supervised_dataloader = None
    if cfg.get('supervised', False):
        print(">>> FULLY SUPERVISED MODE ACTIVE <<<")
        digit_dataset = MnistQueryDataset("training")
        supervised_dataloader = DataLoader(digit_dataset, batch_size=cfg['batch_size'])

    raw_eval_fn = lambda: compute_raw_task_accuracy(program, cfg)
    trainer = Trainer(program, get_train_dataloader, vlidation_dataloader, test_dataloader, nll_loss, optimizer, logger=logger, supervised_dataloader=supervised_dataloader, raw_eval_fn=raw_eval_fn)
    trainer.train(cfg, nb_workers=cfg['nb_workers'])
    avg_epoch_time = sum(trainer.epoch_times) / len(trainer.epoch_times) if trainer.epoch_times else 0.0
    max_param_bytes = max(p.numel() * p.element_size() for p in program.get_store().parameters())
    matrix_size_mb = max_param_bytes / 1024 / 1024

    # Raw task accuracy (consistent with CTSketch/A-NeSI/Scallop)
    task_acc, digit_acc = compute_raw_task_accuracy(program, cfg)

    # Peak GPU memory
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory_mb = -1

    print(f"RESULT: accuracy={task_acc * 100:.4f} violation_rate=N/A peak_memory_mb={peak_memory_mb:.1f} digit_accuracy={digit_acc * 100:.4f} epoch_time_s={avg_epoch_time:.2f} matrix_size_mb={matrix_size_mb:.4f}")
    return task_acc


def eval(folder: str, digits=None):
    cfg = load_config(f"results/{folder}/config.yaml")
    if digits is not None:
        cfg['digits'] = digits
    if cfg['sum_symb']:
        if cfg['digits'] == 2:
            SPECIAL_MODELS[('plus', 2)] = SumFunctor
        elif cfg['digits'] == 4:
            SPECIAL_MODELS[('plus', 2)] = SumFunctor
            SPECIAL_MODELS[('plus2', 2)] = SumFunctor
        elif cfg['digits'] == 16:
            SPECIAL_MODELS[('plus16', 16)] = SumFunctorAll
        elif cfg['digits'] == 64:
            SPECIAL_MODELS[('plus64', 64)] = SumFunctorAll
        elif cfg['digits'] == 256:
            SPECIAL_MODELS[('plus256', 256)] = SumFunctorAll
        elif cfg['digits'] == 1024:
            SPECIAL_MODELS[('plus1024', 1024)] = SumFunctorAll
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
    train("deepsoftlog/experiments/mnist_sum/config.yaml")
