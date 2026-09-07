import os
import random
import resource
from typing import *
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import wandb
from create_scallop_sum import CreateMNISTScallopSum

import scallopy
from mnist_config import mnist_sum_loader, MNISTSumNet

class MNISTSumNNet(nn.Module):
  def __init__(self, provenance, k, digit, dispatch):
    super(MNISTSumNNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTSumNet(digit)
    self.digit = digit

    # Scallop Context
    scallop_context_creator = CreateMNISTScallopSum(digit, provenance, k)
    self.scl_ctx = scallop_context_creator.add_sum_rule()

    # The `sum_n` logical reasoning module
    self.sum_n = self.scl_ctx.forward_function(f"sum_{self.digit}", output_mapping=[(i,) for i in range(digit*9 + 1)], dispatch=dispatch)

  def forward(self, x):
    # First recognize the digits
    distrs = self.mnist_net(x)

    # Call the sum-n reasoning module
    args = {f"digit_{str(i+1)}" : distrs[i] for i in range(self.digit)}
    output = self.sum_n(**args)

    return output, distrs

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
  return F.binary_cross_entropy(output, gt)

def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

EPS = 1e-8

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance, digits, dispatch, supervised=False, digit_loss_weight=1.0, save_model=False):
    self.model_dir = model_dir
    self.network = MNISTSumNNet(provenance, k, digits, dispatch).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.best_loss = None
    self.save_model = save_model
    self.digits = digits
    self.supervised = supervised
    self.digit_loss_weight = digit_loss_weight

    if loss == "nll": self.loss = nll_loss
    elif loss == "bce": self.loss = bce_loss
    else: raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    train_loss = 0
    for (data, target, digits) in iter:
      self.optimizer.zero_grad()
      data = tuple([data_i.to(device) for data_i in data])
      target = target.to(device)
      output, distrs = self.network(data)
      loss = self.loss(output, target)

      if self.supervised:
        digit_preds = torch.stack(distrs, dim=1)  # [batch, n_digits, 10]
        digit_targets = digits.to(device)  # [batch, n_digits]
        digit_log_probs = torch.log(digit_preds + EPS)
        digit_loss = F.nll_loss(digit_log_probs.reshape(-1, 10), digit_targets.reshape(-1))
        loss = loss + self.digit_loss_weight * digit_loss

      train_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    digit_correct = 0
    digit_total = 0
    violations = 0
    violation_total = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, digits) in iter:
        data = tuple([data_i.to(device) for data_i in data])
        target = target.to(device)
        output, distrs = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = float(correct/num_items)

        # Digit-level accuracy
        digit_preds = torch.stack(distrs, dim=1)  # [batch, n_digits, 10]
        digit_pred_labels = digit_preds.argmax(dim=-1)  # [batch, n_digits]
        digit_targets = digits.to(device)
        digit_correct += (digit_pred_labels == digit_targets).sum().item()
        digit_total += digit_targets.numel()

        # Consistency: argmax of scallop sum distribution vs sum of argmax digits
        argmax_digit_sum = digit_pred_labels.sum(dim=1).to(pred.device)
        violations += (pred.view(-1) != argmax_digit_sum).sum().item()
        violation_total += pred.numel()

        digit_perc = digit_correct / digit_total if digit_total > 0 else 0.0
        iter.set_description(f"[Test Epoch {epoch}] Loss: {test_loss:.4f}, Acc: {perc * 100:.2f}%, Digit: {digit_perc * 100:.2f}%")

      if self.save_model and self.best_acc < perc:
        self.best_loss = test_loss
        self.best_acc = perc
        torch.save(self.network, os.path.join(model_dir, f"sum{digits}_best.pt"))

    digit_perc = digit_correct / digit_total if digit_total > 0 else 0.0
    violation_rate = violations / violation_total * 100 if violation_total > 0 else 0.0
    return perc, digit_perc, test_loss, violation_rate

  def train(self, n_epochs):
    epoch_times = []
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_digit_acc, test_loss, violation_rate = self.test_epoch(epoch)
      t2 = time()

      epoch_time = t2 - t0
      epoch_times.append(epoch_time)
      print(f"Epoch {epoch}/{n_epochs}: test_acc={test_acc * 100:.2f}% digit_acc={test_digit_acc * 100:.2f}% violation_rate={violation_rate:.2f}% train_loss={train_loss:.4f} epoch_time={epoch_time:.2f}s")

      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_digit_acc": test_digit_acc,
        "test_loss": test_loss,
        "violation_rate": violation_rate,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": epoch_time})

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    if torch.cuda.is_available():
      peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
      peak_gpu_mb = -1
    # Peak CPU (RSS) memory via getrusage (ru_maxrss is in KB on Linux)
    peak_cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"RESULT: accuracy={test_acc * 100:.4f} violation_rate={violation_rate:.4f} peak_gpu_mb={peak_gpu_mb:.1f} peak_cpu_mb={peak_cpu_mb:.1f} digit_accuracy={test_digit_acc * 100:.4f} epoch_time_s={avg_epoch_time:.2f}")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--digits", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=0.0005)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--dispatch", type=str, default="parallel")
  parser.add_argument("--supervised", action="store_true", help="Add digit-level cross-entropy loss (fully supervised)")
  parser.add_argument("--digit-loss-weight", type=float, default=1.0, help="Weight for digit-level loss when --supervised is set")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  digits = args.digits
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  import pathlib
  data_dir = str(pathlib.Path(__file__).resolve().parents[3] / "data")  # shared repo-root data/
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scallop_sum"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digits)
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance, digits, args.dispatch, supervised=args.supervised, digit_loss_weight=args.digit_loss_weight)
  print(len(train_loader.dataset))
  print(len(test_loader.dataset))

  # setup wandb
  config = vars(args)
  wandb.init(
    project=f"scallop_sum",
    name = f"{digits}_{args.seed}",
    config=config)
  print(config)
  
  trainer.train(n_epochs)