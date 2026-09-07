import os
import random
from typing import *
from tqdm import tqdm
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb

import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
from mnist_config import MNISTSumNet, mnist_sum_loader
from tensor_sketch import TensorSketch
from sketch_configs import full_theta2, full_theta3, full_theta4, sample_theta

EPS = 1e-8

def full_theta(digits, input_dim, output_dim, samples):
  if digits == 4:
    return full_theta4(digits, input_dim, output_dim, samples)
  elif digits == 3:
    return full_theta3(digits, input_dim, output_dim, samples)
  elif digits == 2:
    return full_theta2(digits, input_dim, output_dim, samples)
  else: 
    raise Exception("not implemented")

class Trainer():
  def __init__(self, model, tensor_method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate, save_model=True, supervised=False, digit_loss_weight=1.0):
    self.model_dir = model_dir
    self.all_digit = dims[-1]
    self.network = model(self.all_digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.digits = digits
    self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.tensorsketch = TensorSketch(tensor_method)
    self.tensor_method = tensor_method
    self.t = [None] * len(digits)
    self.digit_acc = 0
    self.best_acc = 0
    self.save_model = save_model
    self.loss_fn = F.l1_loss
    self.supervised = supervised
    self.digit_loss_weight = digit_loss_weight

  def pretrain(self):
    t0 = time()
    err = 0
    self.max_tensor_bytes = 0
    for i, digit_i in enumerate(self.digits):
      gt = full_theta(digit_i, self.dims[i], self.dims[i + 1], 0)
      gt_entries = gt.numel()
      gt_bytes = gt_entries * gt.element_size()
      if gt_bytes > self.max_tensor_bytes:
          self.max_tensor_bytes = gt_bytes
      rerr1, cores1, _ = self.tensorsketch.approx_theta({'gt': gt, 'rank': rank})
      rerr2, cores2, _ = self.tensorsketch.approx_theta({'gt': gt, 'rank': rank})
      if rerr2 is None or rerr1 < rerr2: rerr, cores = rerr1, cores1
      else: rerr, cores = rerr2, cores2
      sketched_entries = sum(c.numel() for c in cores)
      print(f"TENSOR_INFO: layer={i} shape_before={list(gt.shape)} entries_before={gt_entries} entries_after={sketched_entries} core_shapes={[list(c.shape) for c in cores]}")
      self.t[i] = [cores_i.to(device) for cores_i in cores]
      err += rerr
    t1 = time()
    wandb.log({"pretrain": t1 - t0, "rerr": err})
    return rerr

  def sub_product(self, i, *base_inputs):
    if self.tensor_method == 'tt': 
      return self.sub_product_tt(i, *base_inputs)
    elif self.tensor_method == 'cp': 
      return self.sub_product_cp(i, *base_inputs)
    elif self.tensor_method == 'tucker': 
      return self.sub_product_tucker(i, *base_inputs)
    elif self.tensor_method == 'tensor_ring':
      return self.sub_product_tr(i, *base_inputs)
  
  def sub_product_tt(self, i, *base_inputs):
    t = self.t[i]
    output = torch.einsum('ijk, bj -> bk', t[0], base_inputs[0])
    for c1, c2 in zip(t[1:], base_inputs[1:]):
      output = torch.einsum('bi, ika, bk -> ba', output, c1, c2)
    return output

  def sub_product_cp(self, i, *base_inputs):
    t = self.t[i]
    output = torch.einsum('bi, bj, ir, jr -> b', *base_inputs, *t)
    output = output.unsqueeze(-1)
    return output 
  
  def sub_product_tucker(self, i, *base_inputs):
      t = self.t[i]
      output = torch.einsum('ab, ia, jb, ni, nj -> n', *t, *base_inputs)
      output = output.unsqueeze(-1)
      return output
  
  def sub_product_tr(self, i, *base_inputs):
      t = self.t[i]
      output = torch.einsum('ni, nj, aib, bja -> n', *base_inputs, *t)
      output = output.unsqueeze(-1)
      return output

  def program(self, *inputs):
    ps = inputs
    iters = self.all_digit
    batch_size = inputs[0].shape[0]
    
    for i, n_i in enumerate(self.digits):  
      ps2 = []
      iters = iters // n_i
      for j in range(iters):
        output = self.sub_product(i, *tuple(ps[j*n_i : (j+1)*n_i]))
        if i == len(self.digits) - 1: ps2.append(output[:, 0])
        else:
          xs = torch.arange(self.dims[i+1]).repeat(batch_size, 1).to(device)
          sigma = max((1 / ps[j*n_i : (j+1)*n_i][0].max()), 4)
          distr_output = torch.exp(- (xs - output)**2 / sigma)
          distr_output = F.normalize(distr_output, dim=-1, p=1)
          ps2.append(distr_output)
      ps = tuple(ps2)
    output = ps[0]
    return output

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for e, (data, target, digits) in enumerate(iter):
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      output = self.program(*tuple(output_t))
      sum_loss = self.loss_fn(output, target.to(device).float())

      if self.supervised:
        # Digit-level cross-entropy loss (fully supervised mode)
        # output_t is a tuple of [batch, 10] tensors, one per digit
        digit_preds = torch.stack(output_t, dim=1)  # [batch, n_digits, 10]
        digit_targets = digits.to(device)  # [batch, n_digits]
        # Log of softmax outputs for NLL loss (output_t already has softmax applied)
        digit_log_probs = torch.log(digit_preds + EPS)  # [batch, n_digits, 10]
        digit_loss = F.nll_loss(
          digit_log_probs.reshape(-1, 10),
          digit_targets.reshape(-1)
        )
        loss = sum_loss + self.digit_loss_weight * digit_loss
      else:
        loss = sum_loss

      train_loss += loss
      loss.backward()
      self.optimizer.step()
      total_correct += (output.round()==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.val_loader.dataset)
    correct = 0
    sketch_correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    with torch.no_grad():
      iter = tqdm(self.val_loader, total=len(self.val_loader))
      for (data, target, digits) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        sketch_out = self.program(*tuple(output_t))
        sketch_pred = sketch_out.round().long().to(device)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        target = target.to(device)
        correct += (pred == target).sum()
        sketch_correct += (sketch_pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        perc = float(correct/num_items)
        sketch_perc = float(sketch_correct/num_items)
        digit_perc = float(digit_correct / (num_items * self.all_digit))
        iter.set_description(f"[Test Epoch {epoch}] Argmax-Sum: {correct}/{num_items} ({perc * 100:.2f}%) Sketch: {sketch_correct}/{num_items} ({sketch_perc * 100:.2f}%) Digit Accuracy ({digit_perc * 100:.2f}%)")

    return perc, sketch_perc, digit_perc

  def test(self):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    sketch_correct = 0
    digit_correct = 0
    violations = 0
    total = 0
    with torch.no_grad():
      for (data, target, digits) in self.test_loader:
        target = target.to(device)
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        sketch_out = self.program(*tuple(output_t))
        sketch_pred = sketch_out.round().long().to(device)
        correct += (pred == target).sum()
        sketch_correct += (sketch_pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        # Violation: sketch-predicted sum disagrees with sum(argmax digits)
        violations += (sketch_pred != pred).sum().item()
        total += output.shape[1]
      perc = float(correct/num_items)
      sketch_perc = float(sketch_correct/num_items)
      digit_perc = float(digit_correct / (num_items * self.all_digit))
      violation_rate = violations / total * 100 if total > 0 else 0.0
      print(f"[Final Acc] Argmax-Sum {int(correct)}/{int(num_items)} ({perc * 100:.2f}%) Sketch {int(sketch_correct)}/{int(num_items)} ({sketch_perc * 100:.2f}%) Digit Accuracy ({digit_perc * 100:.2f}%)")
      print(f"Violation rate (sketch != argmax-sum): {violation_rate:.2f}%")
    return perc, sketch_perc, digit_perc, violation_rate

  def train(self, n_epochs):
    err = self.pretrain()
    if err != err:  # skip training only on NaN
      print(f"WARNING: pretrain returned NaN error, skipping training")
      n_epochs = 0
    epoch_times = []
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      epoch_times.append(t1 - t0)
      test_acc, test_sketch_acc, test_digit_acc = self.test_epoch(epoch)
      t2 = time()

      if self.save_model and self.best_acc < test_acc:
        self.best_acc = test_acc
        perc, sketch_perc, digit_perc, _ = self.test()

        wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_sketch_acc": test_sketch_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0,
        "final acc": perc,
        "final sketch acc": sketch_perc,
        "final digit acc": digit_perc})
      else:
        wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_sketch_acc": test_sketch_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})
    self.avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0           

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_sum")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--method", type=str, choices=['tt', 'tensor_ring', 'cp', 'tucker'], default='tt')
    parser.add_argument("--loss", type=str, default='l1')  
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--supervised", action="store_true", help="Add digit-level cross-entropy loss (fully supervised)")
    parser.add_argument("--digit-loss-weight", type=float, default=1.0, help="Weight for digit-level loss when --supervised is set")
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    depth = args.depth
    method = args.method
    loss = args.loss
    seed = args.seed
    rank = args.rank

    digits = [2] * depth
    dims = [2 ** (1 + i) for i in range(depth)]

    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    # Data
    data_dir = str(pathlib.Path(__file__).resolve().parents[3] / "data")  # shared repo-root data/
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/tensor_tree{depth}_{seed}"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, dims[-1])

    # setup wandb
    config = vars(args)
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
      project=f"sketch_tree{depth}",
      name=f"{seed}",
      config=config,
      mode=wandb_mode)
    print(config)
    if args.supervised:
      print(f">>> FULLY SUPERVISED MODE ACTIVE (digit_loss_weight={args.digit_loss_weight}) <<<")
    else:
      print(">>> WEAKLY SUPERVISED MODE (sum-only) <<<")

    # Create trainer and train
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(MNISTSumNet, method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate, supervised=args.supervised, digit_loss_weight=args.digit_loss_weight)
    trainer.train(n_epochs)

    # Final evaluation with violation rate
    perc, sketch_perc, digit_perc, violation_rate = trainer.test()

    # Peak GPU memory
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory_mb = -1

    print(f"RESULT: accuracy={perc * 100:.4f} sketch_accuracy={sketch_perc * 100:.4f} violation_rate={violation_rate:.4f} peak_memory_mb={peak_memory_mb:.1f} digit_accuracy={digit_perc * 100:.4f} epoch_time_s={trainer.avg_epoch_time:.2f} matrix_size_mb={trainer.max_tensor_bytes / 1024 / 1024:.4f}")
    run.finish()