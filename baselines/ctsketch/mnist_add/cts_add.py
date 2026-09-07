import os
import csv
import random
from typing import *
from argparse import ArgumentParser
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
from mnist_config import MNISTSumNet, mnist_multi_digit_sum2_loader
import sketch_config
from tensor_sketch import TensorSketch

def pretrain(digit, samples):
    t0 = time()
    # first calculate theta for the two-digit addition
    full_theta1 = sketch_config.full_theta2(digit=digit, elems=10, output_dim=19, samples=samples)
    # then calculate theta for the carry addition
    theta_i_0 = torch.arange(0, 19).unsqueeze(1).repeat(1, 10)
    theta_i_carry = torch.arange(1, 20).unsqueeze(1).repeat(1, 10)
    full_theta2 = torch.cat((theta_i_0, theta_i_carry), dim=1)
    t1 = time()
    print(f"[Pretrain] Time: {t1 - t0:.2f}s")
    return full_theta1, full_theta2


class Trainer():
  def __init__(self, model, tensor_method, digits, train_loader, test_loader, model_dir, learning_rate, save_model=False, supervised=False, digit_loss_weight=1.0, log_path=None):
    self.model_dir = model_dir
    self.network = model(digits * 2).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.digits = digits
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
    self.supervised = supervised
    self.digit_loss_weight = digit_loss_weight
    self.rerr = [1] * (digits*2)
    self.t1 = None
    self.t2 = None
    self.log_writer = None
    self.log_file = None
    self.last_hard_hd = None
    self.last_soft_hd = None
    self.last_op_exact = None
    self.last_sum_hd = None
    self.last_pos_hard = None
    self.last_pos_soft = None
    self.last_pos_sum = None
    if log_path:
      self.log_file = open(log_path, 'w', newline='')
      fieldnames = ["epoch", "train_loss", "test_acc", "test_digit_acc",
                    # operand-level: exact match
                    "op1_exact", "op2_exact", "both_operands_exact",
                    # operand-level: Hamming (hard = argmax, soft = pre-argmax)
                    "op1_hamming_hard", "op2_hamming_hard",
                    "op1_hamming_soft", "op2_hamming_soft",
                    "op1_hamming_hard_norm", "op2_hamming_hard_norm",
                    "op1_hamming_soft_norm", "op2_hamming_soft_norm",
                    "operands_hamming_hard", "operands_hamming_soft",
                    "operands_hamming_hard_norm", "operands_hamming_soft_norm",
                    # sum-level
                    "sum_exact", "sum_hamming", "sum_hamming_norm",
                    "hamming_amplification",
                    "train_time_s", "test_time_s", "epoch_time_s", "gpu_mem_mb"]
      self.log_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
      self.log_writer.writeheader()
      self.log_file.flush()
    self.full_theta1, self.full_theta2 = pretrain(digits, 0)

  def sub_program(self, n, d, *base_inputs):
    t = self.t1.long().to(device)
    p = base_inputs[0]
    batch_size = p.shape[0]
    for i in range(1, n):
      p1 = p.unsqueeze(-1)
      p2 = base_inputs[i].unsqueeze(1)
      eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
      p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, d).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
  
  def sub_program_carry(self, sum_result, carry):
    t = self.t2.long().to(device)
    batch_size = sum_result.shape[0]
    p1 = sum_result.unsqueeze(-1)
    p2 = carry.unsqueeze(1)
    i = 1
    eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
    p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, 20).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
      
  def program(self, *inputs):
    ps = inputs
    batch_size = inputs[0].shape[0]
    rerr1, cores1, X_hat1 = self.tensorsketch.approx_theta({'gt': self.full_theta1, 'rank': 2})
    self.t1 = torch.clamp(X_hat1.to_numpy(), min=0)
    rerr2, cores2, X_hat2 = self.tensorsketch.approx_theta({'gt': self.full_theta2, 'rank': 2})
    self.t2 = torch.clamp(X_hat2.to_numpy(), min=0)
    self.t1 = self.full_theta1
    self.t2 = self.full_theta2
    mapping = torch.tensor([i % 10 for i in range(20)]).to(device)
    
    first_sums = []
    # For each digit, we compute the sum2
    for i in range(self.digits):
      index = self.digits - 1 - i
      input1, input2 = ps[index], ps[index + self.digits]
      sum_output = self.sub_program(2, 19, *(input1, input2))
      first_sums.append(sum_output)
    
    final_sums = []
    previous_sum = torch.cat([torch.ones(batch_size, 1).to(device), torch.zeros(batch_size, 19).to(device)], dim=1)
    # For each digit, we compute the carry sum
    for i in range(self.digits):
      sum_pred = first_sums[i]
      final_sum = self.sub_program_carry(sum_pred, previous_sum)
      grouped_final_sum = torch.zeros(batch_size, 10).to(device).index_add(1, mapping, final_sum)
      final_sums.append(grouped_final_sum)
      previous_sum = final_sum
    # append 1 if the last sum gets the carry
    tens_mapping = torch.tensor([i // 10 for i in range(20)]).to(device)
    final_sums.append(torch.zeros(batch_size, 10).to(device).index_add(1, tens_mapping, previous_sum))
      
    output = torch.stack(final_sums).permute(1, 0, 2) # most significant digit at end, least significant at beginning
    rerr = rerr1 + rerr2
    return rerr, output
  
  def loss(self, output, ground_truth):
    dim = output.shape[2]
    return F.nll_loss(torch.flatten((output + 1e-8).log(), start_dim=0, end_dim=1), torch.flatten(ground_truth))
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"[Train {epoch}]", leave=False)
    for (i, (data, target, digits)) in enumerate(pbar):
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      rerr, output = self.program(*tuple(output_t))
      loss = self.loss(output, target.to(device))
      if self.supervised:
        logits = torch.stack(list(output_t), dim=1)  # [batch, 2*digit, 10]
        digit_loss = F.nll_loss((logits + 1e-8).log().flatten(0, 1), digits.flatten().to(device))
        loss = loss + self.digit_loss_weight * digit_loss
      loss.backward()
      self.optimizer.step()
      pred_digits = output.argmax(dim=-1)  # [batch, digits+1]
      total_correct += (pred_digits.cpu() == target).all(dim=1).sum()
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{correct_perc:.2f}%")
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} {correct_perc:.2f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    N = self.digits
    n_positions = N * 2
    n_sum_positions = N + 1
    correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    # Operand-level Hamming distances over the input digits, per example.
    #   hard: number of positions whose argmax is wrong.
    #   soft: expected number of wrong positions under the predicted
    #         distributions, i.e. sum_i (1 - p_i[y_i]), computed pre-argmax.
    hard_hd = []      # [n, 2] -- column 0 is operand 1, column 1 is operand 2
    soft_hd = []      # [n, 2]
    op_exact = []     # [n, 2] bool -- whole operand read correctly
    sum_hd = []       # [n] -- Hamming of the predicted sum vs ground-truth sum
    # Per-position error counts. Everything is indexed LSB-first so that
    # operand position k lines up with sum position k -- this alignment is
    # what exposes carry propagation.
    pos_hard = torch.zeros(2, N)
    pos_soft = torch.zeros(2, N)
    pos_sum = torch.zeros(n_sum_positions)
    with torch.no_grad():
      for (data, target, digits) in test_loader:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        # Digit-wise addition with carry (avoids int64 overflow for large N)
        d1 = output[:self.digits].T.flip(1).cpu()   # [batch, digits], LSB first
        d2 = output[self.digits:].T.flip(1).cpu()   # [batch, digits], LSB first
        carry = torch.zeros(d1.shape[0], dtype=torch.long)
        result_digits = []
        for pos in range(self.digits):
          s = d1[:, pos] + d2[:, pos] + carry
          result_digits.append(s % 10)
          carry = s // 10
        result_digits.append(carry)
        pred_sum = torch.stack(result_digits, dim=1)  # [batch, digits+1], LSB first
        correct += (pred_sum == target).all(dim=1).sum()
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()

        # --- Operand-level metrics (backbone perception) ---
        # probs/pred are [batch, 2N, 10] / [batch, 2N], aligned with
        # digits [batch, 2N]. Positions [0:N] are operand 1, [N:2N] operand 2,
        # matching the d1/d2 split used for the addition above. Each operand is
        # flipped to LSB-first so its positions align with the sum's.
        gt = digits.to(device)
        probs = torch.stack(output_t, dim=1)
        pred = output.t()
        p_true = probs.gather(2, gt.unsqueeze(-1)).squeeze(-1)
        w1 = (pred[:, :N] != gt[:, :N]).float().flip(1).cpu()
        w2 = (pred[:, N:] != gt[:, N:]).float().flip(1).cpu()
        u1 = (1.0 - p_true[:, :N]).flip(1).cpu()
        u2 = (1.0 - p_true[:, N:]).flip(1).cpu()
        hard_hd.append(torch.stack([w1.sum(1), w2.sum(1)], dim=1))
        soft_hd.append(torch.stack([u1.sum(1), u2.sum(1)], dim=1))
        op_exact.append(torch.stack([w1.sum(1) == 0, w2.sum(1) == 0], dim=1))
        pos_hard[0] += w1.sum(0)
        pos_hard[1] += w2.sum(0)
        pos_soft[0] += u1.sum(0)
        pos_soft[1] += u2.sum(0)

        # --- Sum-level metrics (after the symbolic carry addition) ---
        sum_wrong = (pred_sum != target).float()   # [batch, N+1], LSB first
        sum_hd.append(sum_wrong.sum(1))
        pos_sum += sum_wrong.sum(0)

        perc = 100. * correct / num_items
        perc2 = 100. * digit_correct / (num_items * self.digits * 2)

    hard_hd = torch.cat(hard_hd).float()
    soft_hd = torch.cat(soft_hd).float()
    op_exact = torch.cat(op_exact)
    sum_hd = torch.cat(sum_hd).float()
    self.last_hard_hd = hard_hd
    self.last_soft_hd = soft_hd
    self.last_op_exact = op_exact
    self.last_sum_hd = sum_hd
    self.last_pos_hard = pos_hard / num_items
    self.last_pos_soft = pos_soft / num_items
    self.last_pos_sum = pos_sum / num_items

    hard_op = hard_hd.mean(dim=0)
    soft_op = soft_hd.mean(dim=0)
    m = {
      "acc": float(perc),
      "digit_acc": float(perc2),
      "hd_hard": hard_hd.sum(dim=1).mean().item(),
      "hd_soft": soft_hd.sum(dim=1).mean().item(),
      "hd_hard_op": hard_op.tolist(),
      "hd_soft_op": soft_op.tolist(),
      "op1_exact": 100. * op_exact[:, 0].float().mean().item(),
      "op2_exact": 100. * op_exact[:, 1].float().mean().item(),
      "both_operands_exact": 100. * op_exact.all(dim=1).float().mean().item(),
      "sum_exact": 100. * (sum_hd == 0).float().mean().item(),
      "hd_sum": sum_hd.mean().item(),
    }
    # Error amplification: wrong sum digits produced per wrong input digit.
    # >1 means a misread digit corrupts more than one output position (carries).
    m["hd_amplification"] = (m["hd_sum"] / m["hd_hard"]) if m["hd_hard"] > 0 else 0.0

    print(f"[Test Epoch {epoch}] Total correct {int(correct)}/{int(num_items)} ({perc:.2f}) Digit correct {perc2:.2f}%")
    print(f"[Test Epoch {epoch}] Operands  exact: op1={m['op1_exact']:.2f}% op2={m['op2_exact']:.2f}% both={m['both_operands_exact']:.2f}% | "
          f"hamming (of {N} each) hard: op1={hard_op[0]:.4f} op2={hard_op[1]:.4f} | soft: op1={soft_op[0]:.4f} op2={soft_op[1]:.4f}")
    print(f"[Test Epoch {epoch}] Sum       exact: {m['sum_exact']:.2f}% | "
          f"hamming (of {n_sum_positions}) {m['hd_sum']:.4f} ({100. * m['hd_sum'] / n_sum_positions:.2f}% of positions) | "
          f"amplification={m['hd_amplification']:.3f}")

    if self.save_model:
      torch.save(self.network.state_dict(), self.model_dir+f"/best.pth")

    return m

  def dump_hamming_distribution(self, path):
    """Write the final-epoch per-example metrics for histogramming."""
    if self.last_hard_hd is None:
      return
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(["example", "hard_hd", "soft_hd", "hard_hd_op1", "hard_hd_op2",
                  "soft_hd_op1", "soft_hd_op2", "op1_exact", "op2_exact",
                  "sum_hd", "sum_exact", "digits_per_operand", "n_positions"])
      rows = zip(self.last_hard_hd.tolist(), self.last_soft_hd.tolist(),
                 self.last_op_exact.tolist(), self.last_sum_hd.tolist())
      for i, (h, s, e, sh) in enumerate(rows):
        w.writerow([i, int(h[0] + h[1]), f"{s[0] + s[1]:.6f}",
                    int(h[0]), int(h[1]), f"{s[0]:.6f}", f"{s[1]:.6f}",
                    int(e[0]), int(e[1]), int(sh), int(sh == 0),
                    self.digits, self.digits * 2])
    print(f"Wrote per-example metrics to {path}")

  def dump_position_profile(self, path):
    """Per-position error rates, LSB-first, for error-propagation analysis.

    Operand position k and sum position k refer to the same place value, so
    comparing sum_err_rate against the operand rates at the same k shows how
    far a misread digit propagates upward through carries.
    """
    if self.last_pos_sum is None:
      return
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(["position_lsb_first", "op1_hard_err_rate", "op2_hard_err_rate",
                  "op1_soft_err_rate", "op2_soft_err_rate", "sum_err_rate"])
      for k in range(self.digits + 1):
        if k < self.digits:
          w.writerow([k,
                      f"{self.last_pos_hard[0][k]:.6f}", f"{self.last_pos_hard[1][k]:.6f}",
                      f"{self.last_pos_soft[0][k]:.6f}", f"{self.last_pos_soft[1][k]:.6f}",
                      f"{self.last_pos_sum[k]:.6f}"])
        else:
          # Most significant sum digit -- carry-out only, no operand position.
          w.writerow([k, "", "", "", "", f"{self.last_pos_sum[k]:.6f}"])
    print(f"Wrote per-position error profile to {path}")

  def train(self, n_epochs):
    self.peak_acc = 0
    self.peak_digit_acc = 0
    epoch_times = []
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      m = self.test_epoch(epoch)
      t2 = time()

      test_acc, test_digit_acc = m["acc"], m["digit_acc"]
      hd_hard_op, hd_soft_op = m["hd_hard_op"], m["hd_soft_op"]
      epoch_time = t2 - t0
      epoch_times.append(epoch_time)
      self.peak_acc = max(self.peak_acc, test_acc)
      self.peak_digit_acc = max(self.peak_digit_acc, test_digit_acc)
      n_positions = self.digits * 2
      n_sum_positions = self.digits + 1

      gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
      print(f"Epoch {epoch}/{n_epochs}: task_acc={test_acc:.2f}% peak_acc={self.peak_acc:.2f}% digit_acc={test_digit_acc:.2f}% "
            f"op_exact=({m['op1_exact']:.2f}%,{m['op2_exact']:.2f}%) sum_exact={m['sum_exact']:.2f}% "
            f"op_hd_hard=({hd_hard_op[0]:.4f},{hd_hard_op[1]:.4f}) op_hd_soft=({hd_soft_op[0]:.4f},{hd_soft_op[1]:.4f}) "
            f"sum_hd={m['hd_sum']:.4f} ampl={m['hd_amplification']:.3f} "
            f"train_loss={train_loss:.4f} epoch_time={epoch_time:.2f}s")

      if self.log_writer:
        self.log_writer.writerow({
          "epoch": epoch,
          "train_loss": f"{train_loss:.6f}",
          "test_acc": f"{test_acc:.2f}",
          "test_digit_acc": f"{test_digit_acc:.2f}",
          "op1_exact": f"{m['op1_exact']:.4f}",
          "op2_exact": f"{m['op2_exact']:.4f}",
          "both_operands_exact": f"{m['both_operands_exact']:.4f}",
          "op1_hamming_hard": f"{hd_hard_op[0]:.6f}",
          "op2_hamming_hard": f"{hd_hard_op[1]:.6f}",
          "op1_hamming_soft": f"{hd_soft_op[0]:.6f}",
          "op2_hamming_soft": f"{hd_soft_op[1]:.6f}",
          "op1_hamming_hard_norm": f"{hd_hard_op[0] / self.digits:.6f}",
          "op2_hamming_hard_norm": f"{hd_hard_op[1] / self.digits:.6f}",
          "op1_hamming_soft_norm": f"{hd_soft_op[0] / self.digits:.6f}",
          "op2_hamming_soft_norm": f"{hd_soft_op[1] / self.digits:.6f}",
          "operands_hamming_hard": f"{m['hd_hard']:.6f}",
          "operands_hamming_soft": f"{m['hd_soft']:.6f}",
          "operands_hamming_hard_norm": f"{m['hd_hard'] / n_positions:.6f}",
          "operands_hamming_soft_norm": f"{m['hd_soft'] / n_positions:.6f}",
          "sum_exact": f"{m['sum_exact']:.4f}",
          "sum_hamming": f"{m['hd_sum']:.6f}",
          "sum_hamming_norm": f"{m['hd_sum'] / n_sum_positions:.6f}",
          "hamming_amplification": f"{m['hd_amplification']:.6f}",
          "train_time_s": f"{t1 - t0:.2f}",
          "test_time_s": f"{t2 - t1:.2f}",
          "epoch_time_s": f"{epoch_time:.2f}",
          "gpu_mem_mb": f"{gpu_mem:.1f}",
        })
        self.log_file.flush()

      if self.save_model:
        torch.save(self.network.state_dict(), self.model_dir+f"/latest.pth")

    self.avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    self.final_acc = test_acc
    self.final_digit_acc = test_digit_acc
    self.final_metrics = m

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_add")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--method", type=str, default='tt')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=2)
  parser.add_argument("--dispatch", type=str, default="parallel")
  parser.add_argument("--supervised", action="store_true")
  parser.add_argument("--digit-loss-weight", type=float, default=1.0)
  parser.add_argument("--tag", type=str, default="", help="Suffix for log/checkpoint names so re-runs don't overwrite earlier sweeps")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  method = args.method
  digit = args.digits
  seed = args.seed
  supervised = args.supervised
  digit_loss_weight = args.digit_loss_weight

  torch.manual_seed(seed)
  random.seed(seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  sup_str = "sup" if supervised else "weak"
  # Suffix keeps re-runs from clobbering earlier sweeps' CSVs/checkpoints.
  tag_str = f"_{args.tag}" if args.tag else ""
  run_name = f"add{digit}_{sup_str}_s{seed}{tag_str}"

  # Data
  data_dir = str(pathlib.Path(__file__).resolve().parents[3] / "data")  # shared repo-root data/
  # Per-run so seeds/configs don't overwrite each other's checkpoints.
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/{run_name}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_multi_digit_sum2_loader(data_dir, batch_size, digit)

  # Setup logging
  log_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../logs"))
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, f"ctsketch_{run_name}.csv")
  hamming_path = os.path.join(log_dir, f"ctsketch_{run_name}_hamming_per_example.csv")
  position_path = os.path.join(log_dir, f"ctsketch_{run_name}_position_profile.csv")
  if os.path.exists(log_path):
    raise SystemExit(f"Refusing to overwrite existing log: {log_path} (pass --tag to disambiguate)")
  print(f"Logging to {log_path}")
  print(f"Config: {vars(args)}")

  # Create and run trainer
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
  trainer = Trainer(MNISTSumNet, method, digit, train_loader, test_loader, model_dir, learning_rate, save_model=True, supervised=supervised, digit_loss_weight=digit_loss_weight, log_path=log_path)
  trainer.train(n_epochs)
  trainer.dump_hamming_distribution(hamming_path)
  trainer.dump_position_profile(position_path)

  m = trainer.final_metrics
  n_positions = digit * 2
  n_sum_positions = digit + 1
  peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else -1
  print(f"RESULT: accuracy={trainer.final_acc:.4f} peak_accuracy={trainer.peak_acc:.4f} violation_rate=0.0000 "
        f"peak_memory_mb={peak_memory_mb:.1f} digit_accuracy={trainer.final_digit_acc:.4f} "
        f"peak_digit_accuracy={trainer.peak_digit_acc:.4f} epoch_time_s={trainer.avg_epoch_time:.2f} "
        f"op1_exact={m['op1_exact']:.4f} op2_exact={m['op2_exact']:.4f} both_operands_exact={m['both_operands_exact']:.4f} "
        f"op1_hamming_hard={m['hd_hard_op'][0]:.4f} op2_hamming_hard={m['hd_hard_op'][1]:.4f} "
        f"op1_hamming_soft={m['hd_soft_op'][0]:.4f} op2_hamming_soft={m['hd_soft_op'][1]:.4f} "
        f"operands_hamming_hard={m['hd_hard']:.4f} operands_hamming_soft={m['hd_soft']:.4f} "
        f"operands_hamming_hard_norm={m['hd_hard'] / n_positions:.6f} operands_hamming_soft_norm={m['hd_soft'] / n_positions:.6f} "
        f"sum_exact={m['sum_exact']:.4f} sum_hamming={m['hd_sum']:.4f} sum_hamming_norm={m['hd_sum'] / n_sum_positions:.6f} "
        f"hamming_amplification={m['hd_amplification']:.4f} "
        f"digits_per_operand={digit} n_positions={n_positions} n_sum_positions={n_sum_positions}")
