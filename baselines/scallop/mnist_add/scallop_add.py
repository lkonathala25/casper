import os
import random
import resource
import argparse
import time

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import wandb
import scallopy

from add_config import addition, MNIST_Net

EPS = 1e-8


class ScallopAddNNet(nn.Module):

    def __init__(self, N, provenance, k, dispatch):
        super(ScallopAddNNet, self).__init__()
        self.N = N
        self.provenance = provenance
        self.k = k
        self.dispatch = dispatch
        self.mnist_net = MNIST_Net()
        self.scallop_ctx_init()

    def scallop_ctx_init(self):
        self.output_dim = self.N * 99 + 1
        self.scl_ctx = scallopy.ScallopContext(provenance=self.provenance, k=self.k)
        for i in range(self.N * 2):
            self.scl_ctx.add_relation(f"digit_{i+1}", int, input_mapping=list(range(10)))
        # Build addition rule: e == (d1*10^(N-1) + ... + dN) + (dN+1*10^(N-1) + ... + d2N)
        terms1 = [f"d{i+1} * {10**(self.N-1-i)}" for i in range(self.N)]
        terms2 = [f"d{self.N+i+1} * {10**(self.N-1-i)}" for i in range(self.N)]
        digit_preds = ", ".join([f"digit_{i+1}(d{i+1})" for i in range(self.N * 2)])
        rule = f"sum(e) :- {digit_preds}, e == {' + '.join(terms1)} + {' + '.join(terms2)}"
        print(rule)
        self.scl_ctx.add_rule(rule)
        self.add_fn = self.scl_ctx.forward_function(
            "sum", output_mapping=[(i,) for i in range(self.output_dim)], dispatch=self.dispatch)

    def forward(self, x):
        logits = self.mnist_net(x)  # [batch, 2*N, 10]
        args = {f"digit_{i+1}": logits[:, i] for i in range(self.N * 2)}
        output = self.add_fn(**args)

        # Decompose output into per-digit distributions
        batch_size = output.shape[0]
        xs = torch.arange(self.output_dim).to(output.device)
        ts = []
        for i in range(self.N + 1):
            inds = (torch.floor_divide(xs, 10**i) % 10).repeat(batch_size, 1)
            if i == self.N:
                ts.append(torch.zeros(batch_size, 2).to(output.device).scatter_add_(1, inds, output))
            else:
                ts.append(torch.zeros(batch_size, 10).to(output.device).scatter_add_(1, inds, output))
        return ts, logits


class Trainer():
    def __init__(self, train_loader, test_loader, config, device):
        self.N = config["N"]
        self.device = device
        self.network = ScallopAddNNet(
            self.N, config["provenance"], config["k"], config["dispatch"]).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config["learning_rate"])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.supervised = config.get("supervised", False)
        self.digit_loss_weight = config.get("digit_loss_weight", 1.0)

    def train_epoch(self, epoch):
        self.network.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc=f"[Train {epoch}]", leave=False)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            numb1, numb2, target, label_digits = batch
            x = torch.cat([numb1.to(self.device), numb2.to(self.device)], dim=1)
            ts, logits = self.network(x)

            # Task loss: per-digit BCE on decomposed output
            rs = []
            for j in range(self.N + 1):
                t_i = torch.floor_divide(target.to(self.device), 10**j) % 10
                if j == self.N:
                    l = F.one_hot(t_i, num_classes=2).float()
                else:
                    l = F.one_hot(t_i, num_classes=10).float()
                r = F.binary_cross_entropy(ts[j], l)
                rs.append(r)
            loss = torch.stack(rs).mean()

            # Supervised digit-level loss
            if self.supervised:
                digit_targets = torch.stack(label_digits[0] + label_digits[1], dim=-1).to(self.device)  # [batch, 2*N]
                digit_log_probs = torch.log(logits + EPS)
                digit_loss = F.nll_loss(
                    digit_log_probs.reshape(-1, 10), digit_targets.reshape(-1))
                loss = loss + self.digit_loss_weight * digit_loss

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return train_loss

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        correct = 0
        digit_correct = 0
        digit_total = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[Test {epoch}]", leave=False):
                numb1, numb2, target, label_digits = batch
                x = torch.cat([numb1.to(self.device), numb2.to(self.device)], dim=1)
                _, logits = self.network(x)

                # Raw task accuracy: argmax digits, compute addition, compare
                output = logits.argmax(dim=-1)  # [batch, 2*N]
                N = self.N
                n1 = torch.stack([10 ** (N - 1 - i) * output[:, i] for i in range(N)], -1).sum(-1)
                n2 = torch.stack([10 ** (N - 1 - i) * output[:, N + i] for i in range(N)], -1).sum(-1)
                pred = n1 + n2
                correct += (pred == target.to(self.device)).sum().item()

                # Digit accuracy
                digit_targets = torch.stack(label_digits[0] + label_digits[1], dim=-1).to(self.device)
                digit_correct += (output == digit_targets).sum().item()
                digit_total += digit_targets.numel()

        task_acc = correct / num_items if num_items > 0 else 0
        digit_acc = digit_correct / digit_total if digit_total > 0 else 0
        return task_acc, digit_acc

    def train(self, n_epochs):
        self.peak_acc = 0
        self.peak_digit_acc = 0
        epoch_times = []
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            t1 = time.time()
            test_acc, test_digit_acc = self.test_epoch(epoch)
            t2 = time.time()

            epoch_time = t2 - t0
            epoch_times.append(epoch_time)
            self.peak_acc = max(self.peak_acc, test_acc)
            self.peak_digit_acc = max(self.peak_digit_acc, test_digit_acc)

            print(f"Epoch {epoch}/{n_epochs}: task_acc={test_acc*100:.2f}% peak_acc={self.peak_acc*100:.2f}% "
                  f"digit_acc={test_digit_acc*100:.2f}% train_loss={train_loss:.4f} epoch_time={epoch_time:.2f}s")

        self.avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        self.final_acc = test_acc
        self.final_digit_acc = test_digit_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser("scallop_add")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dispatch", type=str, default="parallel")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--digit-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    config = vars(args)

    wandb_mode = os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        project=f"scallop_add",
        name=f"add{args.N}_{args.seed}",
        config=config,
        mode=wandb_mode)
    print(config)
    if args.supervised:
        print(f">>> FULLY SUPERVISED MODE ACTIVE (digit_loss_weight={args.digit_loss_weight}) <<<")

    train_set = addition(args.N, "train")
    test_set = addition(args.N, "test")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(train_loader, test_loader, config, device)
    trainer.train(args.n_epochs)

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else -1
    peak_cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"RESULT: accuracy={trainer.final_acc*100:.4f} peak_accuracy={trainer.peak_acc*100:.4f} "
          f"violation_rate=N/A peak_gpu_mb={peak_gpu_mb:.1f} peak_cpu_mb={peak_cpu_mb:.1f} "
          f"digit_accuracy={trainer.final_digit_acc*100:.4f} peak_digit_accuracy={trainer.peak_digit_acc*100:.4f} "
          f"epoch_time_s={trainer.avg_epoch_time:.2f}")
    run.finish()
