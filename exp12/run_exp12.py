"""Exp 12: Byzantine-robust LM training on Shakespeare."""

import argparse
import os
import sys
import math
import numpy as np
import torch
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_small_gpt
from dataset import CharDataset, NonIIDSampler, IIDSampler
from worker import LMWorker, LMBitFlippingWorker, LMMimicWorker, LMALIEWorker

from codes.server import TorchServer
from codes.simulator import ParallelTrainer
from codes.aggregator.rfa import RFA
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.krum import Krum
from codes.utils import initialize_logger


def get_args():
    parser = argparse.ArgumentParser(description="Exp 12: Shakespeare LM")
    
    # Basic
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    
    # Experiment
    parser.add_argument("-n", type=int, default=20, help="Number of workers")
    parser.add_argument("-f", type=int, default=3, help="Number of Byzantine workers")
    parser.add_argument("--attack", type=str, default="BF", choices=["BF", "mimic", "ALIE"])
    parser.add_argument("--agg", type=str, default="rfa", choices=["rfa", "krum", "cm"])
    parser.add_argument("--noniid", action="store_true", default=True)
    parser.add_argument("--nnm", action="store_true", default=True)
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--byz-nsgdm", action="store_true", default=False)
    parser.add_argument("--lr-decay", action="store_true", default=False)
    
    # Model/Data
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--plot", action="store_true", default=False)
    
    args = parser.parse_args()
    return args


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "outputs")


def get_aggregator(args):
    if args.agg == "rfa":
        agg = RFA(T=8)
    elif args.agg == "krum":
        agg = Krum(n=args.n, f=args.f, m=1)
    elif args.agg == "cm":
        agg = CM()
    else:
        raise ValueError(f"Unknown aggregator: {args.agg}")
    
    if args.nnm:
        agg = nnm_wrapper(args, agg)
    
    return agg


def nnm_wrapper(args, aggregator):
    def aggr(inputs):
        n = len(inputs)
        f = args.f
        
        if not isinstance(inputs[0], torch.Tensor):
            inputs = [torch.tensor(inp) for inp in inputs]
        
        stacked = torch.stack(inputs)
        mixed = []
        
        for i in range(n):
            xi = stacked[i]
            distances = torch.norm(stacked - xi.unsqueeze(0), dim=1)
            _, sorted_idx = torch.sort(distances)
            neighbors = sorted_idx[:n - f]
            mixed.append(torch.mean(stacked[neighbors], dim=0))
        
        return aggregator(mixed)
    
    return aggr


class ByzNSGDMOptimizer:
    def __init__(self, base_optimizer, lr0):
        self.base_optimizer = base_optimizer
        self.lr0 = lr0
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        current_lr = self.lr0 / (self.step_count ** 0.5)
        
        total_grad_norm = 0.0
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 0:
            for group in self.base_optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.data = p.grad.data / total_grad_norm * current_lr
        
        original_lr = self.base_optimizer.param_groups[0]['lr']
        for group in self.base_optimizer.param_groups:
            group['lr'] = 1.0
        
        self.base_optimizer.step()
        
        for group in self.base_optimizer.param_groups:
            group['lr'] = original_lr
            
    def zero_grad(self):
        self.base_optimizer.zero_grad()
        
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


class LRDecayOptimizer:
    def __init__(self, base_optimizer, lr0):
        self.base_optimizer = base_optimizer
        self.lr0 = lr0
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        current_lr = self.lr0 / (self.step_count ** 0.5)
        
        for group in self.base_optimizer.param_groups:
            group['lr'] = current_lr
        
        self.base_optimizer.step()
            
    def zero_grad(self):
        self.base_optimizer.zero_grad()
        
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


def perplexity_metric(logits, targets):
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1), 
        ignore_index=-1
    )
    loss_val = min(loss.item(), 20.0)
    return math.exp(loss_val)


def create_worker(args, trainer, rank, model, optimizer, data_loader, device):
    n_good = args.n - args.f
    
    if rank < n_good:
        return LMWorker(
            momentum=args.momentum,
            data_loader=data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
        )
    
    # Byzantine worker
    if args.attack == "BF":
        return LMBitFlippingWorker(
            momentum=args.momentum,
            data_loader=data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
        )
    elif args.attack == "mimic":
        worker = LMMimicWorker(
            warmup=50,
            momentum=args.momentum,
            data_loader=data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        worker.configure(trainer)
        return worker
    elif args.attack == "ALIE":
        worker = LMALIEWorker(
            n=args.n,
            m=args.f,
            momentum=args.momentum,
            data_loader=data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        worker.configure(trainer)
        return worker
    
    raise ValueError(f"Unknown attack: {args.attack}")


def get_sampler_callback(args, rank):
    """Get sampler callback for a worker."""
    n_good = args.n - args.f
    
    if args.noniid:
        return lambda ds: NonIIDSampler(
            dataset=ds,
            num_replicas=n_good,
            rank=rank % n_good,
            shuffle=True,
        )
    else:
        return lambda ds: IIDSampler(
            dataset=ds,
            num_replicas=n_good,
            rank=rank % n_good,
            shuffle=True,
        )


def main(args):
    # Setup
    opt_tag = "nsgdm" if args.byz_nsgdm else ("lr_decay" if args.lr_decay else "baseline")
    log_dir = os.path.join(
        OUT_DIR,
        f"n{args.n}_f{args.f}",
        f"{args.agg}_{args.attack}_lr{args.lr}_mom{args.momentum}_seed{args.seed}_{opt_tag}"
    )
    
    # Check if already done
    stats_path = os.path.join(log_dir, "stats")
    if os.path.exists(stats_path) and os.path.getsize(stats_path) > 0:
        print(f"Skipping: Results exist at {stats_path}")
        return
    
    initialize_logger(log_dir)
    json_logger = logging.getLogger("stats")
    
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset first to get vocab size
    train_dataset = CharDataset(
        data_dir=args.data_dir,
        split="train",
        block_size=args.block_size,
        fraction=args.fraction,
    )
    vocab_size = train_dataset.vocab_size
    
    # Model (smaller for char-level: ~1M params)
    model = create_small_gpt(vocab_size=vocab_size, block_size=args.block_size).to(device)
    
    # Optimizer
    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.byz_nsgdm:
        server_opt = ByzNSGDMOptimizer(base_opt, args.lr)
        print("Using Byz-NSGDM optimizer")
    elif args.lr_decay:
        server_opt = LRDecayOptimizer(base_opt, args.lr)
        print("Using LR decay optimizer")
    else:
        server_opt = base_opt
        print("Using baseline optimizer")
    
    # Server and trainer
    server = TorchServer(optimizer=server_opt)
    trainer = ParallelTrainer(
        server=server,
        aggregator=get_aggregator(args),
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batches,
        log_interval=args.log_interval,
        metrics={"perplexity": perplexity_metric},
        use_cuda=args.use_cuda,
        debug=args.debug,
    )
    
    # Create workers
    for rank in range(args.n):
        sampler = get_sampler_callback(args, rank)(train_dataset)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
        )
        worker_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        worker = create_worker(args, trainer, rank, model, worker_opt, data_loader, device)
        trainer.add_worker(worker)
    
    # Validation loader
    val_dataset = CharDataset(
        data_dir=args.data_dir,
        split="validation",
        block_size=args.block_size,
        fraction=1.0,  # Use full validation set
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        drop_last=True,
    )
    
    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, {args.max_batches} batches/epoch")
    nan_detected = False
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= 50:  # Limit validation batches
                    break
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        # Check for NaN loss
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"Epoch {epoch}: NaN/Inf loss detected. Stopping run.")
            json_logger.info({
                "_meta": {"type": "error"},
                "E": epoch,
                "error": "NaN/Inf loss",
            })
            nan_detected = True
            break
        
        val_ppl = math.exp(val_loss)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
        json_logger.info({
            "_meta": {"type": "validation"},
            "E": epoch,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        })
        
        model.train()
        trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
    
    if nan_detected:
        print(f"\nTraining stopped early due to NaN loss.")
    else:
        print(f"\nTraining complete. Results saved to {log_dir}")


if __name__ == "__main__":
    args = get_args()
    
    if args.plot:
        # Plotting mode
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import re
        
        results = []
        base_dir = os.path.join(OUT_DIR, f"n{args.n}_f{args.f}")
        
        if os.path.exists(base_dir):
            for name in os.listdir(base_dir):
                path = os.path.join(base_dir, name)
                if not os.path.isdir(path):
                    continue
                
                # Parse directory name: rfa_BF_lr0.1_mom0.9_seed0_nsgdm
                m = re.match(r"^(rfa|krum|cm)_(BF|mimic|ALIE)_lr([\d\.]+)_mom([\d\.]+)_seed(\d+)_(baseline|lr_decay|nsgdm)$", name)
                if not m:
                    continue
                
                agg, attack, lr_txt, mom_txt, seed_txt, opt_tag = m.groups()
                lr_val = float(lr_txt)
                stats_file = os.path.join(path, "stats")
                
                if not os.path.exists(stats_file):
                    continue
                
                try:
                    with open(stats_file) as f:
                        for line in f:
                            # Parse Python dict format (not JSON)
                            import ast
                            line = line.strip()
                            if not line:
                                continue
                            # Handle np.float64 by replacing with float
                            line = line.replace("np.float64(", "(").replace("np.float32(", "(")
                            try:
                                entry = ast.literal_eval(line)
                            except:
                                continue
                            if entry.get("_meta", {}).get("type") == "validation":
                                opt_label = "Byz-NSGDM" if opt_tag == "nsgdm" else ("Baseline-Decay" if opt_tag == "lr_decay" else "Baseline")
                                results.append({
                                    "Epoch": entry["E"],
                                    "Val Loss": entry["val_loss"],
                                    "Val PPL": entry["val_ppl"],
                                    "Attack": attack,
                                    "Aggregator": agg.upper(),
                                    "Optimizer": opt_label,
                                    "LR": lr_val,
                                    "seed": int(seed_txt),
                                })
                except Exception as e:
                    print(f"Error loading {stats_file}: {e}")
        
        if results:
            df = pd.DataFrame(results)
            print(f"Loaded {len(df)} data points")
            
            # Select best LR per (Attack, Aggregator, Optimizer) by lowest final perplexity
            final_idx = df.groupby(["Attack", "Aggregator", "Optimizer", "LR", "seed"])["Epoch"].transform("max")
            final_df = df[df["Epoch"] == final_idx]
            lr_perf = final_df.groupby(["Attack", "Aggregator", "Optimizer", "LR"])["Val PPL"].mean().reset_index()
            best_rows = lr_perf.sort_values(["Attack", "Aggregator", "Optimizer", "Val PPL"], ascending=[True, True, True, True])\
                                .drop_duplicates(subset=["Attack", "Aggregator", "Optimizer"], keep="first")
            best_lr_map = {(r.Attack, r.Aggregator, r.Optimizer): r.LR for r in best_rows.itertuples()}
            
            # Create plots
            out_dir = os.path.join(OUT_DIR, "plots")
            os.makedirs(out_dir, exist_ok=True)
            
            attacks = ["BF", "mimic", "ALIE"]
            aggregators = ["RFA", "KRUM", "CM"]
            colors = {"Baseline": "#1f77b4", "Baseline-Decay": "#2ca02c", "Byz-NSGDM": "#ff7f0e"}
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            
            for i, attack in enumerate(attacks):
                for j, agg in enumerate(aggregators):
                    ax = axes[i, j]
                    subset = df[(df["Attack"] == attack) & (df["Aggregator"] == agg)]
                    
                    if len(subset) > 0:
                        for opt in ["Baseline", "Baseline-Decay", "Byz-NSGDM"]:
                            key = (attack, agg, opt)
                            if key not in best_lr_map:
                                continue
                            best_lr = best_lr_map[key]
                            opt_data = subset[(subset["Optimizer"] == opt) & (subset["LR"] == best_lr)]
                            if len(opt_data) > 0:
                                grouped = opt_data.groupby("Epoch")["Val PPL"].agg(["mean", "std"]).reset_index()
                                label_lr = f"{best_lr:.4g}"
                                ax.plot(grouped["Epoch"], grouped["mean"], color=colors[opt], 
                                       label=f"{opt} (lr={label_lr})", linewidth=2)
                                ax.fill_between(grouped["Epoch"], grouped["mean"] - grouped["std"], 
                                              grouped["mean"] + grouped["std"], color=colors[opt], alpha=0.2)
                    else:
                        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center", va="center", fontsize=12, alpha=0.5)
                    
                    ax.set_title(f"{attack} + {agg}", fontsize=10, fontweight='bold')
                    ax.set_yscale('log')
                    if i == 2:
                        ax.set_xlabel("Epoch")
                    if j == 0:
                        ax.set_ylabel("Val Perplexity (log)")
                    ax.legend(loc="upper right", fontsize=8)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "exp12_perplexity.pdf"), bbox_inches="tight", dpi=300)
            fig.savefig(os.path.join(out_dir, "exp12_perplexity.png"), bbox_inches="tight", dpi=300)
            print(f"Plots saved to {out_dir}")
        else:
            print("No results found.")
    else:
        main(args)
