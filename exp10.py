r"""Exp 10: Accuracy Comparison - BF, LF, mimic attacks vs rfa, krum,    results = []
    for agg, attack, seed, optimizer in exp_grid():
        opt_suffix = "_nsgdm" if optimizer == "byz_nsgdm" else ""
        grid_identifier = f"{agg}_{attack}_s2_seed{seed}{opt_suffix}" aggregators (with NNM)

- Fix:
    - n=20, f=3
    - Number of iterations = 1500 (50 epochs * 30 batches)
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 3
    - LR = 0.01 (configurable via --lr argument)
    - NNM: True (Nearest Neighbor Mixing pre-aggregation)
    - momentum: 0.9

- Compare:
    - ATK = BF, LF, mimic
    - AGG = rfa, krum, cm (coordinatewise median)
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp10/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"

# Directory name includes optimizer, preprocessing method, and learning rate
opt_suffix = "_nsgdm" if getattr(args, 'byz_nsgdm', False) else "_baseline"
preprocess_suffix = "nnm" if args.nnm else f"s{args.bucketing}"
lr_str = f"lr{args.lr}".replace(".", "p")
LOG_DIR += f"{args.agg}_{args.attack}_{preprocess_suffix}_{lr_str}_seed{args.seed}{opt_suffix}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 10
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 50

if not args.plot:
    main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    # Set up plot styling
    font = {"size": 12}
    plt.rc("font", **font)

    # Scan output folders and parse metadata from names
    import re
    dir_entries = []
    try:
        for name in os.listdir(INP_DIR):
            full_path = os.path.join(INP_DIR, name)
            if not os.path.isdir(full_path):
                continue
            m = re.match(r"^(rfa|krum|cm)_(BF|LF|mimic)_(nnm|s\d+)_lr([\dp\.]+)_seed(\d+)_(baseline|nsgdm)$", name)
            if not m:
                continue
            agg, attack, preprocess, lr_txt, seed_txt, opt_tag = m.groups()
            want_pre = "nnm" if args.nnm else f"s{args.bucketing}"
            if preprocess != want_pre:
                continue
            lr_val = float(lr_txt.replace("p", "."))
            dir_entries.append({
                "name": name,
                "agg": agg,
                "attack": attack,
                "preprocess": preprocess,
                "lr_text": lr_txt,
                "lr": lr_val,
                "seed": int(seed_txt),
                "optimizer_tag": opt_tag,
                "path": os.path.join(full_path, "stats"),
            })
    except FileNotFoundError:
        pass

    results = []
    for entry in dir_entries:
        path = entry["path"]
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "Attack": entry["attack"],
                        "Aggregator": entry["agg"].upper(),
                        "Optimizer": "Byz-NSGDM" if entry["optimizer_tag"] == "nsgdm" else "Baseline",
                        "seed": entry["seed"],
                        "LR": entry["lr"],
                    }
                )
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    results = pd.DataFrame(results)
    print(f"Loaded {len(results)} data points")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Save raw results
    results.to_csv(OUT_DIR + "exp10_results.csv", index=None)

    if len(results) > 0:
        # Select best LR per (Attack, Aggregator, Optimizer) by highest final accuracy
        import numpy as np
        df = results.copy()
        final_idx = df.groupby(["Attack", "Aggregator", "Optimizer", "LR", "seed"])["Iterations"].transform("max")
        final_df = df[df["Iterations"] == final_idx]
        lr_perf = final_df.groupby(["Attack", "Aggregator", "Optimizer", "LR"])["Accuracy (%)"].mean().reset_index()
        best_rows = lr_perf.sort_values(["Attack", "Aggregator", "Optimizer", "Accuracy (%)"], ascending=[True, True, True, False])\
                            .drop_duplicates(subset=["Attack", "Aggregator", "Optimizer"], keep="first")
        best_lr_map = {(r.Attack, r.Aggregator, r.Optimizer): r.LR for r in best_rows.itertuples()}
        
        # 3x3 plot: one subplot per attack-aggregator combination
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("Byz-NSGDM vs Baseline (best LR per setup)", fontsize=16, y=0.98)
        
        attacks = ["BF", "LF", "mimic"]
        aggregators = ["RFA", "KRUM", "CM"]
        colors = {"Baseline": "#1f77b4", "Byz-NSGDM": "#ff7f0e"}
        
        for i, attack in enumerate(attacks):
            for j, agg in enumerate(aggregators):
                ax = axes[i, j]
                subset = df[(df["Attack"] == attack) & (df["Aggregator"] == agg.upper())]
                if len(subset) > 0:
                    for optimizer in ["Baseline", "Byz-NSGDM"]:
                        opt_sub = subset[subset["Optimizer"] == optimizer]
                        key = (attack, agg.upper(), optimizer)
                        if key not in best_lr_map:
                            continue
                        best_lr = best_lr_map[key]
                        opt_lr_sub = opt_sub[opt_sub["LR"] == best_lr]
                        if len(opt_lr_sub) == 0:
                            continue
                        grouped = opt_lr_sub.groupby("Iterations")["Accuracy (%)"].agg(["mean", "std"]).reset_index()
                        label_lr = f"{best_lr:.4g}"
                        ax.plot(grouped["Iterations"], grouped["mean"], color=colors[optimizer], linewidth=2,
                                label=f"{optimizer} (lr={label_lr})")
                        ax.fill_between(grouped["Iterations"], grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"],
                                        color=colors[optimizer], alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center", va="center", fontsize=12, alpha=0.5)
                ax.set_xlim(0, MAX_BATCHES_PER_EPOCH * EPOCHS)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{attack} + {agg}", fontsize=10, fontweight='bold')
                if i == 2:
                    ax.set_xlabel("Iterations")
                if j == 0:
                    ax.set_ylabel("Accuracy (%)")
                ax.legend(loc="lower right", fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        fig.savefig(OUT_DIR + "exp10_accuracy_comparison.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(OUT_DIR + "exp10_accuracy_comparison.png", bbox_inches="tight", dpi=300)

        print(f"\nResults saved to {OUT_DIR}")
    else:
        print("No results found. Make sure to run the experiments first.")