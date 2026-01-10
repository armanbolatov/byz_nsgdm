"""Exp 10: Ablation study for momentum and LR sensitivity."""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp10_ablation/"

if args.identifier and args.identifier != "ablation":
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"

# Directory name includes momentum, learning rate, and seed
mom_str = f"mom{args.momentum}".replace(".", "p")
lr_str = f"lr{args.lr}".replace(".", "p")
LOG_DIR += f"{args.agg}_{args.attack}_{mom_str}_{lr_str}_seed{args.seed}_nsgdm"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 10
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 20

if not args.plot:
    import os
    stats_path = os.path.join(LOG_DIR, "stats")
    if os.path.exists(stats_path) and os.path.isfile(stats_path) and os.path.getsize(stats_path) > 0:
        print(f"Skipping: Results already exist at {stats_path}")
    else:
        main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    font = {"size": 12}
    plt.rc("font", **font)

    import re
    dir_entries = []
    try:
        for name in os.listdir(INP_DIR):
            full_path = os.path.join(INP_DIR, name)
            if not os.path.isdir(full_path):
                continue
            m = re.match(r"^(rfa|krum|cm)_(BF|LF|mimic)_mom([\dp\.]+)_lr([\dp\.]+)_seed(\d+)_nsgdm$", name)
            if not m:
                continue
            agg, attack, mom_txt, lr_txt, seed_txt = m.groups()
            mom_val = float(mom_txt.replace("p", "."))
            lr_val = float(lr_txt.replace("p", "."))
            dir_entries.append({
                "name": name,
                "agg": agg,
                "attack": attack,
                "momentum": mom_val,
                "lr": lr_val,
                "seed": int(seed_txt),
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
                        "Momentum": entry["momentum"],
                        "LR": entry["lr"],
                        "seed": entry["seed"],
                    }
                )
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    results = pd.DataFrame(results)
    print(f"Loaded {len(results)} data points")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp10_ablation_results.csv", index=None)

    if len(results) > 0:
        df = results.copy()
        final_idx = df.groupby(["Momentum", "LR", "seed"])["Iterations"].transform("max")
        final_df = df[df["Iterations"] == final_idx]
        agg_final = final_df.groupby(["Momentum", "LR"])["Accuracy (%)"].agg(["mean", "std"]).reset_index()
        agg_final.columns = ["Momentum", "LR", "Final Accuracy Mean", "Final Accuracy Std"]
        agg_final.to_csv(OUT_DIR + "exp10_ablation_summary.csv", index=None)
        print("\n=== Ablation Study Summary (Final Accuracy %) ===")
        pivot_table = agg_final.pivot(index="Momentum", columns="LR", values="Final Accuracy Mean")
        print(pivot_table.round(2).to_string())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        ax0 = axes[0]
        pivot_mean = agg_final.pivot(index="Momentum", columns="LR", values="Final Accuracy Mean")
        sns.heatmap(pivot_mean, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax0, 
                    cbar_kws={'label': 'Accuracy (%)'})
        ax0.set_xlabel("Learning Rate")
        ax0.set_ylabel("Momentum")
        ax0.set_title("(a) Final Accuracy Heatmap")
        
        lr_perf = agg_final.sort_values(["Momentum", "Final Accuracy Mean"], ascending=[True, False])
        best_lr_per_mom = lr_perf.drop_duplicates(subset=["Momentum"], keep="first")
        best_lr_map = {row.Momentum: row.LR for row in best_lr_per_mom.itertuples()}
        
        ax1 = axes[1]
        momenta = sorted(df["Momentum"].unique())
        colors = plt.cm.viridis([i / len(momenta) for i in range(len(momenta))])
        
        for mom, color in zip(momenta, colors):
            if mom not in best_lr_map:
                continue
            best_lr = best_lr_map[mom]
            subset = df[(df["Momentum"] == mom) & (df["LR"] == best_lr)]
            grouped = subset.groupby("Iterations")["Accuracy (%)"].agg(["mean", "std"]).reset_index()
            ax1.plot(grouped["Iterations"], grouped["mean"], color=color, linewidth=2,
                     label=f"β={mom:.2f}")
            ax1.fill_between(grouped["Iterations"], grouped["mean"] - grouped["std"], 
                            grouped["mean"] + grouped["std"], color=color, alpha=0.2)
        
        ax1.set_xlim(0, MAX_BATCHES_PER_EPOCH * EPOCHS)
        ax1.set_ylim(40, 100)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("(b) Accuracy vs Iterations")
        ax1.legend(loc="lower right", fontsize=8)
        
        ax2 = axes[2]
        lrs = sorted(df["LR"].unique())
        colors_lr = plt.cm.plasma([i / len(lrs) for i in range(len(lrs))])
        
        for lr, color in zip(lrs, colors_lr):
            subset = agg_final[agg_final["LR"] == lr].sort_values("Momentum")
            ax2.errorbar(subset["Momentum"], subset["Final Accuracy Mean"], 
                        yerr=subset["Final Accuracy Std"], color=color, 
                        marker='o', linewidth=2, capsize=3, label=f"lr={lr:.3g}")
        
        ax2.set_xlabel("Momentum (β)")
        ax2.set_ylabel("Final Accuracy (%)")
        ax2.set_title("(c) Final Accuracy vs Momentum")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(40, 100)
        
        plt.tight_layout()
        fig.savefig(OUT_DIR + "exp10_ablation.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(OUT_DIR + "exp10_ablation.png", bbox_inches="tight", dpi=300)

        print(f"\nResults saved to {OUT_DIR}")
    else:
        print("No results found. Make sure to run the experiments first.")
