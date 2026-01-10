#!/bin/bash
# Ablation: momentum/LR sensitivity

N_WORKERS=20
F_BYZANTINE=3
ATTACK="BF"
AGG="rfa"
SEEDS="0 42 256"
MOMENTA="0.0 0.3 0.5 0.7 0.9 0.95 0.99"
LRS="0.001 0.005 0.01 0.05 0.1 0.5"

MAX_JOBS=10

total=0
for mom in $MOMENTA; do
    for lr in $LRS; do
        for seed in $SEEDS; do
            total=$((total + 1))
        done
    done
done
echo "Total experiments to run: $total (max $MAX_JOBS in parallel)"

for mom in $MOMENTA; do
    for lr in $LRS; do
        for seed in $SEEDS; do
            # Wait if we have MAX_JOBS running
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done
            echo "[Starting] momentum=$mom, lr=$lr, seed=$seed"
            python exp10_ablation.py \
                -n $N_WORKERS \
                -f $F_BYZANTINE \
                --attack $ATTACK \
                --agg $AGG \
                --nnm \
                --noniid \
                --momentum $mom \
                --lr $lr \
                --seed $seed \
                --byz-nsgdm \
                --use-cuda \
                --identifier "ablation" &
        done
    done
done

# Wait for all background jobs to finish
wait
echo "Done. To plot: python exp10_ablation.py -n $N_WORKERS -f $F_BYZANTINE --noniid --nnm --agg $AGG --attack $ATTACK --plot --identifier ablation"
