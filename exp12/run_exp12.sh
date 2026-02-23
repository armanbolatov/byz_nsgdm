#!/bin/bash
# Exp 12: Shakespeare LM

cd "$(dirname "$0")"

N_WORKERS=20
F_BYZANTINE=3
LRS="0.5 0.05 0.01 0.1"
MOMENTUM=0.9
EPOCHS=100
MAX_BATCHES=100
BATCH_SIZE=64

ATTACKS="BF mimic ALIE"
AGGS="rfa krum cm"
SEEDS="0 42 123"

MAX_JOBS=15

total=0
for attack in $ATTACKS; do
    for agg in $AGGS; do
        for opt in baseline lr_decay nsgdm; do
            for lr in $LRS; do
                for seed in $SEEDS; do
                    total=$((total + 1))
                done
            done
        done
    done
done
echo "Total experiments: $total (max $MAX_JOBS in parallel)"

for attack in $ATTACKS; do
    for agg in $AGGS; do
        for opt in baseline lr_decay nsgdm; do
            for lr in $LRS; do
                for seed in $SEEDS; do
                    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                        sleep 1
                    done
                    
                    echo "[Starting] attack=$attack, agg=$agg, opt=$opt, lr=$lr, seed=$seed"
                    
                    opt_flags=""
                    if [ "$opt" = "nsgdm" ]; then
                        opt_flags="--byz-nsgdm"
                    elif [ "$opt" = "lr_decay" ]; then
                        opt_flags="--lr-decay"
                    fi
                    
                    python run_exp12.py \
                        -n $N_WORKERS \
                        -f $F_BYZANTINE \
                        --attack $attack \
                        --agg $agg \
                        --lr $lr \
                        --momentum $MOMENTUM \
                        --epochs $EPOCHS \
                        --max-batches $MAX_BATCHES \
                        --seed $seed \
                        --batch-size $BATCH_SIZE \
                        --noniid \
                        --nnm \
                        --use-cuda \
                        $opt_flags &
                done
            done
        done
    done
done

wait
echo "Done. To plot: python run_exp12.py -n $N_WORKERS -f $F_BYZANTINE --plot"
