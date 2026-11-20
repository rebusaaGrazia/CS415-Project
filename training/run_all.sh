
#!/bin/bash
set -e

for seed in 42 123 2026; do

    # Baseline
    python training/train_experiment.py \
        --attention none \
        --seed "$seed" \
        --max-epochs 5 \
        --patience 4

    # SE attention
    for placement in early middle late; do

        # SE Early seed 42 is already complete.
        if [ "$seed" = "42" ] && [ "$placement" = "early" ]; then
            echo "Skipping completed se_early_seed42"
            continue
        fi

        python training/train_experiment.py \
            --attention se \
            --placement "$placement" \
            --seed "$seed" \
            --max-epochs 5 \
            --patience 4

    done

    # CBAM attention
    for placement in early middle late; do

        python training/train_experiment.py \
            --attention cbam \
            --placement "$placement" \
            --seed "$seed" \
            --max-epochs 5 \
            --patience 4

    done

done

echo "All experiments completed!"
