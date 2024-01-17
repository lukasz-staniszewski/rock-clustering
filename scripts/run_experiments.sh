#!/bin/bash

theta_values=(0.65 0.7 0.73 0.8 0.9)
approx_fn_values=("rational_sub" "rational_add" "rational_exp" "rational_sin")
k_values=(2 5 10 15 20)

dataset="congressional"
split_train=0.57
orig_theta=0.73
orig_k=2
orig_approx_fn="rational_sub"

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$CWD

for theta in "${theta_values[@]}"; do
    python3 rock/runner.py --dataset "$dataset" --theta "$theta" --k "$orig_k" --approx_fn "$orig_approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done

for approx_fn in "${approx_fn_values[@]}"; do
    python3 rock/runner.py --dataset "$dataset" --theta "$orig_theta" --k "$orig_k" --approx_fn "$approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done

for k in "${k_values[@]}"; do
    python3 rock/runner.py --dataset "$dataset" --theta "$orig_theta" --k "$k" --approx_fn "$orig_approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done

dataset="mushroom"
split_train=0.25
orig_theta=0.8
orig_k=2
orig_approx_fn="rational_add"

for theta in "${theta_values[@]}"; do
    echo "rock/runner.py --dataset $dataset --theta $theta --k $orig_k --approx_fn $orig_approx_fn --split_train $split_train --skip_outliers --calculate_metrics"

    python3 rock/runner.py --dataset "$dataset" --theta "$theta" --k "$orig_k" --approx_fn "$orig_approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done

for approx_fn in "${approx_fn_values[@]}"; do
    python3 rock/runner.py --dataset "$dataset" --theta "$orig_theta" --k "$orig_k" --approx_fn "$approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done

for k in "${k_values[@]}"; do
    python3 rock/runner.py --dataset "$dataset" --theta "$orig_theta" --k "$k" --approx_fn "$orig_approx_fn" --split_train "$split_train" --skip_outliers --calculate_metrics
done
