#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_hyperparams.py

Runs the two TODOs from HW6:
1. Grid-search learning-rate + epochs on distilbert-base-uncased.
2. Evaluate multiple models with the best hyper-params → bar plot.
"""

import argparse
import itertools
from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from base_classification import pre_process
from base_classification import train
from base_classification import evaluate_model

Result = namedtuple("Result", ["lr", "epochs", "dev_acc", "test_acc", "model_name"])

def run_one_config(model_name, lr, epochs, batch_size, device, small_subset=False, save_plots=False):
    """Train + evaluate a single (model,lr,epoch) triple."""
    try:
        model, train_dl, val_dl, test_dl = pre_process(
            model_name, batch_size, device, small_subset
        )

        # train (disable plots if we're doing grid search)
        train(
            model,
            num_epochs=epochs,
            train_dataloader=train_dl,
            validation_dataloader=val_dl,
            test_dataloder=test_dl,
            device=device,
            lr=lr,
            small_subset=small_subset,
        )

        dev = evaluate_model(model, val_dl, device)["accuracy"]
        test = evaluate_model(model, test_dl, device)["accuracy"]
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return Result(lr, epochs, dev, test, model_name)
    except Exception as e:
        print(f"Error training {model_name} with lr={lr}, epochs={epochs}: {e}")
        return Result(lr, epochs, 0.0, 0.0, model_name)


def grid_search_distilbert(args):
    """TODO 1 – grid over lr & epochs for distilbert-base-uncased."""
    lrs = [1e-4, 5e-4, 1e-3]
    epochs_list = [7, 9]
    results = []

    print("\n=== GRID SEARCH (distilbert-base-uncased) ===")
    print(f"Testing {len(lrs)} learning rates × {len(epochs_list)} epochs = {len(lrs) * len(epochs_list)} configurations")
    
    for lr, epochs in itertools.product(lrs, epochs_list):
        print(f"\n--- Config: lr={lr:.0e} | epochs={epochs} ---")
        res = run_one_config(
            model_name="distilbert-base-uncased",
            lr=lr,
            epochs=epochs,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            small_subset=args.small_subset,
        )
        results.append(res)
        print(f"  → dev acc = {res.dev_acc:.4f} | test acc = {res.test_acc:.4f}")

    # pick the best on dev
    best = max(results, key=lambda r: r.dev_acc)
    print("\n" + "="*60)
    print(">>> BEST MODEL ON DEV SET <<<")
    print("="*60)
    print(f"Model: distilbert-base-uncased")
    print(f"Learning Rate: {best.lr:.0e}")
    print(f"Epochs: {best.epochs}")
    print(f"Dev Accuracy: {best.dev_acc:.4f}")
    print(f"Test Accuracy: {best.test_acc:.4f}")
    print("="*60)
    
    # Print all results for reference
    print("\nAll grid search results:")
    for res in results:
        print(f"  lr={res.lr:.0e}, epochs={res.epochs}: dev={res.dev_acc:.4f}, test={res.test_acc:.4f}")
    
    return best, results


def evaluate_extra_models(best_lr, best_epochs, args):
    """TODO 2 – evaluate multiple models with best hyper-params + bar plot."""
    # All models to try (some may fail due to memory constraints)
    all_models = [
        "distilbert-base-uncased",  # Already tested in grid search, but include for comparison
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-cased",
        "bert-large-cased",
        "roberta-base",
        "roberta-large",
    ]

    print("\n=== EVALUATING MULTIPLE MODELS WITH BEST HYPERPARAMETERS ===")
    print(f"Using best hyperparameters: lr={best_lr:.0e}, epochs={best_epochs}")
    print(f"Attempting to train {len(all_models)} models...\n")
    
    extra_results = []
    for m in all_models:
        print(f"\n--- Training {m} ---")
        try:
            res = run_one_config(
                model_name=m,
                lr=best_lr,
                epochs=best_epochs,
                batch_size=args.batch_size,
                device=torch.device(args.device),
                small_subset=args.small_subset,
            )
            extra_results.append(res)
            print(f"  ✓ Success: dev={res.dev_acc:.4f}, test={res.test_acc:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"  ✗ OOM (Out of Memory): reporting 0.0 for this model")
            else:
                print(f"  ✗ RuntimeError: {e} - reporting 0.0")
            extra_results.append(Result(best_lr, best_epochs, 0.0, 0.0, m))
        except Exception as e:
            print(f"  ✗ Failed with error: {e} - reporting 0.0")
            extra_results.append(Result(best_lr, best_epochs, 0.0, 0.0, m))

    # Select two favorite models (highest test accuracy among successful ones)
    successful_results = [r for r in extra_results if r.test_acc > 0]
    if len(successful_results) >= 2:
        # Sort by test accuracy and pick top 2
        top_two = sorted(successful_results, key=lambda r: r.test_acc, reverse=True)[:2]
        favorite_models = [r.model_name for r in top_two]
        print(f"\n>>> Selected two favorite models (highest test accuracy): {favorite_models}")
    elif len(successful_results) == 1:
        # If only one succeeded, use that one and try to pick another that didn't fail completely
        favorite_models = [successful_results[0].model_name]
        # Pick another model (even if it failed, we'll show it on the plot)
        for r in extra_results:
            if r.model_name not in favorite_models:
                favorite_models.append(r.model_name)
                break
        print(f"\n>>> Selected models: {favorite_models}")
    else:
        # Fallback: use first two models
        favorite_models = [extra_results[0].model_name, extra_results[1].model_name if len(extra_results) > 1 else extra_results[0].model_name]
        print(f"\n>>> Using first two models: {favorite_models}")

    # Filter results to only include favorite models
    plot_results = [r for r in extra_results if r.model_name in favorite_models]

    # ------------------- BAR PLOT -------------------
    if len(plot_results) > 0:
        models = [r.model_name.replace("distilbert-", "DistilBERT-").replace("bert-", "BERT-").replace("roberta-", "RoBERTa-").replace("-", " ").title() 
                  for r in plot_results]
        dev_accs = [r.dev_acc for r in plot_results]
        test_accs = [r.test_acc for r in plot_results]

        x = range(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width/2 for i in x], dev_accs, width, label="Dev Accuracy", color="#66c2a5", alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], test_accs, width, label="Test Accuracy", color="#8da0cb", alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Dev/Test Accuracy for Selected Models\n(Best Hyperparameters: lr={best_lr:.0e}, epochs={best_epochs})", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, max(max(dev_accs), max(test_accs), 0.1) * 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plot_path = "hyperparameter_selection_models.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nBar plot saved to {plot_path}")
        plt.close()  # Close instead of show to avoid blocking
    else:
        print("\nNo successful models to plot!")
    # ------------------------------------------------

    return extra_results


# ----------------------------------------------------------------
#  MAIN ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action="store_true",
                        help="Use tiny data for quick debugging")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size – lower if OOM")
    parser.add_argument("--device", type=str, default=None,
                        help="mps / cuda / cpu")
    args = parser.parse_args()

    # auto-detect device (same logic you already have)
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    # ---------- TODO 1 ----------
    best, _ = grid_search_distilbert(args)

    # ---------- TODO 2 ----------
    evaluate_extra_models(best.lr, best.epochs, args)

    print("\n=== ALL DONE ===")