# -*- coding: utf-8 -*-
"""
Grokking dataset generator (Power et al. 2022, arXiv:2201.02177).

Generates the plaintext dataset for a single binary operation:
modular division x / y (mod 97), where y != 0.

Each equation is a 5-token sequence:  <x> <op> <y> <=> <answer>
- operands and answers are discrete symbols in 0..96 (no numeric structure)
- the dataset is randomly split into train / val by a given fraction

Usage:
    python dataset.py --p 97 --train-frac 0.5 --out-dir ../data
"""
import argparse
import csv
import json
import os
import random


def binary_div(x: int, y: int, p: int) -> int:
    """Modular division: x / y = x * y^(-1) (mod p). Requires y != 0."""
    return (x * pow(y, -1, p)) % p


def generate_equations(p: int):
    """Enumerate all valid (x, y, answer) triples for x / y mod p."""
    equations = []
    for x in range(p):
        for y in range(1, p):  # y = 0 is excluded
            equations.append((x, y, binary_div(x, y, p)))
    return equations


def save_csv(path: str, rows):
    """Write plaintext CSV with columns: a, b, answer."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "answer"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Grokking plaintext dataset generator")
    ap.add_argument("--p", type=int, default=97,
                    help="prime modulus (paper uses 97)")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="fraction of equations used for training (Fig.1 uses 0.5)")
    ap.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    equations = generate_equations(args.p)
    rng.shuffle(equations)

    n_train = int(round(len(equations) * args.train_frac))
    train, val = equations[:n_train], equations[n_train:]

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"div_{args.p}")
    save_csv(prefix + "_train.csv", train)
    save_csv(prefix + "_val.csv", val)

    meta = {
        "op": "div", "p": args.p, "seed": args.seed,
        "train_frac": args.train_frac,
        "n_equations": len(equations),
        "n_train": len(train), "n_val": len(val),
        "files": [prefix + "_train.csv", prefix + "_val.csv"],
    }
    with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"op: x / y mod {args.p}   train_frac: {args.train_frac}")
    print(f"total equations: {len(equations)}   train: {len(train)}   val: {len(val)}")
    print(f"written: {prefix}_train.csv / {prefix}_val.csv / {prefix}_meta.json")
    print("samples:", [f"{a} / {b} = {c} mod {args.p}" for a, b, c in train[:3]])


if __name__ == "__main__":
    main()
