# -*- coding: utf-8 -*-
"""
Grokking training script (Power et al. 2022, arXiv:2201.02177).

Paper default config (Appendix A.1.2):
  AdamW, lr = 1e-3, weight decay = 1, betas = (0.9, 0.98),
  linear warmup over the first 10 steps, batch size = 512,
  optimization budget = 1e5 steps (early-stops once grokked).

TensorBoard scalars:
  train/loss, train/acc, val/loss, val/acc   (the grokking curves)
  train/lr

Usage:
  python train.py --steps 100000 --wd 1.0
  python train.py --wd 0.0 --full-batch      # Fig.1 style: late grokking
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import GrokTransformer, make_vocab, encode_batch


def load_dataset(data_dir, p, train_frac, seed):
    """Load plaintext CSV if it exists; otherwise generate and cache on disk."""
    prefix = os.path.join(data_dir, f"div_{p}")
    train_csv, val_csv = prefix + "_train.csv", prefix + "_val.csv"

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        def read_csv(path):
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r)  # skip header
                return np.array([[int(v) for v in row] for row in r],
                                dtype=np.int64)
        return read_csv(train_csv), read_csv(val_csv)

    # fallback: generate in-place
    import random
    from dataset import generate_equations
    rng = random.Random(seed)
    eqs = generate_equations(p)
    rng.shuffle(eqs)
    n_train = int(round(len(eqs) * train_frac))
    return (np.array(eqs[:n_train], dtype=np.int64),
            np.array(eqs[n_train:], dtype=np.int64))


@torch.no_grad()
def evaluate(model, tokens, batch_size=4096, device="cuda"):
    """Evaluate mean loss / accuracy over the full dataset."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for i in range(0, len(tokens), batch_size):
        t = tokens[i:i + batch_size].to(device)
        loss, correct, n = model.loss_and_acc(t)
        total_loss += loss.item() * n
        total_correct += correct.item()
        total_n += n
    model.train()
    return total_loss / total_n, total_correct / total_n


def main():
    ap = argparse.ArgumentParser(description="Grokking training (Power et al. 2022)")
    ap.add_argument("--p", type=int, default=97, help="prime modulus")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--full-batch", action="store_true",
                    help="full-batch gradients (deterministic optimization, "
                         "Sec. 3.3 of the paper: grokking arrives much later)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0,
                    help="weight decay (paper default: 1)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--warmup", type=int, default=10, help="linear warmup steps")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--early-stop-acc", type=float, default=0.999,
                    help="stop when val acc reaches this value (<=0 disables)")
    ap.add_argument("--data-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    run_name = args.run_name or f"div{args.p}_wd{args.wd}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---------- data ----------
    train_np, val_np = load_dataset(args.data_dir, args.p,
                                    args.train_frac, args.seed)
    print(f"[data] x / y mod {args.p}   train {len(train_np)} / val {len(val_np)}")

    train_batch = encode_batch(train_np[:, 0], train_np[:, 1],
                               train_np[:, 2], args.p)
    val_batch = encode_batch(val_np[:, 0], val_np[:, 1],
                             val_np[:, 2], args.p)
    n_train = len(train_np)
    batch_size = n_train if args.full_batch else min(args.batch_size, n_train)

    # ---------- model (built on nn.TransformerEncoder) ----------
    model = GrokTransformer(make_vocab(args.p),
                            d_model=args.d_model,
                            n_layers=args.n_layers,
                            n_heads=args.n_heads).to(device)
    n_params = sum(par.numel() for par in model.parameters())
    print(f"[model] {args.n_layers}-layer decoder-only transformer, "
          f"d_model={args.d_model}, heads={args.n_heads}, params {n_params:,}")

    # ---------- optimizer (paper config) ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.wd)

    writer = SummaryWriter(log_dir=out_dir)
    writer.add_text("config", json.dumps(vars(args), indent=2))

    rng = np.random.default_rng(args.seed)
    log_rows = []
    t0 = time.time()
    grok_step = None
    stop_reason = "budget"

    for step in range(1, args.steps + 1):
        # linear warmup (paper: first 10 steps)
        lr_now = args.lr * min(1.0, step / max(1, args.warmup))
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # sample a minibatch (or take the full batch)
        if args.full_batch:
            batch = train_batch.to(device)
        else:
            idx = rng.integers(0, n_train, size=batch_size)
            batch = train_batch[idx].to(device)

        loss, _, _ = model.loss_and_acc(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ---------- logging ----------
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            tr_loss, tr_acc = evaluate(model, train_batch, device=device)
            va_loss, va_acc = evaluate(model, val_batch, device=device)
            writer.add_scalar("train/loss", tr_loss, step)
            writer.add_scalar("train/acc", tr_acc, step)
            writer.add_scalar("val/loss", va_loss, step)
            writer.add_scalar("val/acc", va_acc, step)
            writer.add_scalar("train/lr", lr_now, step)
            print(f"step {step:>7d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            log_rows.append((step, tr_loss, tr_acc, va_loss, va_acc))

            if grok_step is None and va_acc >= 0.99:
                grok_step = step
                print(f">>> GROKKED: val acc first >= 99% at step {step}")
                torch.save(model.state_dict(),
                           os.path.join(out_dir, "model_grokked.pt"))

            # early stop: perfect generalization reached
            if (args.early_stop_acc > 0 and tr_acc >= 0.999
                    and va_acc >= args.early_stop_acc and step >= 1000):
                stop_reason = "early_stop"
                break

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(log_rows)
    writer.close()

    print(f"[done] reason={stop_reason}  grok_step={grok_step}  "
          f"wall time {(time.time() - t0) / 60:.1f} min")
    print(f"[artifacts] {out_dir}  (model_final.pt / model_grokked.pt / "
          f"train_log.csv / tensorboard events)")


if __name__ == "__main__":
    main()
