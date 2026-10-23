#!/usr/bin/env python
"""Trains the AKT baseline on the full FoundationalASSIST training split on a GPU instance
and reports its validation AUC, accuracy, ECE and Brier score.

Inputs: the AKT tensor bundle (integer concept, problem and response ids with masks; no
raw text) built by scripts/p3_05b_prepare_akt_full_data.py, and configs/akt_baseline.yaml,
both read from the S3 bucket given with `--bucket`. Outputs: periodic and final
checkpoints and results/summary.json, uploaded to the same bucket. The model is the
unmodified `pykt.models.akt.AKT`, trained in minibatches of `full_run_batch_size`
because full-batch attention over all chunks does not fit in 24GB. Self-contained (no
`atom` import); shuts the instance down when finished. Serves the AKT row of the FA
baseline comparison (Section 6.2).

Usage (on the instance):
  python3 p3_05b_akt_full_train.py --bucket <bucket> \
      --data-prefix p3-05b-akt-full/data --out-prefix p3-05b-akt-full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _masked_bce_from_probs(probs, targets, mask):
    probs = probs.clamp(1e-6, 1 - 1e-6)
    bce = -(targets * probs.log() + (1 - targets) * (1 - probs).log())
    return (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)


def _score(probs, correct, mask):
    from sklearn.metrics import roc_auc_score

    m = mask.cpu()
    y = correct.cpu()[m].numpy()
    p = probs.detach().cpu()[m].numpy()
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    acc = float(((p >= 0.5).astype(int) == y).mean())
    brier = float(((p - y) ** 2).mean())
    # ECE, 10 equal-width bins -- same formula as src/atom/eval/metrics.py::ece.
    import numpy as np

    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        bmask = idx == b
        if not bmask.any():
            continue
        ece += (bmask.sum() / len(y)) * abs(y[bmask].mean() - p[bmask].mean())
    return {"auc": auc, "accuracy": acc, "brier": brier, "ece": float(ece)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--data-prefix", default="p3-05b-akt-full/data")
    ap.add_argument("--out-prefix", default="p3-05b-akt-full")
    ap.add_argument("--checkpoint-every-s", type=float, default=1800.0)  # checkpoint at least every 30 min
    args = ap.parse_args()

    t0 = time.time()
    last_ckpt_t = t0

    import boto3
    import torch

    s3 = boto3.client("s3")
    local_bundle = "/tmp/akt_full_tensors.pt"
    local_cfg = "/tmp/akt_baseline.yaml"
    s3.download_file(args.bucket, f"{args.data_prefix}/akt_full_tensors.pt", local_bundle)
    s3.download_file(args.bucket, f"{args.out_prefix}/akt_baseline.yaml", local_cfg)
    print(f"[p3-05b] downloaded bundle + config, elapsed={time.time()-t0:.1f}s", flush=True)

    import yaml

    cfg = yaml.safe_load(Path(local_cfg).read_text())
    payload = torch.load(local_bundle, map_location="cpu", weights_only=False)

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"[p3-05b] cuda_available={cuda_ok} device={device} "
          f"gpu_name={torch.cuda.get_device_name(0) if cuda_ok else 'n/a'}", flush=True)
    if not cuda_ok:
        print("[p3-05b] FAIL: no CUDA device visible on a g5.xlarge")
        return 1

    import pykt.models.akt as _akt_module
    from pykt.models.akt import AKT

    # Verify (not assume) whether pyKT's MPS-era device patch is needed on CUDA: pyKT
    # 0.0.38's akt.py hard-codes a module-level `device = cuda-or-cpu` used internally for
    # causal-mask construction. On CUDA that expression naturally resolves to "cuda" --
    # the same device the model itself lives on -- so, unlike on Mac/MPS (where it silently
    # fell back to "cpu" and crashed), the patch is normally a no-op here. Both values
    # are printed and the module global is patched only if they differ.
    pre_patch_device = _akt_module.device
    print(f"[p3-05b] pykt.models.akt.device (pre-patch) = {pre_patch_device} vs our device = {device}", flush=True)
    if str(pre_patch_device) != str(device):
        print("[p3-05b] mismatch detected -- patching pykt.models.akt.device to the model device", flush=True)
        _akt_module.device = device
    else:
        print("[p3-05b] no patch needed: pyKT's own cuda-or-cpu global already matches our device", flush=True)

    torch.manual_seed(cfg["seed"])
    model = AKT(n_question=payload["n_question"], n_pid=payload["n_pid"], **cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train = payload["train"]
    val = payload["val"]
    n_train = train["concepts"].shape[0]
    batch_size = cfg["full_run_batch_size"]
    epochs = cfg["epochs"]

    val_concepts = val["concepts"].to(device)
    val_pids = val["pids"].to(device)
    val_responses = val["responses"].to(device)
    val_mask = val["valid_mask"].clone()
    val_mask[:, 0] = False  # never score position 0 -- pyKT's own convention (zero in-chunk history)
    val_mask = val_mask.to(device)

    def checkpoint_and_upload(step: int, epoch: int, metric: float | None, is_final: bool) -> str:
        ckpt_path = f"/tmp/akt_full_ckpt_step{step}.pt"
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                   "epoch": epoch, "step": step, "metric": metric, "config": cfg}, ckpt_path)
        key = f"{args.out_prefix}/checkpoints/{'final' if is_final else f'step{step:06d}'}.pt"
        s3.upload_file(ckpt_path, args.bucket, key)
        print(f"[p3-05b] checkpoint uploaded to bucket {args.bucket}: {key}", flush=True)
        return key

    rng = torch.Generator().manual_seed(cfg["seed"])
    loss_history = []
    step = 0
    last_hb_t = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        epoch_loss_sum, epoch_loss_n = 0.0, 0
        for bstart in range(0, n_train, batch_size):
            idx = perm[bstart:bstart + batch_size]
            c = train["concepts"][idx].to(device)
            p = train["pids"][idx].to(device)
            r = train["responses"][idx].to(device)
            m = train["valid_mask"][idx].clone()
            m[:, 0] = False
            m = m.to(device)

            opt.zero_grad()
            preds, reg_loss = model(c, r, p)
            loss = _masked_bce_from_probs(preds, r.float(), m) + reg_loss
            loss_val = loss.item()
            if not (loss_val == loss_val) or loss_val in (float("inf"), float("-inf")):
                print(f"[p3-05b] TRIPWIRE non-finite loss at epoch={epoch} step={step}: {loss_val}", flush=True)
                return 2
            loss.backward()
            opt.step()
            step += 1
            epoch_loss_sum += loss_val * m.sum().item()
            epoch_loss_n += m.sum().item()

            now = time.time()
            if now - last_hb_t >= 30.0:  # heartbeat every 30 s
                elapsed = now - t0
                print(f"[hb] step={step} epoch={epoch}/{epochs} loss={loss_val:.4f} "
                      f"elapsed={elapsed:.0f}s gpu_mem_gb={torch.cuda.memory_allocated()/1e9:.2f}", flush=True)
                last_hb_t = now

            if now - last_ckpt_t >= args.checkpoint_every_s:
                checkpoint_and_upload(step, epoch, None, is_final=False)
                last_ckpt_t = now

        epoch_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        loss_history.append(epoch_loss)
        print(f"[p3-05b] epoch={epoch}/{epochs} mean_loss={epoch_loss:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        val_preds, _ = model(val_concepts, val_responses, val_pids)
    val_metrics = _score(val_preds, val_responses, val_mask)

    lo, hi = cfg["full_run_gate_auc_lo"], cfg["full_run_gate_auc_hi"]
    gate_pass = bool(lo <= val_metrics["auc"] <= hi)
    loss_decreased = bool(loss_history[-1] < loss_history[0])

    final_ckpt_key = checkpoint_and_upload(step, epochs, val_metrics["auc"], is_final=True)

    summary = {
        "passed": gate_pass, "loss_decreased": loss_decreased, "val_metrics": val_metrics,
        "loss_first": loss_history[0], "loss_last": loss_history[-1],
        "loss_history": loss_history, "train_s": train_s, "total_s": time.time() - t0,
        "final_checkpoint_key": final_ckpt_key, "n_train_chunks": n_train,
        "n_val_chunks": val_concepts.shape[0], "batch_size": batch_size, "epochs": epochs,
        "gpu_name": torch.cuda.get_device_name(0),
        "gate_band": [lo, hi],
    }
    summary_path = "/tmp/akt_full_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    summary_key = f"{args.out_prefix}/results/summary.json"
    s3.upload_file(summary_path, args.bucket, summary_key)

    # Confirm the summary upload before shutting down (the launching machine should also
    # check the bucket independently rather than rely on this script's output).
    head = s3.head_object(Bucket=args.bucket, Key=summary_key)
    print(f"[p3-05b] verified summary upload to bucket {args.bucket}: {summary_key} "
          f"({head['ContentLength']} bytes)", flush=True)

    print("[p3-05b] SUMMARY " + json.dumps(summary), flush=True)
    print(f"[p3-05b] {'PASS' if gate_pass else 'FAIL'} val_auc={val_metrics['auc']:.4f} "
          f"band=[{lo},{hi}] wall={summary['total_s']:.0f}s", flush=True)

    import subprocess

    print("[p3-05b] self-terminating instance now", flush=True)
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
