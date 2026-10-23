#!/usr/bin/env python
"""Rebuild the trained checkpoints that the scripts load, from the published safetensors.

Inputs: checkpoints/<name>.safetensors + checkpoints/<name>.json (small models are in this
repository; the two large simulators are downloaded from the Hugging Face Hub on demand).
Outputs: one .pt file per model at the path the code expects (`materialize_to` in its JSON),
checked against checkpoints/SHA256SUMS. Run: `make weights` (or `--only fa-sim3 cw-sim`).

torch.save is deterministic on the pinned torch, except for one 40-byte record,
`.data/serialization_id`, which PyTorch derives with the C++ standard library's string hash and
so differs between macOS and Linux. The tensors and every other record are identical. The script
writes the reference value from checkpoints/SERIALIZATION_IDS into that record, so the rebuilt
file is byte-identical to the one the thesis ran on, on either platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile
import zlib

import numpy as np
import torch
from safetensors.torch import load_file

REPO = pathlib.Path(__file__).resolve().parents[1]
CK = REPO / "checkpoints"
HUB = {  # large models hosted on the Hugging Face Hub
    "fa-sim3": "adgodoyouk/atom-fa-sim3",
    "cw-sim": "adgodoyouk/atom-cw-sim",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reference_ids() -> dict[str, bytes]:
    out = {}
    for line in (CK / "SERIALIZATION_IDS").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            sid, p = line.split(maxsplit=1)
            out[p.strip()] = sid.encode()
    return out


def set_serialization_id(path: pathlib.Path, sid: bytes) -> None:
    """Overwrite the stored (uncompressed) serialization_id record and its CRC-32 in place."""
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        info = next(i for i in z.infolist() if i.filename.endswith("/.data/serialization_id"))
    if info.compress_type != zipfile.ZIP_STORED or info.file_size != len(sid):
        raise ValueError(f"unexpected serialization_id record in {path}")
    crc = zlib.crc32(sid).to_bytes(4, "little")
    off = info.header_offset
    flags = int.from_bytes(data[off + 6:off + 8], "little")
    start = off + 30 + int.from_bytes(data[off + 26:off + 28], "little") + int.from_bytes(data[off + 28:off + 30], "little")
    data[start:start + len(sid)] = sid
    if flags & 8:  # CRC lives in the data descriptor after the record
        d = start + len(sid)
        d += 4 if data[d:d + 4] == b"PK\x07\x08" else 0
        data[d:d + 4] = crc
    else:
        data[off + 14:off + 18] = crc
    name, i = info.filename.encode(), data.find(b"PK\x01\x02")
    while i != -1:  # the central-directory copy of the CRC
        if data[i + 46:i + 46 + int.from_bytes(data[i + 28:i + 30], "little")] == name:
            data[i + 16:i + 20] = crc
            break
        i = data.find(b"PK\x01\x02", i + 4)
    path.write_bytes(data)


def decode(cfg, tensors):
    if isinstance(cfg, dict):
        if "__tensor__" in cfg:
            return tensors[cfg["__tensor__"]]
        if "__ndarray__" in cfg:
            return tensors[cfg["__ndarray__"]].numpy()
        if "__tuple__" in cfg:
            return tuple(decode(v, tensors) for v in cfg["__tuple__"])
        if "__size__" in cfg:
            return torch.Size(cfg["__size__"])
        if "__dtype__" in cfg:
            return getattr(torch, cfg["__dtype__"])
        if "__npscalar__" in cfg:
            return np.dtype(cfg["__npscalar__"]).type(cfg["value"])
        if "__path__" in cfg:
            return pathlib.Path(cfg["__path__"])
        return {k: decode(v, tensors) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [decode(v, tensors) for v in cfg]
    return cfg


def expected_hashes() -> dict[str, str]:
    out = {}
    for line in (CK / "SHA256SUMS").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, p = line.split(maxsplit=1)
            out[p.strip()] = h
    return out


def fetch(name: str) -> pathlib.Path:
    st = CK / f"{name}.safetensors"
    if st.exists():
        return st
    if name not in HUB:
        raise FileNotFoundError(st)
    from huggingface_hub import hf_hub_download
    return pathlib.Path(hf_hub_download(HUB[name], f"{name}.safetensors", local_dir=CK))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="release names to rebuild (default: all)")
    ap.add_argument("--skip-hub", action="store_true", help="do not download the large models")
    args = ap.parse_args()
    want = expected_hashes()
    ids = reference_ids()
    bad = 0
    for js in sorted(CK.glob("*.json")):
        name = js.stem
        if args.only and name not in args.only:
            continue
        if args.skip_hub and name in HUB and not (CK / f"{name}.safetensors").exists():
            print(f"skip  {name} (hosted on the Hub)")
            continue
        meta = json.loads(js.read_text())
        st = fetch(name)
        if want.get(f"checkpoints/{name}.safetensors") not in (None, sha256(st)):
            print(f"FAIL  {name}: safetensors hash mismatch")
            bad += 1
            continue
        obj = decode(meta["config"], load_file(str(st)))
        dest = REPO / meta["materialize_to"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, dest)
        if meta["materialize_to"] in ids:
            set_serialization_id(dest, ids[meta["materialize_to"]])
        h = sha256(dest)
        ok = want.get(meta["materialize_to"]) == h
        print(f"{'ok   ' if ok else 'WARN '} {name:38s} -> {meta['materialize_to']}")
        if not ok:
            bad += 1
            print("      the rebuilt file's hash differs from SHA256SUMS; use the torch version pinned in"
                  " uv.lock, since scripts that verify checkpoint hashes will refuse this file")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
