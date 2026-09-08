# -*- coding: utf-8 -*-
r"""
download_massive_steps_official_build_fixed.py

一键下载 Massive-STEPS 官方 Hugging Face New York 数据集的 train / validation / test，
并转换成 EffiPOI / RecBole 可以读取的数据目录。

使用：
conda activate effipoi50
pip install datasets pyarrow pandas numpy faiss-cpu

python download_massive_steps_official_build_fixed.py ^
  --out_dir "D:\code\EffiPOI-main\data\NYC_STEPS" ^
  --dataset_name NYC_STEPS
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset


VISIT_PATTERN = re.compile(
    r"At\s+([\d\-:\s]+),\s+user\s+(\d+)\s+visited\s+POI id\s+(\d+)\s+which is a\s+(.+?),\s+at\s+(.+?),\s+([A-Z]{2})\.",
    re.IGNORECASE,
)
TARGET_PATTERN = re.compile(
    r"user\s+(\d+)\s+will visit\s+POI id\s+(\d+)",
    re.IGNORECASE,
)


def numeric_sort_key(x):
    x = str(x)
    return int(x) if x.isdigit() else x


def parse_split(hf_split):
    samples = []
    poi_meta = {}
    bad_rows = 0

    for r in hf_split:
        raw_user_id = str(r["user_id"])
        trail_id = str(r.get("trail_id", ""))
        inputs = str(r["inputs"])
        targets = str(r["targets"])

        visits = VISIT_PATTERN.findall(inputs)
        target_match = TARGET_PATTERN.search(targets)

        if not visits or target_match is None:
            bad_rows += 1
            continue

        raw_seq = []
        for ts, u, poi, cate, loc, country in visits:
            poi = str(poi)
            raw_seq.append(poi)

            if poi not in poi_meta:
                poi_meta[poi] = {
                    "raw_poi_id": poi,
                    "category": cate.strip(),
                    "location": loc.strip(),
                    "country": country.strip(),
                    "first_seen_time": ts.strip(),
                }

        raw_target = str(target_match.group(2))
        if not raw_seq:
            bad_rows += 1
            continue

        if raw_target not in poi_meta:
            poi_meta[raw_target] = {
                "raw_poi_id": raw_target,
                "category": "",
                "location": "",
                "country": "",
                "first_seen_time": "",
            }

        samples.append({
            "raw_user_id": raw_user_id,
            "trail_id": trail_id,
            "raw_seq": raw_seq,
            "raw_target": raw_target,
        })

    return samples, poi_meta, bad_rows


def build_maps(parts):
    raw_users = sorted(
        {s["raw_user_id"] for samples in parts for s in samples},
        key=numeric_sort_key,
    )

    raw_items = set()
    for samples in parts:
        for s in samples:
            raw_items.add(s["raw_target"])
            raw_items.update(s["raw_seq"])

    raw_items = sorted(raw_items, key=numeric_sort_key)

    user_map = {u: str(i + 1) for i, u in enumerate(raw_users)}
    item_map = {it: str(i + 1) for i, it in enumerate(raw_items)}
    return user_map, item_map


def remap_samples(samples, user_map, item_map):
    rows = []
    for s in samples:
        seq = [item_map[x] for x in s["raw_seq"] if x in item_map]
        if not seq:
            continue
        rows.append({
            "user_id:token": user_map[s["raw_user_id"]],
            "item_id_list:token_seq": " ".join(seq),
            "item_id:token": item_map[s["raw_target"]],
        })
    return rows


def save_inter(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id:token", "item_id_list:token_seq", "item_id:token"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_meta(poi_meta, item_map, path: Path):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "item_id_token",
            "raw_poi_id",
            "category",
            "location",
            "country",
            "first_seen_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for raw_id, token in sorted(item_map.items(), key=lambda kv: int(kv[1])):
            meta = poi_meta.get(raw_id, {})
            writer.writerow({
                "item_id_token": token,
                "raw_poi_id": raw_id,
                "category": meta.get("category", ""),
                "location": meta.get("location", ""),
                "country": meta.get("country", ""),
                "first_seen_time": meta.get("first_seen_time", ""),
            })


def stable_hash_int(text: str):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def hashed_text_feature(text: str, dim=1664, density=64):
    vec = np.zeros(dim, dtype=np.float32)

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower()) or ["unknown"]
    features = tokens[:]
    for i in range(len(tokens) - 1):
        features.append(tokens[i] + "_" + tokens[i + 1])

    reps = max(1, density // max(1, len(features)))
    for tok in features:
        base = stable_hash_int(tok)
        for j in range(reps):
            idx = (base + j * 9973) % dim
            sign = 1.0 if ((base >> (j % 16)) & 1) else -1.0
            vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build_feat1cls(item_map, poi_meta, out_path: Path, dim=1664):
    n = len(item_map)
    feat = np.zeros((n, dim), dtype=np.float32)
    token_to_row = {}

    for raw_id, token in item_map.items():
        row = int(token) - 1
        token_to_row[token] = row

        meta = poi_meta.get(raw_id, {})
        text = " ".join([
            "poi",
            raw_id,
            meta.get("category", ""),
            meta.get("location", ""),
            meta.get("country", ""),
        ])
        feat[row] = hashed_text_feature(text, dim=dim)

    feat.tofile(out_path)

    token_map_path = out_path.with_name(out_path.name.replace(".feat1CLS", ".feat_token_to_row.json"))
    with token_map_path.open("w", encoding="utf-8") as f:
        json.dump(token_to_row, f, ensure_ascii=False, indent=2)

    return feat, token_map_path


def build_faiss_index(feat, out_path: Path, m=128, nbits=4):
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("请先安装 faiss-cpu: pip install faiss-cpu") from e

    x = feat.astype(np.float32)
    d = x.shape[1]

    print(f"[FAISS] training OPQ{m},IVF1,PQ{m}x{nbits}, x={x.shape}")
    opq = faiss.OPQMatrix(d, m)
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, 1, m, nbits)
    index = faiss.IndexPreTransform(opq, ivfpq)

    index.train(x)
    index.add(x)

    faiss.write_index(index, str(out_path))
    print(f"[FAISS] saved: {out_path}")
    print(f"[FAISS] ntotal={index.ntotal}, d={index.d}")


def pick_split(ds, *names):
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of these splits exist: {names}. Available: {list(ds.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="CRUISEResearchGroup/Massive-STEPS-New-York")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_name", default="NYC_STEPS")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--dim", type=int, default=1664)
    parser.add_argument("--skip_faiss", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HF] loading dataset: {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, cache_dir=args.cache_dir or None)
    print("[HF] splits:", list(ds.keys()))

    train_split = pick_split(ds, "train")
    valid_split = pick_split(ds, "validation", "valid", "val")
    test_split = pick_split(ds, "test")

    print(f"[HF] train={len(train_split)}, valid={len(valid_split)}, test={len(test_split)}")

    train_samples, train_meta, train_bad = parse_split(train_split)
    valid_samples, valid_meta, valid_bad = parse_split(valid_split)
    test_samples, test_meta, test_bad = parse_split(test_split)

    poi_meta = {}
    poi_meta.update(train_meta)
    poi_meta.update(valid_meta)
    poi_meta.update(test_meta)

    user_map, item_map = build_maps([train_samples, valid_samples, test_samples])

    train_rows = remap_samples(train_samples, user_map, item_map)
    valid_rows = remap_samples(valid_samples, user_map, item_map)
    test_rows = remap_samples(test_samples, user_map, item_map)

    name = args.dataset_name
    save_inter(train_rows, out_dir / f"{name}.train.inter")
    save_inter(valid_rows, out_dir / f"{name}.valid.inter")
    save_inter(test_rows, out_dir / f"{name}.test.inter")

    with (out_dir / f"{name}.user_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(user_map, f, ensure_ascii=False, indent=2)

    with (out_dir / f"{name}.item_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(item_map, f, ensure_ascii=False, indent=2)

    save_meta(poi_meta, item_map, out_dir / f"{name}.poi_meta_from_inputs.csv")

    feat_path = out_dir / f"{name}.feat1CLS"
    feat, token_map_path = build_feat1cls(item_map, poi_meta, feat_path, dim=args.dim)

    index_path = out_dir / f"{name}.OPQ128,IVF1,PQ128x4.strict.index"
    if not args.skip_faiss:
        build_faiss_index(feat, index_path, m=128, nbits=4)

    print("\nDone.")
    print("Output dir:", out_dir)
    print("train rows:", len(train_rows))
    print("valid rows:", len(valid_rows))
    print("test rows:", len(test_rows))
    print("users:", len(user_map))
    print("items without padding:", len(item_map))
    print("feat shape:", feat.shape)
    print("feat token map:", token_map_path)
    print("bad rows:", train_bad + valid_bad + test_bad)


if __name__ == "__main__":
    main()
