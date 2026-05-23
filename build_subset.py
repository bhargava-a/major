"""
Build an optimized subset NPZ for training the 5 models.

Strategy
--------
1. Select the top-N best-covered entities (most rows, contiguous coverage).
2. Take a contiguous window of W weeks from the source 10-min data.
3. Produce a per-entity temporal split: train / val / test by week proportion.
4. Save a single compact NPZ containing all three splits with their id_time,
   entity_id, and split label. The model loader only needs to read this one
   file — no per-model preprocessing.

Why per-entity split
--------------------
Sliding windows must not cross entity boundaries (that would mix unrelated
subnets into the same sequence). The NPZ stores data sorted by (entity_id,
id_time), and the loader can safely build windows within each entity.

Defaults (tunable via CLI flags)
-------------------------------
  entities:  50   (out of 511)
  weeks   :   8   (6 train + 1 val + 1 test)
  source  : cesnet_10min.npz
  output  : data/cesnet_subset.npz

Storage layout of the output NPZ
--------------------------------
  data          float32 [N, 18]
  id_time       int32   [N]
  entity_id     int32   [N]
  split         uint8   [N]    0=train, 1=val, 2=test
  feature_names array of str
  meta          0-d str (JSON describing build parameters)
"""

import argparse
import json
import os
import numpy as np

SRC = r"C:\Major_Project\data\cesnet_10min.npz"
OUT = r"C:\Major_Project\data\cesnet_subset.npz"

SLOTS_PER_DAY = 144  # 10-min: 24h * 6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=50,
                    help="number of top entities to keep")
    ap.add_argument("--weeks", type=int, default=8,
                    help="contiguous weeks to keep per entity")
    ap.add_argument("--train-weeks", type=int, default=6)
    ap.add_argument("--val-weeks",   type=int, default=1)
    ap.add_argument("--test-weeks",  type=int, default=1)
    ap.add_argument("--start-slot", type=int, default=0,
                    help="start offset into each entity's timeline (in slots)")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    return ap.parse_args()


def main():
    args = parse_args()
    assert args.train_weeks + args.val_weeks + args.test_weeks == args.weeks, \
        "train+val+test must equal --weeks"

    print(f"Loading {args.src} ...")
    z = np.load(args.src, allow_pickle=True)
    data      = z['data']
    id_time   = z['id_time']
    entity_id = z['entity_id']
    feat_names = [str(x) for x in z['feature_names']]
    print(f"  rows={len(data):,}  features={data.shape[1]}")

    # --- pick best-covered entities --------------------------------------
    unique_ents, counts = np.unique(entity_id, return_counts=True)
    order = np.argsort(-counts)  # descending
    top_ents = unique_ents[order[:args.entities]]
    top_counts = counts[order[:args.entities]]
    print(f"Top {args.entities} entities: "
          f"rows per entity min={top_counts.min()}  "
          f"median={int(np.median(top_counts))}  "
          f"max={top_counts.max()}")

    # --- slice contiguous window per entity ------------------------------
    slots_per_week = 7 * SLOTS_PER_DAY  # 1008
    total_slots = args.weeks * slots_per_week
    start = args.start_slot
    end   = start + total_slots
    train_end = start + args.train_weeks * slots_per_week
    val_end   = train_end + args.val_weeks   * slots_per_week
    print(f"Window: slots [{start}, {end})  total={total_slots}  "
          f"(train {start}-{train_end}, val {train_end}-{val_end}, test {val_end}-{end})")

    # membership masks
    mask_top = np.isin(entity_id, top_ents)
    mask_win = (id_time >= start) & (id_time < end)
    mask = mask_top & mask_win
    kept = int(mask.sum())
    print(f"Rows after entity+time filter: {kept:,}")

    # --- build split labels ---------------------------------------------
    sub_time = id_time[mask]
    split = np.full(kept, 255, dtype=np.uint8)
    split[sub_time <  train_end] = 0
    split[(sub_time >= train_end) & (sub_time < val_end)] = 1
    split[(sub_time >= val_end)   & (sub_time < end)] = 2
    assert (split != 255).all(), "unassigned split rows remain"

    sub_data  = data[mask]
    sub_eid   = entity_id[mask]

    # sort by (entity_id, id_time) for safe per-entity window construction
    order2 = np.lexsort((sub_time, sub_eid))
    sub_data  = sub_data[order2]
    sub_time  = sub_time[order2]
    sub_eid   = sub_eid[order2]
    split     = split[order2]

    # --- split stats -----------------------------------------------------
    for label, name in [(0, 'train'), (1, 'val'), (2, 'test')]:
        n = int((split == label).sum())
        uniq = np.unique(sub_eid[split == label]).size
        print(f"  {name:5s}: rows={n:,}  entities={uniq}")

    meta = {
        'source': args.src,
        'entities_selected': int(len(top_ents)),
        'entity_ids': [int(e) for e in top_ents],
        'weeks_total': args.weeks,
        'train_weeks': args.train_weeks,
        'val_weeks':   args.val_weeks,
        'test_weeks':  args.test_weeks,
        'start_slot': start,
        'end_slot':   end,
        'slots_per_week': slots_per_week,
        'slots_per_day':  SLOTS_PER_DAY,
        'rows_total': kept,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        data=sub_data.astype(np.float32),
        id_time=sub_time.astype(np.int32),
        entity_id=sub_eid.astype(np.int32),
        split=split,
        feature_names=np.array(feat_names),
        meta=np.array(json.dumps(meta)),
    )
    print(f"\nSaved {args.out}")
    print(f"  size MB: {os.path.getsize(args.out)/1e6:.2f}")


if __name__ == "__main__":
    main()
