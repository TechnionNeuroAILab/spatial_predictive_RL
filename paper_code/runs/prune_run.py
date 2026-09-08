"""
prune_run.py
Free disk space in a finished run directory by deleting the large activation
probes that downstream analysis does not need, keeping only the essentials.

By default it keeps ONLY the frozen-replay pair (replay_c1_pred, replay_c1_postrl)
activations -- everything the multi-seed dissociation aggregator needs. All small
files (df_behavior_*, df_skaggs_*, df_centers_*, config.json, figures) are kept.
Checkpoints are kept unless --drop-checkpoints is passed (they are only ~32 MB and
are needed for the reward-head readout analysis).

This is DESTRUCTIVE (deleted activation probes can only be recovered by
re-training). Use --dry-run first to see exactly what would be deleted.

Usage:
  python paper_code/runs/prune_run.py --run-dir New_Results_v2/seed_runs/head_seed42 --dry-run
  python paper_code/runs/prune_run.py --run-dir New_Results_v2/seed_runs/head_seed42
  python paper_code/runs/prune_run.py --run-dir <DIR> --keep replay_c1_pred replay_c1_postrl goal_conditioned
"""
from __future__ import annotations
import argparse
import glob
import os


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", "--run_dir", dest="run_dir", required=True)
    ap.add_argument("--keep", nargs="+", default=["replay_c1_pred", "replay_c1_postrl"],
                    help="periods whose df_activations_*.pkl to KEEP (default: the replay pair)")
    ap.add_argument("--drop-checkpoints", action="store_true",
                    help="also delete checkpoints/*.pt (not needed for the shift aggregation)")
    ap.add_argument("--dry-run", action="store_true", help="only report; delete nothing")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(args.run_dir, "data")
    keep = set(args.keep)
    to_delete = []

    # only ever touch the big df_activations_*.pkl files
    for path in sorted(glob.glob(os.path.join(data_dir, "df_activations_*.pkl"))):
        period = os.path.basename(path)[len("df_activations_"):-len(".pkl")]
        if period not in keep:
            to_delete.append(path)

    if args.drop_checkpoints:
        to_delete += sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "*.pt")))

    freed = 0
    for path in to_delete:
        sz = os.path.getsize(path)
        freed += sz
        if args.dry_run:
            print(f"  would delete  {sz/1e6:7.1f} MB  {path}")
        else:
            os.remove(path)
            print(f"  deleted       {sz/1e6:7.1f} MB  {path}")

    kept = sorted(os.path.basename(p)[len('df_activations_'):-len('.pkl')]
                  for p in glob.glob(os.path.join(data_dir, "df_activations_*.pkl"))
                  if os.path.basename(p)[len('df_activations_'):-len('.pkl')] in keep)
    verb = "would free" if args.dry_run else "freed"
    print(f"[prune_run] {args.run_dir}: {verb} {freed/1e6:.0f} MB  "
          f"(kept activations: {kept or 'none'})")


if __name__ == "__main__":
    main()
