"""Generate the shipped sample dataset from the package's simulation.

Produces a CSV with deliberately non-canonical column names (survival_time,
died, feat_0 ...) so the mapping step in the UI has something to do.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `import paths` works

import paths  # noqa: E402  -- discovers the repo and puts deepquantreg on sys.path
paths.require_repo()

from deepquantreg.data import generate_simulation  # noqa: E402


def main(n: int = 300, seed: int = 7) -> Path:
    frame = generate_simulation("Gaussian_Uniform10D_v1", n, seed)
    features = [c for c in frame.columns if c not in {"duration", "event", "subject_id"}]
    renamed = frame[features].copy()
    renamed.columns = [f"feat_{i}" for i in range(len(features))]
    renamed["survival_time"] = frame["duration"].round(4)
    renamed["died"] = frame["event"].astype(int)
    out = HERE / "sample.csv"
    renamed.to_csv(out, index=False)
    print(f"wrote {out} ({len(renamed)} rows, {len(features)} features, "
          f"{renamed['died'].mean() * 100:.1f}% events)")
    return out


if __name__ == "__main__":
    main()
