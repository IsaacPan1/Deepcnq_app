"""Developer CLI to build **committable** demo assets in ``demo/``.

The app generates the demo *data* automatically (see ``demo.ensure_demo_data``),
so end users never need this. It exists only to (re)produce the committed
`demo/demo_model.cnqmodel` (and refreshed CSVs) on a machine with torch:

    python make_demo.py              # write demo/*.csv + train demo/demo_model.cnqmodel
    python make_demo.py --data-only  # CSVs only (no torch needed)
    python make_demo.py --epochs 25  # fewer epochs if training is slow
    python make_demo.py --model KAN_gaps

Then commit the result: ``git add demo/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402
import demo  # noqa: E402

# Re-export for the existing test-suite import surface.
FILES = {**demo.CSV_FILES, "model": demo.MODEL_FILE}
DEFAULT_MODEL = demo.DEFAULT_MODEL


def build_all(out_dir=None, *, model_name: str = demo.DEFAULT_MODEL,
              epochs: int = demo.DEFAULT_EPOCHS, with_model: bool = True,
              deterministic: bool = True, **_ignored) -> dict:
    """Write the CSVs (and optionally the model) into ``out_dir`` (default demo/)."""
    out_dir = Path(out_dir) if out_dir else demo.COMMITTED_DIR
    sizes = demo.write_all_csvs(out_dir)
    info = {"sizes": sizes}
    if with_model:
        paths.require_repo()
        info["model"] = demo.build_demo_model(
            model_name=model_name, epochs=epochs, deterministic=deterministic,
            out_path=out_dir / demo.MODEL_FILE)
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build committable CNQ demo assets in demo/.")
    parser.add_argument("--model", default=demo.DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=demo.DEFAULT_EPOCHS)
    parser.add_argument("--data-only", action="store_true", help="write CSVs only (no torch)")
    args = parser.parse_args(argv)

    info = build_all(demo.COMMITTED_DIR, model_name=args.model, epochs=args.epochs,
                     with_model=not args.data_only)
    print("Demo CSVs written to", demo.COMMITTED_DIR)
    for key, n in info["sizes"].items():
        print(f"  {demo.CSV_FILES[key]}: {n} rows")
    if info.get("model"):
        m = info["model"]
        print(f"Demo model: {m['model']} — {m['size_mb']} MB, {m['seconds']} s, "
              f"{m['n_levels']} levels, {m['n_features']} features")
        print("  metrics:", m["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
