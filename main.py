#!/usr/bin/env python3
"""CLI: chunk .md/.txt documents and create embeddings (config: config.json).

Usage:
    python main.py                  # chunk + embed
    python main.py --dry-run        # chunk only, no embeddings (free)
    python main.py --config path.json

    # Re-embed an existing run with a different model (same chunk IDs):
    python main.py --reembed <run-dir> [--out-dir DIR] [--out-name NAME]

GUI variant: python app.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import find_input_files, reembed_run, run_pipeline


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk only, do not create embeddings")
    parser.add_argument("--reembed", metavar="RUN_DIR",
                        help="re-embed the chunks of an existing run with the "
                             "currently configured model (keeps chunk IDs)")
    parser.add_argument("--out-name", metavar="NAME",
                        help="name of the target folder (default: model name)")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="target directory (default: output_dir from config)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.reembed:
        try:
            reembed_run(Path(args.reembed), config, run_name=args.out_name,
                        output_dir=args.out_dir, progress=print)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    input_dir = Path(config.get("input_dir", "./input"))
    if not input_dir.is_dir():
        print(f"Error: input_dir {input_dir} does not exist.", file=sys.stderr)
        return 1

    files = find_input_files(input_dir)
    if not files:
        print(f"No .md/.txt files found in {input_dir}.", file=sys.stderr)
        return 1

    try:
        run_pipeline(files, config, dry_run=args.dry_run,
                     input_dir=input_dir, progress=print)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
