#!/usr/bin/env python
"""Build the Terrier inverted index for the configured collection.

Usage:
    python scripts/build_index.py [--config config/config.yaml] [--overwrite]
"""

import argparse
import logging

from ir_system.data import load_dataset
from ir_system.indexing import build_index
from ir_system.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rebuild even if an index exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config = load_config(args.config)
    dataset = load_dataset(config["dataset"])
    index = build_index(dataset, config["index_dir"], overwrite=args.overwrite)
    stats = index.getCollectionStatistics()
    print(stats.toString())


if __name__ == "__main__":
    main()
