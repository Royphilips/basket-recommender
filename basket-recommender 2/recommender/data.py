"""Loading helpers for the transaction and catalog files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_transactions(path: Path = DATA_DIR / "transactions.csv") -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["items"].split(";") for row in csv.DictReader(f)]


def load_catalog(path: Path = DATA_DIR / "catalog.csv") -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["item"]: row["category"] for row in csv.DictReader(f)}
