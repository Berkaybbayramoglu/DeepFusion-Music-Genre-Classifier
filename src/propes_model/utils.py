import os
import csv
from typing import Any


def log_results_to_csv(file_path: str, data: Any, mode: str = 'a', header: bool = True) -> None:
    if isinstance(data, dict):
        data = [data]
    file_exists = os.path.exists(file_path)
    with open(file_path, mode, newline='') as csvfile:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if header and (not file_exists or mode == 'w'):
            writer.writeheader()
        writer.writerows(data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
