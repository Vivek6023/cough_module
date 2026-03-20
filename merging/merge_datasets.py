import os
import shutil
import pandas as pd
from utils.config import LABEL_MAP

LOG = []

def merge(source_folder, target_label, dataset_name):
    target_dir = f"data/merged_dataset/{target_label}"
    os.makedirs(target_dir, exist_ok=True)

    for file in os.listdir(source_folder):
        if file.endswith(".wav"):
            src = os.path.join(source_folder, file)
            dst = os.path.join(target_dir, f"{dataset_name}_{file}")
            shutil.copy(src, dst)

            LOG.append({
                "file": dst,
                "label": target_label,
                "source": dataset_name
            })
