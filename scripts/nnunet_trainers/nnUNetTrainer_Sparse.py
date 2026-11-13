"""Trainer for the sparse canal labels (Dataset506: background / canal / ignore).

With the default learning rate (1e-2) fold 0 reached pseudo dice 0.69 by epoch 35 and then
collapsed to 0 (everything predicted background) by epoch 59: the canal is ~5k voxels per
case against >1e8 background, and most of the region around it is ignore. A tenth of the
learning rate, trained from scratch (not from the 504 weights, so what the model draws
where the wall is unclear is learned from these labels only, not from the public set's
smooth tubes).

nnU-Net only finds trainers inside its own package, so this file is copied to
nnunetv2/training/nnUNetTrainer/variants/ before training.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Sparse(nnUNetTrainer):
    # The signature must match the base class exactly: nnU-Net records the constructor
    # arguments by name to rebuild the trainer at inference time.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 500


class nnUNetTrainer_Sparse10(nnUNetTrainer_Sparse):
    """Same training, 10-fold CV (professor's request, 2026-11-12).

    Reads splits_10fold.json (made by scripts/make_508_10fold.py) instead of splits_final.json,
    so the 5-fold results (fold_0..4 under nnUNetTrainer_Sparse) are kept; this trainer's results
    go to their own folder. No silent fallback: nnU-Net's default would train a random 80:20 split
    when the fold is missing.
    """

    def do_split(self):
        import os
        from batchgenerators.utilities.file_and_folder_operations import load_json
        p = os.path.join(self.preprocessed_dataset_folder_base, "splits_10fold.json")
        if not os.path.isfile(p):
            raise RuntimeError(f"找不到 {p}:先把 splits_10fold.json 複製到 nnUNet_preprocessed 的資料集資料夾")
        splits = load_json(p)
        if len(splits) != 10 or not 0 <= int(self.fold) < 10:
            raise RuntimeError(f"splits_10fold.json 有 {len(splits)} 折,要求 fold {self.fold}")
        tr, val = splits[int(self.fold)]["train"], splits[int(self.fold)]["val"]
        assert not set(tr) & set(val), "訓練和驗證重疊"
        self.print_to_log_file(f"10-fold split {self.fold}: train {len(tr)} val {len(val)} ({p})")
        return tr, val
