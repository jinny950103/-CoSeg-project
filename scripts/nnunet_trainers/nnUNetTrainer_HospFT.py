"""Hospital-only fine-tuning trainer.

Starts from the Dataset504 mixed model (passed with -pretrained_weights) and continues on
the hospital cases alone, so the final model commits to the hospital label convention: a
centreline dilated to a fixed 1.2mm radius, rather than the public set's hand-traced
outline. In the mixed run hospital cases were only ~13% of sampled patches.

A tenth of the default learning rate and a short schedule, because 27 training cases
overfit fast and the pretrained weights are already close.

nnU-Net only finds trainers inside its own package, so this file is copied to
nnunetv2/training/nnUNetTrainer/variants/ by scripts/nano4/20_prepare_505_hospft.sh.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_HospFT(nnUNetTrainer):
    # The signature must match the base class exactly: nnU-Net records the constructor
    # arguments by name to rebuild the trainer at inference time.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 100
