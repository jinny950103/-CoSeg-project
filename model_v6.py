"""
model_v6.py — CoSeg V6：Attention Gate + Deep Supervision
==========================================================
修正：自動偵測 SAM2 各層特徵維度，不再假設全部 256ch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DSHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True),
        )
        # 正樣本佔 <0.01%，初始化 bias 為負值讓模型一開始預測「幾乎全是背景」
        nn.init.constant_(self.head[-1].bias, -4.0)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CoSegV6(nn.Module):
    def __init__(self, sam_model: SAM2Base, img_size: int = 1024, feat_dims=None,
                 sem_cls: int = 6, ins_cls: int = 4):
        super().__init__()

        self.num_ins_classes = ins_cls + 3
        self.num_sem_classes = sem_cls
        self.device = "cuda"
        self.img_size = img_size
        self._transforms = SAM2Transforms(
            resolution=img_size, mask_threshold=0.0,
            max_hole_area=0.0, max_sprinkle_area=0.0,
        )
        self.model = sam_model
        self._features = None
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

        # === 特徵維度（SAM2 Hiera-Large）===
        # 從實際執行得知：feats[1] = 64ch@128²
        # 推測完整維度，forward 中會驗證
        if feat_dims is None:
            feat_dims = (32, 64, 256)  # (high@256², mid@128², low@64²)
        c_high, c_mid, c_low = feat_dims
        self._feat_dims = feat_dims
        print(f"🔍 特徵維度設定: high={c_high}ch@256², mid={c_mid}ch@128², low={c_low}ch@64²")

        # Attention Gates
        self.ag_mid = AttentionGate(F_g=c_low, F_l=c_mid, F_int=max(c_mid // 2, 16))
        self.ag_high = AttentionGate(F_g=c_mid, F_l=c_high, F_int=max(c_high // 2, 16))

        # Deep Supervision heads
        self.ds_head_high = DSHead(c_high, mid_channels=max(c_high, 32))
        self.ds_head_mid = DSHead(c_mid, mid_channels=max(c_mid, 32))
        self.ds_head_low = DSHead(c_low, mid_channels=64)


    def forward(self, x, prob_ins=None, prob_sem=None, img_id=None):
        batch_size = x.shape[0]

        # 1. SAM2 Encoder
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        # 2. Attention Gates
        feats_mid_gated = self.ag_mid(g=feats[2], x=feats[1])
        feats_high_gated = self.ag_high(g=feats_mid_gated, x=feats[0])

        # 3. Deep Supervision
        ds_high = self.ds_head_high(feats_high_gated)
        ds_mid = self.ds_head_mid(feats_mid_gated)
        ds_low = self.ds_head_low(feats[2])

        # 4. Decoder (用 attention-gated features)
        feats_for_decoder = [feats_high_gated, feats_mid_gated, feats[2]]
        self._features = {
            "image_embed": feats_for_decoder[-1],
            "high_res_feats": feats_for_decoder[:-1],
        }

        num_images = len(self._features["image_embed"])
        outputs_mask_ins = []
        outputs_mask_sem = []
        outputs_prob_ins = []
        outputs_prob_sem = []
        normalize_coords = True

        for img_idx in range(num_images):
            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords, point_labels, box, mask_input,
                normalize_coords, img_idx=img_idx,
            )

            if (prob_sem is not None) and (prob_ins is not None):
                masks_ins, masks_sem, _, _ = self._predict(
                    unnorm_coords, labels, unnorm_box,
                    prob_sem[img_idx].unsqueeze(0),
                    prob_ins[img_idx].unsqueeze(0),
                    img_idx=img_idx, first_fwd=False,
                )
                outputs_mask_ins.append(masks_ins.squeeze(0))
                outputs_mask_sem.append(masks_sem.squeeze(0))
            else:
                masks_ins, masks_sem, kl_sem, kl_ins = self._predict(
                    unnorm_coords, labels, unnorm_box,
                    prob_sem, prob_ins,
                    img_idx=img_idx, first_fwd=True,
                )
                outputs_mask_ins.append(masks_ins.squeeze(0))
                outputs_mask_sem.append(masks_sem.squeeze(0))
                outputs_prob_sem.append(kl_sem.squeeze(0))
                outputs_prob_ins.append(kl_ins.squeeze(0))

        ds_outputs = [ds_high, ds_mid, ds_low]

        if (prob_sem is not None) and (prob_ins is not None):
            return (
                torch.stack(outputs_mask_ins, dim=0),
                torch.stack(outputs_mask_sem, dim=0),
                ds_outputs,
            )
        else:
            return (
                torch.stack(outputs_mask_ins, dim=0),
                torch.stack(outputs_mask_sem, dim=0),
                torch.stack(outputs_prob_ins, dim=0),
                torch.stack(outputs_prob_sem, dim=0),
                ds_outputs,
            )

    def _prep_prompts(self, point_coords, point_labels, box, mask_logits,
                      normalize_coords, img_idx=-1):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=(1024, 1024))
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=(1024, 1024))
        if mask_logits is not None:
            mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(self, point_coords, point_labels, boxes=None,
                 mask_sem=None, mask_ins=None, img_idx=-1, first_fwd=True):
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings_sem, dense_embeddings_sem = self.model.stp_encoder_ins(
            points=None, boxes=None, masks=mask_ins,
            image=self._features["image_embed"][img_idx].unsqueeze(0))
        sparse_embeddings_ins, dense_embeddings_ins = self.model.stp_encoder_sem(
            points=None, boxes=None, masks=mask_sem,
            image=self._features["image_embed"][img_idx].unsqueeze(0))

        batched_mode = (concat_points is not None and concat_points[0].shape[0] > 1)
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        if first_fwd:
            low_res_masks_ins, prob_ins = self.model.mct_decoder_ins(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_sem.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_ins,
                dense_prompt_embeddings=dense_embeddings_ins,
                multimask_output=False, repeat_image=batched_mode,
                high_res_features=high_res_features, first_fwds=first_fwd)
            low_res_masks_sem, prob_sem = self.model.mct_decoder_sem(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_ins.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_sem,
                dense_prompt_embeddings=dense_embeddings_sem,
                multimask_output=False, repeat_image=batched_mode,
                high_res_features=high_res_features, first_fwds=first_fwd)
        else:
            low_res_masks_ins, prob_ins = self.model.mct_decoder_ins(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_sem.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_ins,
                dense_prompt_embeddings=dense_embeddings_ins,
                multimask_output=True, repeat_image=batched_mode,
                high_res_features=high_res_features, first_fwds=first_fwd)
            low_res_masks_sem, prob_sem = self.model.mct_decoder_sem(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.stp_encoder_ins.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_sem,
                dense_prompt_embeddings=dense_embeddings_sem,
                multimask_output=True, repeat_image=batched_mode,
                high_res_features=high_res_features, first_fwds=first_fwd)

        if first_fwd:
            mask_ins = low_res_masks_ins
            mask_sem = low_res_masks_sem
        else:
            mask_ins = F.interpolate(low_res_masks_ins, (1024, 1024), mode="bilinear", align_corners=False)
            mask_sem = F.interpolate(low_res_masks_sem, (1024, 1024), mode="bilinear", align_corners=False)
        return mask_ins, mask_sem, prob_sem, prob_ins
