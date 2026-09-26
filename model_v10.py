"""
model_v10.py — CoSeg V10: 統一模型，整合 7 大策略
====================================================
vs V6 的改動：
  1. LoRA — 取代大面積解凍，可訓練參數 ~40M vs 196M
  2. Domain-Specific BN — 公開/醫院各用一組 BN 統計量
  3. Cross-Slice Attention — z 軸特徵融合，解決 2.5D 碎片化
  4. Attention Gate (from v6)
  5. Deep Supervision (from v6)

用法：
  model = CoSegV10(sam_model,
                   use_lora=True, lora_rank=16,
                   use_domain_bn=True, num_domains=2,
                   use_cross_slice=True, cross_slice_k=5)
  model.set_domain(0)  # 0=公開, 1=醫院
  out = model(x)       # x: (B,3,H,W) 或 (B,K,3,H,W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms


# ==============================================================
# 1. LoRA
# ==============================================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear"""
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original
        # 凍結原始權重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        # A 用 Kaiming，B 用零初始化 → 初始時 LoRA 輸出為 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


def _get_submodule(model, target: str):
    atoms = target.split('.')
    mod = model
    for atom in atoms:
        if atom.isdigit():
            mod = mod[int(atom)]
        else:
            mod = getattr(mod, atom)
    return mod


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0,
                target_keywords: Optional[List[str]] = None) -> int:
    """
    把 model 中所有 attention 相關的 Linear 層替換成 LoRA 版本。
    回傳替換數量。
    """
    if target_keywords is None:
        target_keywords = ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj',
                           'out_proj', 'linear_q', 'linear_k', 'linear_v']

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = name.split('.')[-1]
            if any(kw == last for kw in target_keywords):
                replacements.append(name)

    for name in replacements:
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = _get_submodule(model, parts[0])
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]
        old = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(old, rank, alpha))

    return len(replacements)


# ==============================================================
# 2. Domain-Specific BatchNorm
# ==============================================================
class DomainBatchNorm2d(nn.Module):
    """每個 domain 各自一組 BN (mean/var/γ/β)"""
    def __init__(self, num_features: int, num_domains: int = 2):
        super().__init__()
        self.num_domains = num_domains
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(num_domains)
        ])
        self._domain = 0

    def set_domain(self, d: int):
        self._domain = d

    def forward(self, x):
        return self.bns[self._domain](x)


# ==============================================================
# 3. Cross-Slice Attention
# ==============================================================
class CrossSliceAttention(nn.Module):
    """
    輕量 z 軸注意力：用 global pooling 產生 slice descriptor，
    self-attention 沿 z 軸，再以 channel gating 調制空間特徵。
    gate 初始化為 0 → 初始時等同 identity，不破壞預訓練。
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        # 確保 channels 能被 num_heads 整除
        if channels < num_heads:
            num_heads = 1
        while channels % num_heads != 0:
            num_heads -= 1

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        # Residual gate，初始為 0（identity）
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, K, C, H, W)
        Returns:
            (B, K, C, H, W) — enhanced features
        """
        B, K, C, H, W = feats.shape
        if K == 1:
            return feats  # 單片不做

        # Slice descriptor: global average pooling
        desc = feats.mean(dim=(-2, -1))          # (B, K, C)

        # Self-attention along z
        h = self.norm1(desc)
        attn_out, _ = self.attn(h, h, h)         # (B, K, C)
        desc = desc + attn_out

        # FFN
        desc = desc + self.ffn(self.norm2(desc))  # (B, K, C)

        # Channel gating
        weights = desc.sigmoid().unsqueeze(-1).unsqueeze(-1)  # (B,K,C,1,1)

        # Gated residual: gate=0 → 完全 identity
        return feats + self.gate * (feats * weights - feats)


# ==============================================================
# 4. Attention Gate（支援 DomainBN）
# ==============================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int,
                 use_domain_bn: bool = False, num_domains: int = 2):
        super().__init__()
        BN = lambda c: DomainBatchNorm2d(c, num_domains) if use_domain_bn else nn.BatchNorm2d(c)
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), BN(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), BN(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), BN(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def set_domain(self, d: int):
        for m in self.modules():
            if isinstance(m, DomainBatchNorm2d):
                m.set_domain(d)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ==============================================================
# 5. Deep Supervision Head（支援 DomainBN）
# ==============================================================
class DSHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64,
                 use_domain_bn: bool = False, num_domains: int = 2):
        super().__init__()
        BN = lambda c: DomainBatchNorm2d(c, num_domains) if use_domain_bn else nn.BatchNorm2d(c)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            BN(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True),
        )
        nn.init.constant_(self.head[-1].bias, -4.0)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def set_domain(self, d: int):
        for m in self.modules():
            if isinstance(m, DomainBatchNorm2d):
                m.set_domain(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ==============================================================
# 主模型
# ==============================================================
class CoSegV10(nn.Module):
    def __init__(self, sam_model: SAM2Base, img_size: int = 1024,
                 feat_dims: Optional[Tuple[int, int, int]] = None,
                 sem_cls: int = 6, ins_cls: int = 4,
                 # --- 新增開關 ---
                 use_lora: bool = True, lora_rank: int = 16, lora_alpha: float = 16.0,
                 use_domain_bn: bool = True, num_domains: int = 2,
                 use_cross_slice: bool = True, cross_slice_k: int = 5):
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

        # 特徵維度
        if feat_dims is None:
            feat_dims = (32, 64, 256)
        c_high, c_mid, c_low = feat_dims
        self._feat_dims = feat_dims

        # --- 策略開關 ---
        self.use_lora = use_lora
        self.use_domain_bn = use_domain_bn
        self.use_cross_slice = use_cross_slice
        self.cross_slice_k = cross_slice_k
        self._current_domain = 0

        # (1) LoRA
        if use_lora:
            n = inject_lora(self.model.image_encoder, rank=lora_rank, alpha=lora_alpha)
            print(f"🔧 LoRA: 注入 {n} 個 adapter (rank={lora_rank})")
            # 凍結 backbone 非 LoRA 參數
            for name, p in self.model.image_encoder.named_parameters():
                if 'lora_' not in name:
                    p.requires_grad = False

        # (2) Cross-Slice Attention（每個特徵層各一個）
        if use_cross_slice:
            self.csa_high = CrossSliceAttention(c_high, num_heads=min(4, c_high))
            self.csa_mid = CrossSliceAttention(c_mid, num_heads=min(4, c_mid))
            self.csa_low = CrossSliceAttention(c_low, num_heads=min(8, c_low))
            print(f"🔧 Cross-Slice Attention: K={cross_slice_k}, 3 levels")

        # (3) Attention Gates
        self.ag_mid = AttentionGate(
            F_g=c_low, F_l=c_mid, F_int=max(c_mid // 2, 16),
            use_domain_bn=use_domain_bn, num_domains=num_domains)
        self.ag_high = AttentionGate(
            F_g=c_mid, F_l=c_high, F_int=max(c_high // 2, 16),
            use_domain_bn=use_domain_bn, num_domains=num_domains)

        # (4) Deep Supervision
        self.ds_head_high = DSHead(c_high, max(c_high, 32), use_domain_bn, num_domains)
        self.ds_head_mid = DSHead(c_mid, max(c_mid, 32), use_domain_bn, num_domains)
        self.ds_head_low = DSHead(c_low, 64, use_domain_bn, num_domains)

        if use_domain_bn:
            print(f"🔧 Domain-Specific BN: {num_domains} domains")

        self._print_params()

    def _print_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"🧠 可訓練: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")

    def set_domain(self, domain: int):
        """切換 domain（0=公開, 1=醫院）→ 影響所有 DomainBN"""
        self._current_domain = domain
        for m in self.modules():
            if isinstance(m, DomainBatchNorm2d):
                m.set_domain(domain)

    def _encode_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """(B, 3, H, W) → [high, mid, low] feature maps"""
        B = x.shape[0]
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return feats  # [high@256², mid@128², low@64²]

    def forward(self, x, prob_ins=None, prob_sem=None, img_id=None):
        """
        Args:
            x: (B, 3, H, W) — 單片模式
               (B, K, 3, H, W) — Cross-Slice 模式
        """
        if x.dim() == 5 and self.use_cross_slice:
            return self._forward_cross_slice(x, prob_ins, prob_sem)
        elif x.dim() == 5:
            # 不用 cross-slice 但收到 5D → 只取中間片
            center = x.shape[1] // 2
            x = x[:, center]
        return self._forward_single(x, prob_ins, prob_sem)

    def _forward_cross_slice(self, x, prob_ins, prob_sem):
        """
        x: (B, K, 3, H, W)
        過所有 K 片 backbone → cross-slice attention → 取中間片 → AG/DS/Decoder
        """
        B, K, C_in, H, W = x.shape
        center = K // 2

        # 全部通過 backbone
        x_flat = x.view(B * K, C_in, H, W)
        feats_flat = self._encode_backbone(x_flat)  # 3 × (B*K, C, h, w)

        # Reshape → (B, K, C, h, w)
        feats_5d = [f.view(B, K, *f.shape[1:]) for f in feats_flat]

        # Cross-Slice Attention
        feats_5d[0] = self.csa_high(feats_5d[0])
        feats_5d[1] = self.csa_mid(feats_5d[1])
        feats_5d[2] = self.csa_low(feats_5d[2])

        # 取中間片
        feats = [f[:, center] for f in feats_5d]  # 3 × (B, C, h, w)

        return self._decode(feats, B, prob_ins, prob_sem)

    def _forward_single(self, x, prob_ins, prob_sem):
        """x: (B, 3, H, W)"""
        B = x.shape[0]
        feats = self._encode_backbone(x)
        return self._decode(feats, B, prob_ins, prob_sem)

    def _decode(self, feats, batch_size, prob_ins, prob_sem):
        """AG → DS → SAM2 Decoder"""
        # Attention Gates
        feats_mid_gated = self.ag_mid(g=feats[2], x=feats[1])
        feats_high_gated = self.ag_high(g=feats_mid_gated, x=feats[0])

        # Deep Supervision
        ds_high = self.ds_head_high(feats_high_gated)
        ds_mid = self.ds_head_mid(feats_mid_gated)
        ds_low = self.ds_head_low(feats[2])
        ds_outputs = [ds_high, ds_mid, ds_low]

        # Decoder
        feats_for_decoder = [feats_high_gated, feats_mid_gated, feats[2]]
        self._features = {
            "image_embed": feats_for_decoder[-1],
            "high_res_feats": feats_for_decoder[:-1],
        }

        outputs_mask_ins, outputs_mask_sem = [], []
        outputs_prob_ins, outputs_prob_sem = [], []

        for img_idx in range(batch_size):
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                None, None, None, None, True, img_idx=img_idx)

            if (prob_sem is not None) and (prob_ins is not None):
                mi, ms, _, _ = self._predict(
                    unnorm_coords, labels, unnorm_box,
                    prob_sem[img_idx].unsqueeze(0),
                    prob_ins[img_idx].unsqueeze(0),
                    img_idx=img_idx, first_fwd=False)
                outputs_mask_ins.append(mi.squeeze(0))
                outputs_mask_sem.append(ms.squeeze(0))
            else:
                mi, ms, ks, ki = self._predict(
                    unnorm_coords, labels, unnorm_box,
                    prob_sem, prob_ins,
                    img_idx=img_idx, first_fwd=True)
                outputs_mask_ins.append(mi.squeeze(0))
                outputs_mask_sem.append(ms.squeeze(0))
                outputs_prob_sem.append(ks.squeeze(0))
                outputs_prob_ins.append(ki.squeeze(0))

        if (prob_sem is not None) and (prob_ins is not None):
            return (torch.stack(outputs_mask_ins, 0),
                    torch.stack(outputs_mask_sem, 0),
                    ds_outputs)
        else:
            return (torch.stack(outputs_mask_ins, 0),
                    torch.stack(outputs_mask_sem, 0),
                    torch.stack(outputs_prob_ins, 0),
                    torch.stack(outputs_prob_sem, 0),
                    ds_outputs)

    # === SAM2 Decoder helpers（跟 v6 一樣）===
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
            mask_ins = F.interpolate(low_res_masks_ins, (1024, 1024),
                                     mode="bilinear", align_corners=False)
            mask_sem = F.interpolate(low_res_masks_sem, (1024, 1024),
                                     mode="bilinear", align_corners=False)
        return mask_ins, mask_sem, prob_sem, prob_ins
