"""Small four-head ResNet-18 and explicitly full-frame input processing."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from baseline_data import encode_labels, file_hash, metrics, write_json
from clean_dataset import CLASSES, LIMBS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def choose_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device(name)


class ImageNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        return (images.float() / 255 - self.mean) / self.std


class LimbModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        width = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(width, len(CLASSES)) for _ in LIMBS])

    def classify(self, features):
        return torch.stack([head(features) for head in self.heads], dim=1)

    def forward(self, images):
        return self.classify(self.backbone(images))

    def configure_stage(self, stage: str) -> None:
        if stage not in {'frozen', 'finetuned'}:
            raise ValueError(f'Unknown model stage {stage}')
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if stage == 'finetuned':
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)
        for parameter in self.heads.parameters():
            parameter.requires_grad_(True)
        # Keep BatchNorm running statistics fixed in both stages. Convolution and
        # affine-layer gradients still work in eval mode for the unfrozen block.
        self.backbone.eval()
        self.heads.train()


class PreviewDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        self.targets = torch.tensor(encode_labels(records), dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['preview_filepath']) as image:
            if image.mode != 'RGB' or image.size != (448, 448):
                raise ValueError('Expected an unannotated RGB 448x448 audit preview')
            pixels = np.array(image, dtype=np.uint8, copy=True).transpose(2, 0, 1).copy()
        return torch.from_numpy(pixels), self.targets[index]


def image_loader(records, batch_size, shuffle=False, seed=42):
    return DataLoader(PreviewDataset(records), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def loss_fn(logits, targets):
    # Equal contribution from the four limbs; no label or class is masked.
    return F.cross_entropy(logits.reshape(-1, len(CLASSES)), targets.reshape(-1))


@torch.inference_mode()
def image_logits(model: LimbModel, records: list[dict], device, batch_size: int) -> torch.Tensor:
    model.to(device).eval()
    normalizer = ImageNormalizer().to(device)
    outputs = []
    for pixels, _ in image_loader(records, batch_size):
        outputs.append(model(normalizer(pixels.to(device))).cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def extract_features(model, records, device, batch_size, run, fingerprint):
    cache = run / 'features.pt'
    source_ids = [r['source_record'] for r in records]
    if cache.exists():
        cache_info = run / 'features.json'
        if not cache_info.is_file():
            raise ValueError('Feature cache metadata missing; remove the incomplete derived cache')
        import json
        if file_hash(cache) != json.loads(cache_info.read_text())['sha256']:
            raise ValueError('Feature cache bytes changed since extraction')
        stored = torch.load(cache, map_location='cpu', weights_only=True)
        if stored['fingerprint'] != fingerprint or stored['source_records'] != source_ids:
            raise ValueError('Feature cache identity mismatch; use a new run directory')
        features = stored['features']
        if features.shape != (len(records), 512) or not torch.isfinite(features).all():
            raise ValueError('Invalid feature cache')
        return features
    model.to(device).eval()
    normalizer = ImageNormalizer().to(device)
    result = []
    start = time.monotonic()
    seen = 0
    for pixels, _ in image_loader(records, batch_size):
        result.append(model.backbone(normalizer(pixels.to(device))).cpu())
        seen += len(pixels)
        if len(result) % 10 == 0:
            print(f'features {seen}/{len(records)} ({time.monotonic()-start:.1f}s)', flush=True)
    features = torch.cat(result)
    if features.shape != (len(records), 512) or not torch.isfinite(features).all():
        raise ValueError('Invalid extracted features')
    torch.save({'features': features, 'source_records': source_ids,
                'fingerprint': fingerprint}, cache)
    write_json(run / 'features.json', {'images': len(records), 'shape': list(features.shape),
               'fingerprint': fingerprint, 'sha256': file_hash(cache),
               'seconds': time.monotonic()-start, 'backbone': 'resnet18 IMAGENET1K_V1',
               'batchnorm': 'eval; running statistics fixed', 'preprocessing': 'full-frame 448 RGB; ImageNet normalization'})
    return features


def metric_from_logits(records, logits):
    return metrics(encode_labels(records), logits.argmax(-1).tolist(),
                   [r['casualty_id'] for r in records])


def save_checkpoint(path, model, epoch, score, fingerprint, stage):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'epoch': epoch, 'validation_macro_f1': score, 'stage': stage,
                'input_fingerprint': fingerprint, 'classes': list(CLASSES),
                'limbs': list(LIMBS), 'architecture': 'resnet18_four_linear_heads',
                'input_size': 448, 'preprocessing': 'full-frame LANCZOS resize and RGB114 padding; no crop; EXIF orientation once',
                'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225]}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, fingerprint: str | None = None):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['classes'] != list(CLASSES) or data['limbs'] != list(LIMBS):
        raise ValueError('Checkpoint class/limb order mismatch')
    if fingerprint is not None and data['input_fingerprint'] != fingerprint:
        raise ValueError('Checkpoint belongs to another prepared input/split')
    model = LimbModel(pretrained=False)
    model.load_state_dict(data['state_dict'], strict=True)
    return model, data
