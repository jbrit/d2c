"""Independent whole-image classifiers, combined in a fixed anatomical order."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from baseline_data import file_hash
from clean_dataset import CLASSES, LIMBS


class SingleLimbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, len(CLASSES))

    @classmethod
    def from_shared(cls, shared, index):
        model = cls()
        model.backbone.load_state_dict(shared.backbone.state_dict())
        model.head.load_state_dict(shared.heads[index].state_dict())
        return model

    def forward(self, images):
        return self.head(self.backbone(images))

    def configure_stage(self, stage):
        if stage not in {'frozen', 'finetuned'}:
            raise ValueError(f'Unknown stage: {stage}')
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if stage == 'finetuned':
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.backbone.eval()  # Fixed BatchNorm statistics, including layer4.
        self.head.train()


class PerLimbPipeline(nn.Module):
    def __init__(self, models):
        super().__init__()
        if len(models) != len(LIMBS):
            raise ValueError('Exactly four independent limb models are required')
        self.models = nn.ModuleList(models)

    def forward(self, images):
        return torch.stack([model(images) for model in self.models], dim=1)


def save_single(path, model, limb, stage, epoch, score, fingerprint):
    temporary = path.with_suffix('.tmp')
    torch.save({'architecture': 'resnet18_single_linear_head', 'limb': limb,
                'classes': list(CLASSES), 'input_fingerprint': fingerprint,
                'stage': stage, 'epoch': epoch, 'validation_macro_f1': score,
                'input_size': 448,
                'preprocessing': 'EXIF once; whole-frame RGB LANCZOS resize, RGB114 padding; ImageNet normalization',
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, temporary)
    temporary.replace(path)


def load_single(path, limb, fingerprint=None):
    info = torch.load(path, map_location='cpu', weights_only=True)
    if (info['architecture'] != 'resnet18_single_linear_head' or
            info['limb'] != limb or info['classes'] != list(CLASSES)):
        raise ValueError('Single-limb checkpoint identity mismatch')
    if fingerprint is not None and info['input_fingerprint'] != fingerprint:
        raise ValueError('Checkpoint input/split mismatch')
    model = SingleLimbModel()
    model.load_state_dict(info['state_dict'], strict=True)
    return model, info


def load_pipeline(path: Path):
    info = json.loads(path.read_text())
    if (info['architecture'] != 'four_independent_resnet18' or
            info['limbs'] != list(LIMBS) or info['classes'] != list(CLASSES) or
            len(info['checkpoints']) != len(LIMBS)):
        raise ValueError('Pipeline limb/class order mismatch')
    models = []
    for limb, entry in zip(LIMBS, info['checkpoints']):
        checkpoint = (path.parent / entry['file']).resolve()
        if not checkpoint.is_relative_to(path.parent.resolve()):
            raise ValueError('Pipeline checkpoint must be inside the bundle directory')
        if entry['limb'] != limb or file_hash(checkpoint) != entry['sha256']:
            raise ValueError('Pipeline checkpoint limb/hash mismatch')
        model, saved = load_single(checkpoint, limb, info['input_fingerprint'])
        if saved['stage'] != info['stage']:
            raise ValueError('Pipeline checkpoint stage mismatch')
        models.append(model)
    return PerLimbPipeline(models), info
