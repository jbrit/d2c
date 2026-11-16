#!/usr/bin/env python3
"""Compare four independent ResNet-18 classifiers with an existing shared run."""
from __future__ import annotations

import argparse
import copy
import html
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from audit_images import load_image, make_preview
from baseline import PROJECT, fingerprint, load_run, write_predictions
from baseline_data import encode_labels, file_hash, metrics, safe_run_directory, write_json
from baseline_model import (ImageNormalizer, LimbModel, choose_device, image_loader,
                            image_logits, metric_from_logits, seed_everything)
from clean_dataset import CLASSES, LIMBS
from per_limb_model import SingleLimbModel, load_pipeline, load_single, save_single


DEFAULT_RUN = PROJECT / 'runs/per_limb_v1'


def tree_hashes(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}


def limb_score(records, logits, index):
    # Reuse the established metric policy, retaining only this limb's result.
    truth = encode_labels(records)
    predicted = [row.copy() for row in truth]
    for row, label in zip(predicted, logits.argmax(-1).tolist()):
        row[index] = label
    return metrics(truth, predicted)['by_limb'][LIMBS[index]]['macro_f1']


def prepare(args):
    reference = args.reference_run.resolve()
    run = args.run_dir.resolve()
    records, context = load_run(reference)
    safe_run_directory(run, Path(context['dataset_root']))
    if run == reference or run.is_relative_to(reference) or reference.is_relative_to(run):
        raise ValueError('Experiment and reference directories must be separate')
    if run.exists():
        raise FileExistsError(f'Run already exists: {run}')
    # Check the exact implementation that produced the cached fixed features.
    environment = json.loads((reference / 'environment.json').read_text())
    for name, sha in environment['code_sha256'].items():
        if file_hash(PROJECT / 'scripts' / name) != sha:
            raise ValueError(f'Reference implementation changed: {name}')
    packages = {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'numpy', 'Pillow']}
    if packages != environment['packages']:
        raise ValueError('Use the same package versions as the reference run')
    source_hashes = tree_hashes(reference)
    config = json.loads((reference / 'training_config.json').read_text())
    run.mkdir(parents=True)
    for name in ['inputs.json', 'records.json', 'split_manifest.csv', 'split_summary.json',
                 'majority_metrics.json', 'majority_predictions.jsonl', 'requirements.lock']:
        shutil.copyfile(reference / name, run / name)
    config.update({'architecture': 'four independent ResNet-18 backbones with one Linear(512,4) head each',
                   'loss': 'unweighted cross entropy for that limb only',
                   'model_selection': 'each limb/stage: highest supported-class validation macro-F1; earliest tie',
                   'initialization': 'same ImageNet weights and corresponding initial head as shared run',
                   'comparison_scope': 'same split and training budget per model; 4x backbone capacity/compute; independent checkpoint selection',
                   'stage_selection': 'compare frozen and finetuned bundles separately; no mixing stages or shared/per-limb models'})
    write_json(run / 'training_config.json', config)
    write_json(run / 'reference.json', {'directory': str(reference), 'file_sha256': source_hashes,
               'input_fingerprint': context['input_fingerprint'], 'exact_split_copied': True,
               'training_config_sha256': file_hash(run / 'training_config.json')})
    code = run / 'code'
    code.mkdir()
    names = ['per_limb.py', 'per_limb_model.py', 'baseline.py', 'baseline_data.py',
             'baseline_model.py', 'audit_images.py', 'clean_dataset.py']
    hashes = {}
    for name in names:
        source = PROJECT / 'scripts' / name
        shutil.copyfile(source, code / name)
        hashes[name] = file_hash(source)
    write_json(run / 'environment.json', {'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'packages': packages, 'code_sha256': hashes})
    print(f'Copied exact reference split: {sum(r["split"]=="train" for r in records)} train / '
          f'{sum(r["split"]=="validation" for r in records)} validation images', flush=True)


def checked_run(run, check_code=True):
    records, context = load_run(run)
    reference = json.loads((run / 'reference.json').read_text())
    root = Path(reference['directory'])
    if tree_hashes(root) != reference['file_sha256']:
        raise ValueError('Reference run changed since experiment preparation')
    if context['input_fingerprint'] != reference['input_fingerprint']:
        raise ValueError('Reference input fingerprint mismatch')
    if file_hash(run / 'training_config.json') != reference['training_config_sha256']:
        raise ValueError('Experiment training configuration was edited; use a new run')
    if check_code:
        environment = json.loads((run / 'environment.json').read_text())
        if {p: importlib.metadata.version(p) for p in environment['packages']} != environment['packages']:
            raise ValueError('Experiment package versions changed')
        for name, sha in environment['code_sha256'].items():
            if file_hash(PROJECT / 'scripts' / name) != sha:
                raise ValueError(f'Experiment code changed: {name}; use a new run')
    config = json.loads((run / 'training_config.json').read_text())
    return records, context, root, config


def initialization(reference, context, records, config):
    info = json.loads((reference / 'features.json').read_text())
    pretrained = json.loads((reference / 'pretrained.json').read_text())
    feature_path = reference / 'features.pt'
    if file_hash(feature_path) != info['sha256']:
        raise ValueError('Fixed-feature cache hash mismatch')
    cache = torch.load(feature_path, map_location='cpu', weights_only=True)
    expected = fingerprint({'input_fingerprint': context['input_fingerprint'],
                            'pretrained_sha256': pretrained['sha256'],
                            'model_code_sha256': file_hash(PROJECT / 'scripts/baseline_model.py'),
                            'torch': torch.__version__,
                            'torchvision': importlib.metadata.version('torchvision')})
    if (cache['fingerprint'] != expected or expected != info['fingerprint'] or
            cache['source_records'] != [r['source_record'] for r in records] or
            cache['features'].shape != (len(records), 512) or
            not torch.isfinite(cache['features']).all()):
        raise ValueError('Fixed-feature identity mismatch')
    weights = reference / 'pretrained_weights/checkpoints/resnet18-f37072fd.pth'
    if file_hash(weights) != pretrained['sha256']:
        raise ValueError('Pretrained weight hash mismatch')
    seed_everything(config['seed'])
    # Identical construction/RNG consumption to LimbModel(pretrained=True), with
    # explicit local weight loading to avoid downloads or writes to the old run.
    shared = LimbModel(pretrained=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    state.pop('fc.weight'); state.pop('fc.bias')
    shared.backbone.load_state_dict(state, strict=True)
    return shared, cache['features']


def smoke(run, shared, features, records, reference, context):
    old = json.loads((reference / 'smoke.json').read_text())
    indices = {r['source_record']: i for i, r in enumerate(records)}
    selected = [indices[source] for source in old['source_records']]
    if any(records[i]['split'] != 'train' for i in selected):
        raise ValueError('Smoke images must all be training images')
    targets = torch.tensor(encode_labels(records))[selected]
    results = {}
    for index, limb in enumerate(LIMBS):
        head = copy.deepcopy(shared.heads[index])
        optimizer = AdamW(head.parameters(), lr=.02, weight_decay=0)
        initial = float(F.cross_entropy(head(features[selected]), targets[:, index]).detach())
        for step in range(500):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features[selected]), targets[:, index])
            loss.backward(); optimizer.step()
            with torch.no_grad():
                output = head(features[selected])
                final = float(F.cross_entropy(output, targets[:, index]))
                accuracy = float((output.argmax(-1) == targets[:, index]).float().mean())
            if accuracy == 1 and final < .02:
                break
        results[limb] = {'initial_loss': initial, 'final_loss': final, 'accuracy': accuracy,
                         'steps': step + 1, 'passed': accuracy == 1 and final < .05}
    write_json(run / 'smoke.json', {'input_fingerprint': context['input_fingerprint'],
               'source_records': old['source_records'], 'limbs': results,
               'disposable_heads_only': True, 'passed': all(x['passed'] for x in results.values())})
    if not all(x['passed'] for x in results.values()):
        raise RuntimeError('Independent head smoke check failed')


def train_stage(run, model, records, features, index, stage, config, context, device):
    limb = LIMBS[index]
    path = run / f'{limb}_{stage}.pt'
    if path.exists():
        return load_single(path, limb, context['input_fingerprint'])[0]
    frozen = stage == 'frozen'
    seed_everything(config['seed'])
    model.to('cpu' if frozen else device).configure_stage(stage)
    train_ids = [i for i, r in enumerate(records) if r['split'] == 'train']
    val_ids = [i for i, r in enumerate(records) if r['split'] == 'validation']
    train = [records[i] for i in train_ids]
    val = [records[i] for i in val_ids]
    labels = torch.tensor(encode_labels(records), dtype=torch.long)
    normalizer = ImageNormalizer().to(device)
    if frozen:
        loader = DataLoader(TensorDataset(features[train_ids], labels[train_ids, index]),
                            batch_size=config['feature_batch_size'], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']))
        optimizer = AdamW(model.head.parameters(), lr=config['head_lr'], weight_decay=config['weight_decay'])
    else:
        loader = image_loader(train, config['batch_size'], shuffle=True, seed=config['seed'])
        optimizer = AdamW([{'params': model.backbone.layer4.parameters(), 'lr': config['backbone_lr']},
                           {'params': model.head.parameters(), 'lr': config['finetune_head_lr']}],
                          weight_decay=config['weight_decay'])
    max_epochs = config['head_epochs' if frozen else 'finetune_epochs']
    patience = config['head_patience' if frozen else 'finetune_patience']
    buffers = {k: v.detach().cpu().clone() for k, v in model.backbone.named_buffers()}
    fixed = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()
             if frozen or not k.startswith('layer4.')}
    best = -1.; best_epoch = 0; best_state = None; stale = 0
    start = time.monotonic()
    with (run / f'{limb}_{stage}_history.jsonl').open('w') as history:
        for epoch in range(1, max_epochs + 1):
            model.configure_stage(stage)
            total = 0.; correct = 0; seen = 0
            for x, y in loader:
                if not frozen:
                    x = normalizer(x.to(device)); y = y[:, index].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.head(x) if frozen else model(x)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                if not frozen:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                optimizer.step()
                total += float(loss.detach()) * len(y)
                correct += int((logits.detach().argmax(-1) == y).sum()); seen += len(y)
            with torch.no_grad():
                logits = model.head(features[val_ids]) if frozen else image_logits(model, val, device, config['batch_size'])
                val_loss = float(F.cross_entropy(logits.cpu(), labels[val_ids, index]))
            score = limb_score(val, logits, index)
            item = {'epoch': epoch, 'train_loss': total / seen, 'train_label_accuracy': correct / seen,
                    'validation_loss': val_loss, 'validation_macro_f1': score, 'seconds': time.monotonic() - start}
            history.write(json.dumps(item) + '\n'); history.flush()
            if score > best + 1e-8:
                best = score; best_epoch = epoch; stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if not frozen or epoch == 1 or epoch % 10 == 0:
                print(f'{limb} {stage} epoch {epoch}: train loss {total/seen:.4f}, val macro-F1 {score:.4f}, best {best:.4f}', flush=True)
            if stale >= patience:
                break
    current = model.backbone.state_dict()
    if any(not torch.equal(v, current[k].cpu()) for k, v in fixed.items()):
        raise RuntimeError('Frozen backbone parameters/buffers changed')
    if any(not torch.equal(v, dict(model.backbone.named_buffers())[k].cpu()) for k, v in buffers.items()):
        raise RuntimeError('BatchNorm statistics changed')
    model.load_state_dict(best_state)
    save_single(path, model, limb, stage, best_epoch, best, context['input_fingerprint'])
    write_json(run / f'{limb}_{stage}_training.json', {'best_epoch': best_epoch, 'epochs_completed': epoch,
               'best_validation_macro_f1': best, 'seconds': time.monotonic() - start,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'batchnorm_statistics_fixed': True, 'frozen_parameters_verified': True,
               'early_stopping_triggered': stale >= patience,
               'device': 'cpu' if frozen else str(device),
               'warm_start': 'ImageNet + corresponding initial shared head' if frozen else f'{limb}_frozen.pt'})
    return model.cpu()


def train(args):
    records, context, reference, config = checked_run(args.run_dir)
    device = choose_device(args.device)
    execution = {'device': str(device), 'seed': config['seed'], 'cpu_threads': 4}
    execution_path = args.run_dir / 'execution.json'
    if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
        raise ValueError('Execution settings changed; use the original device or a new run')
    write_json(execution_path, execution)
    shared, features = initialization(reference, context, records, config)
    smoke(args.run_dir, shared, features, records, reference, context)
    for index, limb in enumerate(LIMBS):
        model = SingleLimbModel.from_shared(shared, index)
        model = train_stage(args.run_dir, model, records, features, index, 'frozen', config, context, device)
        model = train_stage(args.run_dir, model, records, features, index, 'finetuned', config, context, device)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    for stage in ['frozen', 'finetuned']:
        write_json(args.run_dir / ('pipeline.json' if stage == 'finetuned' else 'frozen_pipeline.json'), {
            'architecture': 'four_independent_resnet18', 'stage': stage,
            'limbs': list(LIMBS), 'classes': list(CLASSES), 'input_fingerprint': context['input_fingerprint'],
            'dataset_root': context['dataset_root'], 'input_size': 448,
            'preprocessing': context['preprocessing'], 'probabilities_are_calibrated': False,
            'checkpoints': [{'limb': limb, 'file': f'{limb}_{stage}.pt',
                             'sha256': file_hash(args.run_dir / f'{limb}_{stage}.pt')} for limb in LIMBS]})
    checked_run(args.run_dir)


def evaluate(args):
    records, context, reference, config = checked_run(args.run_dir)
    val = [r for r in records if r['split'] == 'validation']
    ids = [r['source_record'] for r in val]
    previous = json.loads((reference / 'comparison.json').read_text())
    if previous['all_models_use_identical_validation_source_records'] != ids:
        raise ValueError('Reference evaluation source records differ')
    results = {'majority': previous['models']['majority'],
               'shared_frozen': previous['models']['frozen'],
               'shared_finetuned': previous['models']['finetuned']}
    device = choose_device(args.device)
    for stage in ['frozen', 'finetuned']:
        name = f'per_limb_{stage}'
        pipeline = args.run_dir / ('pipeline.json' if stage == 'finetuned' else 'frozen_pipeline.json')
        model, info = load_pipeline(pipeline)
        start = time.monotonic()
        logits = image_logits(model, val, device, config['batch_size'])
        result = metric_from_logits(val, logits)
        result.update({'pipeline_sha256': file_hash(pipeline), 'evaluation_device': str(device),
                       'evaluation_seconds': time.monotonic() - start,
                       'parameter_count': sum(p.numel() for p in model.parameters()),
                       'checkpoint_epochs': {e['limb']: torch.load(args.run_dir / e['file'], map_location='cpu', weights_only=True)['epoch']
                                             for e in info['checkpoints']}})
        results[name] = result
        write_json(args.run_dir / f'{name}_metrics.json', result)
        write_predictions(args.run_dir / f'{name}_predictions.jsonl', val, logits.argmax(-1).tolist(), logits.softmax(-1).tolist())
        print(f'{name}: macro-F1 {result["mean_limb_macro_f1"]:.4f}, label accuracy {result["label_accuracy"]:.4f}, all-four {result["exact_match_accuracy"]:.4f}', flush=True)
        model.cpu(); del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    comparison = {'input_fingerprint': context['input_fingerprint'],
                  'split_manifest_sha256': context['split_manifest_sha256'],
                  'reference_run': str(reference), 'models': results,
                  'all_models_use_identical_validation_source_records': ids,
                  'selection_policy': config['model_selection'],
                  'evaluation_scope': 'development casualty holdout used for checkpoint selection; not an untouched test set',
                  'comparison_limits': 'one split/seed; 4x backbone capacity and independent per-limb epoch selection; no class weighting or augmentation; unresolved annotations retained'}
    write_json(args.run_dir / 'comparison.json', comparison)
    report(args.run_dir, comparison)
    checked_run(args.run_dir)
    write_json(args.run_dir / 'preservation.json', {'canonical_inputs_and_inventory_unchanged': True,
               'reference_run_all_file_hashes_unchanged': True, 'split_manifest_copied_byte_for_byte': True,
               'labels_modified': 0, 'videos_used': 0})


def report(run, comparison):
    esc = html.escape
    fmt = lambda x: 'n.a.' if x is None else f'{100*x:.1f}%'
    rows = []; details = []
    lines = ['Independent per-limb comparison', '', 'Model / macro-F1 / label accuracy / all four correct']
    for name, result in comparison['models'].items():
        values = [name, f'{result["mean_limb_macro_f1"]:.3f}', fmt(result['label_accuracy']),
                  fmt(result['exact_match_accuracy']), fmt(result['casualty_mean_label_accuracy'])]
        rows.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in values) + '</tr>')
        lines.append(' / '.join(values))
        class_rows = []; matrices = []
        for limb in LIMBS:
            data = result['by_limb'][limb]
            for label, values in data['classes'].items():
                cells = [limb, label, str(values['support']), str(values['predicted']),
                         fmt(values['precision']), fmt(values['recall']), fmt(values['f1'])]
                class_rows.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in cells) + '</tr>')
            matrix = data['confusion_matrix_true_rows_predicted_columns']
            body = ''.join('<tr><th>' + CLASSES[i] + '</th>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
                           for i, row in enumerate(matrix))
            matrices.append(f'<h4>{limb}: macro-F1 {data["macro_f1"]:.3f}</h4><table><tr><th>True / predicted</th>'
                            + ''.join(f'<th>{c}</th>' for c in CLASSES) + '</tr>' + body + '</table>')
        link = f'<p><a href="{name}_metrics.json">Metrics</a> · <a href="{name}_predictions.jsonl">Predictions</a></p>' if name.startswith('per_limb') else ''
        details.append(f'<section><h2>{name}</h2><details><summary>Class precision, recall and F1</summary><div class="scroll"><table>'
                       '<tr><th>Limb</th><th>Class</th><th>True</th><th>Predicted</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
                       + ''.join(class_rows) + '</table></div></details><details><summary>Limb scores and confusion matrices</summary>'
                       + ''.join(matrices) + '</details>' + link + '</section>')
    note = ('All models use the identical 758 training images / 50 casualties and 194 validation images / 12 casualties. '
            'Each independent model has its own ImageNet ResNet-18 backbone and Linear(512,4) classifier, sees the full image, '
            'and trains only on its own limb label. Heads are fitted first; layer4 and that head are then fine-tuned. '
            'Each limb selects its own epoch. The fine-tuned pipeline combines all four fine-tuned models without averaging or cross-limb voting. '
            'The four models have about four times the backbone parameters and computation of the shared model. '
            'No class weighting, augmentation, crops, flips, annotation changes or video frames. '
            'Macro-F1 omits classes without true support within each limb: left-arm amputation has zero validation support. '
            'These are development results on one split/seed; the validation set selected checkpoints.')
    lines += ['', note, '', 'Canonical sources, labels, and shared baseline run remain unchanged.']
    (run / 'summary.txt').write_text('\n'.join(lines) + '\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Independent per-limb comparison</title><style>body{font:16px/1.5 system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#17212b;background:#f5f7f9}section,.intro{background:white;border:1px solid #d6dce2;border-radius:8px;padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #d6dce2}details{margin:16px 0}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}a{color:#075faa}</style>'
    page += '<h1>Shared versus independent limb models</h1><div class="intro"><table><tr><th>Model</th><th>Mean limb macro-F1</th><th>Label accuracy</th><th>All four correct</th><th>Mean casualty accuracy</th></tr>'
    page += ''.join(rows) + '</table><p>' + esc(note) + '</p><p><a href="pipeline.json">Fine-tuned pipeline manifest</a> · <a href="comparison.json">Comparison JSON</a> · <a href="split_manifest.csv">Exact split</a> · <a href="training_config.json">Training configuration</a> · <a href="summary.txt">Text summary</a></p></div>'
    page += ''.join(details) + '</html>'
    (run / 'index.html').write_text(page)


def predict(args):
    model, info = load_pipeline(args.pipeline.resolve())
    device = choose_device(args.device)
    model.to(device).eval()
    original, metadata = load_image(args.image.resolve())
    try:
        preview, _ = make_preview(original, 448)
    finally:
        original.close()
    try:
        pixels = torch.from_numpy(np.array(preview, dtype=np.uint8, copy=True).transpose(2, 0, 1).copy()).unsqueeze(0)
    finally:
        preview.close()
    with torch.inference_mode():
        probabilities = model(ImageNormalizer().to(device)(pixels.to(device))).softmax(-1)[0].cpu()
    result = {'image': str(args.image.resolve()), 'image_sha256': metadata['image_sha256'],
              'pipeline': str(args.pipeline.resolve()), 'pipeline_sha256': file_hash(args.pipeline),
              'stage': info['stage'], 'input_fingerprint': info['input_fingerprint'],
              'prediction': {limb: CLASSES[int(probabilities[i].argmax())] for i, limb in enumerate(LIMBS)},
              'probabilities': {limb: dict(zip(CLASSES, probabilities[i].tolist())) for i, limb in enumerate(LIMBS)},
              'probabilities_are_calibrated': False}
    if args.json_output:
        target = args.json_output.resolve()
        source = Path(info['dataset_root']).resolve()
        if target.exists() or target.is_relative_to(source) or target.is_relative_to((PROJECT / 'train').resolve()):
            raise ValueError('Prediction output must be a new file outside the dataset tree')
        write_json(target, result)
    print(json.dumps(result, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--reference-run', type=Path, default=PROJECT / 'runs/baseline_v1')
    prep.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    prep.set_defaults(function=prepare)
    for command, function in [('train', train), ('evaluate', evaluate)]:
        sub = subs.add_parser(command)
        sub.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
        sub.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
        sub.set_defaults(function=function)
    sub = subs.add_parser('predict')
    sub.add_argument('--pipeline', type=Path, default=DEFAULT_RUN / 'pipeline.json')
    sub.add_argument('--image', type=Path, required=True)
    sub.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    sub.add_argument('--json-output', type=Path)
    sub.set_defaults(function=predict)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    args.function(args)
