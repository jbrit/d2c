"""Manifest lineage, grouped splits and metrics shared by the baseline commands."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from clean_dataset import CLASSES, LIMBS, local_path, read_manifest, snapshot


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream, strict=True))


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_run_directory(run: Path, root: Path) -> None:
    if run.is_relative_to(root) or root.is_relative_to(run):
        raise ValueError('Run directory must be outside and not contain the dataset root')


def audited_records(root: Path) -> tuple[list[dict], dict]:
    """Cross-check every passing image against the canonical CSV and audit identity."""
    audit = root / 'cleaned/image_audit'
    files = [root / 'train_dataset.csv', root / 'cleaned/train_dataset_cleaned.csv',
             audit / 'image_dataset_validated.csv', audit / 'image_validation.csv',
             audit / 'image_audit_summary.json']
    identity = {str(p): file_hash(p) for p in files}
    summary = json.loads(files[-1].read_text())
    if identity[str(files[0])] != summary['source_csv_sha256']:
        raise ValueError('Canonical CSV no longer matches the image audit')
    if identity[str(files[1])] != summary['manifest_sha256']:
        raise ValueError('Cleaned CSV no longer matches the image audit')
    _, canonical, _ = read_manifest(files[0])
    _, validated, _ = read_manifest(files[2])
    audits = read_csv(files[3])
    passed = [r for r in audits if r['status'] == 'ok']
    if len(passed) != len(validated) or len(passed) != summary['images_passed']:
        raise ValueError('Passing audit and validated manifest counts disagree')
    records = []
    seen = set()
    for a, v in zip(passed, validated):
        i = int(a['source_record'])
        if i in seen or not 1 <= i <= len(canonical):
            raise ValueError('Duplicate or invalid source_record in image audit')
        seen.add(i)
        c = canonical[i - 1]
        if c['media_format'] != 'image' or v['media_format'] != 'image':
            raise ValueError('Non-image row in image manifest')
        path = local_path(c, root)
        if str(path) != a['filepath'] or str(path) != v['filepath']:
            raise ValueError(f'Path lineage mismatch for source {i}')
        for column in ('casualty_id', *LIMBS):
            if not c[column] == v[column] == a[column]:
                raise ValueError(f'Label/identity lineage mismatch: source {i}, {column}')
        preview = (audit / a['preview_filepath']).resolve()
        if not preview.is_relative_to((audit / 'previews').resolve()) or preview.suffix != '.png':
            raise ValueError('Only unannotated audit previews can be training inputs')
        if int(a['preview_size']) != 448:
            raise ValueError('This baseline requires the existing 448-pixel full-frame previews')
        records.append({'source_record': i, 'casualty_id': c['casualty_id'],
                        'filepath': str(path), 'image_sha256': a['image_sha256'],
                        'preview_filepath': str(preview),
                        **{limb: c[limb] for limb in LIMBS}})
    return records, identity


def grouped_split(records: list[dict], seed: int = 42, fraction: float = .2,
                  candidates: int = 2000) -> tuple[dict[int, str], dict]:
    """Choose a split using label counts only, never fitted model performance.

    Sole-casualty class support stays in training. Other supported limb/classes
    must occur in both sets. Random candidate casualty subsets are ranked by
    image-fraction and per-limb/class-fraction deviation from the target.
    """
    if not 0 < fraction < 1 or candidates < 1:
        raise ValueError('Expected 0 < validation fraction < 1 and positive candidates')
    groups = defaultdict(list)
    support = defaultdict(set)
    for r in records:
        groups[r['casualty_id']].append(r)
        for limb in LIMBS:
            support[(limb, r[limb])].add(r['casualty_id'])
    if len(groups) < 3:
        raise ValueError('At least three casualty groups are required')
    train_only = set().union(*(v for v in support.values() if len(v) == 1))
    eligible = sorted(set(groups) - train_only)
    n_val = max(1, round(len(groups) * fraction))
    if n_val >= len(groups) or n_val > len(eligible):
        raise ValueError('Insufficient independent casualty support for requested split')
    dimensions = sorted(support)
    totals = Counter((limb, r[limb]) for r in records for limb in LIMBS)
    vectors = {g: Counter((limb, r[limb]) for r in rows for limb in LIMBS)
               for g, rows in groups.items()}
    rng = random.Random(seed)
    best = None
    for _ in range(candidates):
        selected = set(rng.sample(eligible, n_val))
        val = sum((vectors[g] for g in sorted(selected)), Counter())
        if any(len(support[d]) >= 2 and not 0 < val[d] < totals[d] for d in dimensions):
            continue
        size = sum(len(groups[g]) for g in selected)
        score = 2 * abs(size / len(records) - fraction)
        splitable = [d for d in dimensions if len(support[d]) >= 2]
        score += sum(abs(val[d] / totals[d] - fraction) for d in splitable) / len(splitable)
        key = (score, tuple(sorted(selected)))
        if best is None or key < best[0]:
            best = (key, selected)
    if best is None:
        raise ValueError('No candidate split preserves all independently supported classes')
    validation_groups = best[1]
    assignment = {r['source_record']: 'validation' if r['casualty_id'] in validation_groups else 'train'
                  for r in records}
    hash_splits = defaultdict(set)
    hash_records = defaultdict(list)
    for r in records:
        hash_splits[r['image_sha256']].add(assignment[r['source_record']])
        hash_records[r['image_sha256']].append(r['source_record'])
    if any(len(v) > 1 for v in hash_splits.values()):
        raise ValueError('Exact duplicate image bytes cross the split; reconcile their casualty grouping')
    coverage = {}
    for split in ('train', 'validation'):
        subset = [r for r in records if assignment[r['source_record']] == split]
        coverage[split] = {'images': len(subset),
                           'casualties': sorted({r['casualty_id'] for r in subset}),
                           'labels': {limb: {c: sum(r[limb] == c for r in subset) for c in CLASSES}
                                      for limb in LIMBS}}
    unsupported = [{'limb': d[0], 'class': d[1], 'sole_casualty': next(iter(g))}
                   for d, g in sorted(support.items()) if len(g) == 1]
    return assignment, {'seed': seed, 'target_validation_fraction': fraction,
                        'candidates_considered': candidates, 'selection_score': best[0][0],
                        'selection_uses_model_scores': False,
                        'train_only_casualties': sorted(train_only),
                        'unsupported_validation_classes': unsupported,
                        'exact_duplicate_groups_retained': [v for v in hash_records.values() if len(v) > 1],
                        'casualty_overlap': [], 'exact_image_hash_overlap': [], **coverage}


def encode_labels(records: list[dict]) -> list[list[int]]:
    return [[CLASSES.index(r[limb]) for limb in LIMBS] for r in records]


def majority_predictions(train: list[dict], count: int) -> list[list[int]]:
    """Class frequencies come exclusively from training; class order breaks ties."""
    labels = [max(range(len(CLASSES)), key=lambda i: sum(r[limb] == CLASSES[i] for r in train))
              for limb in LIMBS]
    return [labels.copy() for _ in range(count)]


def metrics(truth: list[list[int]], predictions: list[list[int]],
            casualties: list[str] | None = None) -> dict:
    if not truth or len(truth) != len(predictions):
        raise ValueError('Expected equally sized, nonempty target and prediction lists')
    for row in [*truth, *predictions]:
        if len(row) != len(LIMBS) or any(type(x) is not int or not 0 <= x < len(CLASSES) for x in row):
            raise ValueError('Expected four integer class indices per image')
    n = len(truth)
    by_limb = {}
    for li, limb in enumerate(LIMBS):
        matrix = [[0] * len(CLASSES) for _ in CLASSES]
        for target, pred in zip(truth, predictions):
            matrix[target[li]][pred[li]] += 1
        per_class = {}
        for ci, name in enumerate(CLASSES):
            tp = matrix[ci][ci]
            support = sum(matrix[ci])
            predicted = sum(row[ci] for row in matrix)
            per_class[name] = {'support': support, 'predicted': predicted,
                               'precision': tp / predicted if predicted else None,
                               'recall': tp / support if support else None,
                               'f1': 2 * tp / (support + predicted) if support else None}
        supported = [v for v in per_class.values() if v['support']]
        by_limb[limb] = {'accuracy': sum(matrix[i][i] for i in range(len(CLASSES))) / n,
                         'macro_f1': sum(v['f1'] for v in supported) / len(supported),
                         'macro_f1_fixed4': sum(v['f1'] or 0 for v in per_class.values()) / len(CLASSES),
                         'balanced_accuracy': sum(v['recall'] for v in supported) / len(supported),
                         'supported_class_count': len(supported), 'classes': per_class,
                         'confusion_matrix_true_rows_predicted_columns': matrix}
    exact = [int(t == p) for t, p in zip(truth, predictions)]
    correct = [sum(a == b for a, b in zip(t, p)) / len(LIMBS) for t, p in zip(truth, predictions)]
    output = {'images': n, 'label_accuracy': sum(correct) / n,
              'exact_match_accuracy': sum(exact) / n,
              'mean_limb_macro_f1': sum(v['macro_f1'] for v in by_limb.values()) / len(LIMBS),
              'mean_limb_macro_f1_fixed4': sum(v['macro_f1_fixed4'] for v in by_limb.values()) / len(LIMBS),
              'by_limb': by_limb,
              'metric_policy': 'Primary macro-F1 averages classes with true validation support within each limb, then the four limbs equally. Unsupported class recall/F1 is null. Fixed-four macro-F1 additionally counts unsupported classes as zero.'}
    if casualties is not None:
        if len(casualties) != n:
            raise ValueError('Casualty list length mismatch')
        ids = sorted(set(casualties))
        output['casualties'] = len(ids)
        output['casualty_mean_label_accuracy'] = sum(
            sum(c for c, g in zip(correct, casualties) if g == name) / casualties.count(name)
            for name in ids) / len(ids)
        output['casualty_mean_exact_match_accuracy'] = sum(
            sum(c for c, g in zip(exact, casualties) if g == name) / casualties.count(name)
            for name in ids) / len(ids)
    return output


def verify_inputs(identity: dict) -> None:
    for path, expected in identity.items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Input changed since preparation: {path}')


def canonical_snapshot(root: Path) -> dict:
    return {k: list(v) for k, v in snapshot(root, root / 'cleaned').items()}
