#!/usr/bin/env python3
"""Build padded native casualty crops and retest the existing model recipes."""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import html
import importlib.metadata
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import baseline
import per_limb
from audit_images import load_image
from baseline_data import encode_labels, file_hash, safe_run_directory, write_json
from baseline_model import (ImageNormalizer, choose_device, image_logits, load_checkpoint,
                            metric_from_logits, seed_everything)
from casualty_crop import POLICY, CasualtyDetector, CoverageDetector, choose_crop, validated_crop, crop_preview
from clean_dataset import CLASSES, LIMBS
from per_limb_model import load_pipeline

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / 'runs/crop_v2'
OLD_SHARED = PROJECT / 'runs/baseline_v1'
OLD_INDEPENDENT = PROJECT / 'runs/per_limb_v1'
_COVERAGE_MODEL = None


def coverage_worker_init(weights):
    global _COVERAGE_MODEL
    _COVERAGE_MODEL=CoverageDetector(Path(weights))


def coverage_worker(task):
    source,path,sha=task
    image,meta=load_image(Path(path))
    try:
        if meta['image_sha256']!=sha:
            raise ValueError('Original image changed during coverage check')
        return {'source_record':source,'image_sha256':sha,'people':_COVERAGE_MODEL.detect(image)}
    finally:
        image.close()


def ensure_coverage_cache(root,records,detections):
    path=root/'coverage_detections.jsonl';done={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['source_record'] in done:
                raise ValueError('Duplicate coverage cache source')
            done[row['source_record']]=row
    tasks=[]
    for row in records:
        source=row['source_record'];found=detections[source]
        if source in done:
            if done[source]['image_sha256']!=row['image_sha256']:
                raise ValueError('Coverage cache identity mismatch')
        elif choose_crop(found['oriented_size'],found['people'])['status']=='cropped':
            tasks.append((source,row['filepath'],row['image_sha256']))
    if not tasks:
        return
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=coverage_worker_init,
            initargs=(str(root/'detector_weights'),)) as pool,path.open('a') as stream:
        for index,item in enumerate(pool.map(coverage_worker,tasks,chunksize=1),1):
            stream.write(json.dumps(item)+'\n');stream.flush()
            if index%25==0:
                print(f'Coverage checked {index}/{len(tasks)} ({time.monotonic()-start:.1f}s)',flush=True)


def check_sources(root):
    protected = json.loads((root / 'protected_runs.json').read_text())
    for name, hashes in protected.items():
        if per_limb.tree_hashes(Path(name)) != hashes:
            raise ValueError(f'Previous run changed: {name}')
    baseline.load_run(OLD_SHARED, verify_previews=False)


def build(args):
    root = args.root.resolve()
    records, context = baseline.load_run(OLD_SHARED)
    safe_run_directory(root, Path(context['dataset_root']))
    for old in [OLD_SHARED, OLD_INDEPENDENT]:
        if root == old or root.is_relative_to(old) or old.is_relative_to(root):
            raise ValueError('Crop experiment must be separate from previous runs')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'shared').exists():
        raise FileExistsError('Crops already prepared; use the train/evaluate commands')
    seed_everything(42)
    detector = CasualtyDetector(root / 'detector_weights', args.detector_device)
    coverage_weights=root/'detector_weights/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
    if not coverage_weights.exists():
        torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
                                          model_dir=str(coverage_weights.parent),progress=False,check_hash=True)
    config = {'policy': POLICY, 'detector_device': str(detector.device),
              'detector_sha256': detector.sha256,
              'detector_weights': str(detector.weights_file.relative_to(root)),
              'detector_url': 'https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth',
              'coverage_detector_sha256':file_hash(root/'detector_weights/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'),
              'crop_code_sha256': file_hash(PROJECT / 'scripts/casualty_crop.py'),
              'packages': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy']},
              'policy_selected_before_classifier_training': True,
              'manual_per_image_overrides': False}
    config_path = root / 'crop_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Crop configuration changed; choose a new experiment directory')
    write_json(config_path, config)
    protected = {str(old): per_limb.tree_hashes(old) for old in [OLD_SHARED, OLD_INDEPENDENT]}
    protected_path = root / 'protected_runs.json'
    if protected_path.exists() and json.loads(protected_path.read_text()) != protected:
        raise ValueError('Previous runs changed since crop preparation started')
    write_json(protected_path, protected)
    cache = root / 'detections.jsonl'
    existing = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            if item['source_record'] in existing:
                raise ValueError('Duplicate source in detector cache')
            existing[item['source_record']] = item
    started = time.monotonic()
    with cache.open('a') as stream:
        for index, row in enumerate(records):
            if row['source_record'] in existing:
                if existing[row['source_record']]['image_sha256'] != row['image_sha256']:
                    raise ValueError('Detection cache image identity mismatch')
                continue
            original, meta = load_image(Path(row['filepath']))
            try:
                if meta['image_sha256'] != row['image_sha256']:
                    raise ValueError('Original image changed since audit')
                detections = detector.detect(original)
                item = {'source_record': row['source_record'], 'image_sha256': meta['image_sha256'],
                        'oriented_size': list(original.size), 'people': detections}
                existing[row['source_record']] = item
                stream.write(json.dumps(item, allow_nan=False) + '\n'); stream.flush()
            finally:
                original.close()
            if (index + 1) % 25 == 0:
                print(f'Detected {index+1}/{len(records)} ({time.monotonic()-started:.1f}s)', flush=True)
    detector.model.cpu(); del detector
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    ensure_coverage_cache(root,records,existing)
    previews = root / 'previews'; previews.mkdir(exist_ok=True)
    overlays = root / 'overlays'; overlays.mkdir(exist_ok=True)
    decisions = []
    coverage={r['source_record']:r for r in map(json.loads,(root/'coverage_detections.jsonl').read_text().splitlines())}
    new_records = copy.deepcopy(records)
    for index, row in enumerate(new_records):
        found = existing[row['source_record']]
        if choose_crop(found['oriented_size'],found['people'])['status']=='cropped' and row['source_record'] not in coverage:
            raise ValueError('Second detector coverage check missing')
        second=coverage.get(row['source_record'],{'people':[]})
        if row['source_record'] in coverage and second['image_sha256']!=row['image_sha256']:
            raise ValueError('Coverage detector cache identity mismatch')
        decision = validated_crop(found['oriented_size'], found['people'],second['people'])
        original, metadata = load_image(Path(row['filepath']))
        try:
            if metadata['image_sha256'] != row['image_sha256']:
                raise ValueError('Original image identity changed')
            preview = crop_preview(original, decision)
            target = previews / f'source_{row["source_record"]:06d}.png'
            preview.save(target); preview.close()
            overlay = original.copy(); overlay.thumbnail((640, 640), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(overlay)
            box = decision['crop_box']; sx = overlay.width / original.width; sy = overlay.height / original.height
            draw.rectangle([box[0]*sx, box[1]*sy, min(overlay.width-1,box[2]*sx), min(overlay.height-1,box[3]*sy)],
                           outline='#06df8c' if decision['status']=='cropped' else '#ffb000', width=4)
            overlay.save(overlays / f'source_{row["source_record"]:06d}.jpg', quality=90); overlay.close()
        finally:
            original.close()
        row['full_frame_preview_filepath'] = row['preview_filepath']
        row['preview_filepath'] = str(target)
        row['preview_sha256'] = file_hash(target)
        row['crop'] = decision
        decisions.append({'source_record': row['source_record'], 'image_sha256': row['image_sha256'],
                          'oriented_size': found['oriented_size'], **decision})
        if (index+1)%100==0:
            print(f'Built {index+1}/{len(records)} crops and full-image overlays', flush=True)
    write_json(root / 'crop_manifest.json', decisions)
    counts = Counter(d['reason'] for d in decisions)
    cropped = [d for d in decisions if d['status']=='cropped']
    summary = {'images': len(records), 'cropped': len(cropped), 'full_frame_fallbacks': len(records)-len(cropped),
               'reasons': dict(counts), 'all_images_retained': True, 'labels_modified': 0,
               'margin_each_side': POLICY['margin_each_side'],
               'median_cropped_area_fraction': statistics.median(d['area_fraction'] for d in cropped) if cropped else None,
               'median_linear_magnification_when_cropped': statistics.median(d['linear_magnification'] for d in cropped) if cropped else None,
               'by_split': {s: dict(Counter(r['crop']['status'] for r in new_records if r['split']==s)) for s in ['train','validation']}}
    write_json(root / 'crop_summary.json', summary)
    gallery(root, new_records)
    shared = root / 'shared'; shared.mkdir()
    for name in ['split_manifest.csv','split_summary.json','majority_metrics.json','majority_predictions.jsonl','requirements.lock']:
        shutil.copyfile(OLD_SHARED / name, shared / name)
    write_json(shared / 'records.json', new_records)
    new_context = copy.deepcopy(context)
    new_context['file_hashes'].update({str(root/name): file_hash(root/name)
                                     for name in ['crop_config.json','crop_manifest.json']})
    new_context['records_sha256'] = file_hash(shared/'records.json')
    new_context['input_fingerprint'] = baseline.fingerprint({'files':new_context['file_hashes'],'records':new_records})
    new_context['preprocessing'] = {'source':'native EXIF-oriented RGB; frozen FCOS person detector; padded crop with full-frame fallback',
                                    'size':448,'whole_frame':False,'crop_config':str(config_path),
                                    'crop_config_sha256':file_hash(config_path),'padding_rgb':[114,114,114],
                                    'normalization_mean':[.485,.456,.406],'normalization_std':[.229,.224,.225],
                                    'augmentation':'none; deterministic casualty crop only; no flips'}
    write_json(shared/'inputs.json', new_context)
    weights=shared/'pretrained_weights/checkpoints';weights.mkdir(parents=True)
    shutil.copyfile(OLD_SHARED/'pretrained_weights/checkpoints/resnet18-f37072fd.pth',weights/'resnet18-f37072fd.pth')
    code=root/'code';code.mkdir(exist_ok=True)
    for name in ['crop_experiment.py','casualty_crop.py']:
        shutil.copyfile(PROJECT/'scripts'/name,code/name)
    check_sources(root)
    print(json.dumps(summary,indent=2),flush=True)


def gallery(root, records):
    sheets=root/'sheets';sheets.mkdir(exist_ok=True)
    links=[]
    for page,start in enumerate(range(0,len(records),64),1):
        subset=records[start:start+64]
        canvas=Image.new('RGB',(8*224,8*250),'#e5e5e5');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(subset):
            x,y=(j%8)*224,(j//8)*250
            with Image.open(root/'overlays'/f'source_{row["source_record"]:06d}.jpg') as im:
                thumb=ImageOps.contain(im,(220,220))
                canvas.paste(thumb,(x+(224-thumb.width)//2,y+22));thumb.close()
            draw.text((x+3,y+3),f'{row["source_record"]}: {row["crop"]["status"]} {row["crop"]["linear_magnification"]:.1f}x',fill='black')
        name=f'sheet_{page:02d}.jpg';canvas.save(sheets/name,quality=92);canvas.close()
        links.append(f'<a href="sheets/{name}">Overlay sheet {page}</a>')
    cards=[]
    for row in records:
        sid=row['source_record'];d=row['crop']
        cards.append(f'<article><b>Source {sid} · {html.escape(row["casualty_id"])}</b><p>{d["reason"]}; '
                     f'{d["linear_magnification"]:.2f}× linear magnification</p><div class="pair">'
                     f'<img src="overlays/source_{sid:06d}.jpg" alt="Original with crop boundary">'
                     f'<img src="previews/source_{sid:06d}.png" alt="Actual unannotated model input"></div></article>')
    page='<!doctype html><meta charset="utf-8"><title>Padded casualty crop audit</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}article{border-top:1px solid #ccc;padding:16px 0}.pair{display:flex;gap:12px;align-items:center}.pair img{width:45%;max-width:448px;object-fit:contain}a{margin-right:15px}</style>'
    page+='<h1>Padded casualty crop audit</h1><p>Green: selected padded crop. Orange: full-image fallback. Left: original image with boundary. Right: actual 448px model input. All original labels retained. No boxes were trained on this dataset.</p><p>'+' '.join(links)+'</p>'+''.join(cards)
    (root/'crops.html').write_text(page)


def annotate_checkpoint(path, context):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('requires_crop_pipeline'):
        return
    data.update({'requires_crop_pipeline':True,'preprocessing':context['preprocessing'],
                 'inference_entrypoint':'scripts/crop_experiment.py predict; apply frozen detector and padded native crop before normalization'})
    temporary=path.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(path)


def train_shared(args):
    check_sources(args.root)
    shared=args.root/'shared'
    if not (shared/'smoke.json').exists():
        parsed=baseline.parser().parse_args(['smoke','--run-dir',str(shared)])
        baseline.smoke(parsed)
    parsed=baseline.parser().parse_args(['train','--run-dir',str(shared)])
    baseline.train(parsed)
    records,context=baseline.load_run(shared)
    for stage in ['frozen','finetuned']:
        annotate_checkpoint(shared/f'{stage}.pt',context)
    evaluate_shared(args.root)


def evaluate_shared(root):
    shared=root/'shared';records,context=baseline.load_run(shared)
    val=[r for r in records if r['split']=='validation'];device=choose_device('auto')
    results={'majority':json.loads((shared/'majority_metrics.json').read_text())}
    for stage in ['frozen','finetuned']:
        model,info=load_checkpoint(shared/f'{stage}.pt',context['input_fingerprint'])
        logits=image_logits(model,val,device,8)
        result=metric_from_logits(val,logits)
        result.update({'checkpoint_epoch':info['epoch'],'checkpoint_sha256':file_hash(shared/f'{stage}.pt'),
                       'preprocessing':context['preprocessing']})
        results[stage]=result
        write_json(shared/f'{stage}_metrics.json',result)
        baseline.write_predictions(shared/f'{stage}_predictions.jsonl',val,logits.argmax(-1).tolist(),logits.softmax(-1).tolist())
        print(f'cropped shared {stage}: macro-F1 {result["mean_limb_macro_f1"]:.4f}',flush=True)
        model.cpu();del model
        if device.type=='mps':torch.mps.empty_cache()
    comparison={'input_fingerprint':context['input_fingerprint'],'models':results,
                'all_models_use_identical_validation_source_records':[r['source_record'] for r in val],
                'evaluation_scope':'same development casualty holdout used for checkpoint selection',
                'preprocessing':context['preprocessing']}
    write_json(shared/'comparison.json',comparison)


def train_independent(args):
    check_sources(args.root)
    run=args.root/'per_limb'
    if not run.exists():
        per_limb.prepare(argparse.Namespace(reference_run=args.root/'shared',run_dir=run))
    per_limb.train(argparse.Namespace(run_dir=run,device='auto'))
    _,context=baseline.load_run(run)
    for stage in ['frozen','finetuned']:
        path=run/('pipeline.json' if stage=='finetuned' else 'frozen_pipeline.json')
        bundle=json.loads(path.read_text())
        bundle.update(requires_crop_pipeline=True,inference_entrypoint='scripts/crop_experiment.py predict --variant per_limb')
        for entry in bundle['checkpoints']:
            checkpoint=run/entry['file'];annotate_checkpoint(checkpoint,context)
            entry['sha256']=file_hash(checkpoint)
        write_json(path,bundle)
    per_limb.checked_run(run)


def evaluate(args):
    root=args.root;check_sources(root)
    run=root/'per_limb';records,context=baseline.load_run(run)
    per_limb.checked_run(run)
    val=[r for r in records if r['split']=='validation'];device=choose_device('auto')
    old=json.loads((OLD_INDEPENDENT/'comparison.json').read_text())
    results={name:copy.deepcopy(result) for name,result in old['models'].items()}
    current=json.loads((root/'shared/comparison.json').read_text())
    for stage in ['frozen','finetuned']:
        results['cropped_shared_'+stage]=current['models'][stage]
        model,info=load_pipeline(run/('pipeline.json' if stage=='finetuned' else 'frozen_pipeline.json'))
        logits=image_logits(model,val,device,8)
        result=metric_from_logits(val,logits)
        name='cropped_per_limb_'+stage;results[name]=result
        write_json(run/f'{stage}_metrics.json',result)
        baseline.write_predictions(run/f'{stage}_predictions.jsonl',val,logits.argmax(-1).tolist(),logits.softmax(-1).tolist())
        print(f'{name}: macro-F1 {result["mean_limb_macro_f1"]:.4f}, accuracy {result["label_accuracy"]:.4f}, all-four {result["exact_match_accuracy"]:.4f}',flush=True)
        model.cpu();del model
        if device.type=='mps':torch.mps.empty_cache()
    comparison={'models':results,'input_fingerprint':context['input_fingerprint'],
                'all_models_use_identical_validation_source_records':[r['source_record'] for r in val],
                'split_manifest_sha256':context['split_manifest_sha256'],
                'selection_metric':'same supported-class mean limb macro-F1; separate selection per limb for independent models',
                'scope':'one development split/seed; validation used for checkpoints; not an untouched test set',
                'crop_config_sha256':file_hash(root/'crop_config.json'),'labels_modified':0}
    assert comparison['all_models_use_identical_validation_source_records']==old['all_models_use_identical_validation_source_records']
    write_json(root/'comparison.json',comparison)
    report(root,comparison)
    check_sources(root)


def report(root, comparison):
    summary=json.loads((root/'crop_summary.json').read_text());rows=[];details=[]
    fmt=lambda x:'n.a.' if x is None else f'{100*x:.1f}%'
    lines=['Native padded casualty crop comparison','','Model / macro-F1 / limb accuracy / all four correct / mean casualty accuracy']
    for name,m in comparison['models'].items():
        values=[name,f'{m["mean_limb_macro_f1"]:.3f}',fmt(m['label_accuracy']),fmt(m['exact_match_accuracy']),fmt(m['casualty_mean_label_accuracy'])]
        rows.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in values)+'</tr>');lines.append(' / '.join(values))
        stats=[];matrices=[]
        for limb,v in m['by_limb'].items():
            for label,c in v['classes'].items():
                cells=[limb,label,str(c['support']),str(c['predicted']),fmt(c['precision']),fmt(c['recall']),fmt(c['f1'])]
                stats.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in cells)+'</tr>')
            matrix=v['confusion_matrix_true_rows_predicted_columns']
            matrices.append(f'<h4>{limb}: macro-F1 {v["macro_f1"]:.3f}</h4><table><tr><th>True / predicted</th>'+''.join(f'<th>{c}</th>' for c in CLASSES)+'</tr>'+''.join('<tr><th>'+CLASSES[i]+'</th>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for i,row in enumerate(matrix))+'</table>')
        details.append(f'<section><h2>{name}</h2><details><summary>Per-class metrics</summary><table><tr><th>Limb</th><th>Class</th><th>True</th><th>Predicted</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'+''.join(stats)+'</table></details><details><summary>Confusion matrices</summary>'+''.join(matrices)+'</details></section>')
    note=(f'{summary["cropped"]}/952 images use an automatic casualty crop; {summary["full_frame_fallbacks"]} retain the full image. '
          'Frozen COCO-pretrained FCOS, confidence ≥0.65; overlapping Faster R-CNN ≥0.90 confirms coverage; union of boxes plus 30% width/height on each side, clamped to original image. '
          'Missing, very small or ambiguous detections use the full frame. No images removed. '
          'Crops come from EXIF-oriented native RGB images, then resize/pad to 448 with RGB114; ImageNet normalization. '
          'Classifier architecture, initial weights, training recipes, casualty split and labels match the full-frame experiments. '
          '758 train images / 50 casualties; 194 validation images / 12 casualties. '
          'Each classifier selects its own permitted checkpoint; this is a development holdout, not an untouched test set. '
          'Left-arm amputation has zero independent validation support and is omitted from that limb’s supported-class macro-F1. '
          'Detector outputs have no ground-truth bounding boxes; visual review coverage is recorded separately. '
          'Original media, annotations and previous runs remain unchanged.')
    lines+=['',note,'',json.dumps(summary,indent=2)];(root/'summary.txt').write_text('\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Casualty crop comparison</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:28px auto;padding:0 18px;background:#f5f7f9;color:#17212b}section,.intro{background:white;border:1px solid #ddd;padding:20px;margin:20px 0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}details{margin:16px 0}summary{font-weight:600;cursor:pointer}a{color:#075faa}</style><h1>Full frame versus padded casualty crop</h1><div class="intro"><table><tr><th>Model</th><th>Macro-F1</th><th>Limb accuracy</th><th>All four correct</th><th>Mean casualty accuracy</th></tr>'+''.join(rows)+'</table><p>'+html.escape(note)+'</p><p><a href="crops.html">Every crop and original overlay</a> · <a href="crop_summary.json">Crop coverage</a> · <a href="comparison.json">Complete results</a> · <a href="shared/split_manifest.csv">Fixed split</a> · <a href="summary.txt">Text summary</a></p></div>'+''.join(details)
    (root/'index.html').write_text(page)


def predict(args):
    root=args.root.resolve();config=json.loads((root/'crop_config.json').read_text())
    if config['policy']!=POLICY or file_hash(PROJECT/'scripts/casualty_crop.py')!=config['crop_code_sha256']:
        raise ValueError('Crop inference implementation/configuration changed')
    if {p:importlib.metadata.version(p) for p in config['packages']}!=config['packages']:
        raise ValueError('Use the recorded package versions for the crop pipeline')
    seed_everything(42)
    detector=CasualtyDetector(root/'detector_weights',config['detector_device'],download=False)
    if detector.sha256!=config['detector_sha256']:
        raise ValueError('Detector weights changed')
    original,meta=load_image(args.image.resolve())
    try:
        people=detector.detect(original)
        second=CoverageDetector(root/'detector_weights')
        if second.sha256!=config['coverage_detector_sha256']:
            raise ValueError('Coverage detector weights changed')
        coverage=second.detect(original) if choose_crop(original.size,people)['status']=='cropped' else []
        decision=validated_crop(original.size,people,coverage);preview=crop_preview(original,decision)
        del second
    finally:
        original.close()
    pixels=torch.from_numpy(np.array(preview,dtype=np.uint8,copy=True).transpose(2,0,1).copy()).unsqueeze(0)
    preview.close();detector.model.cpu();del detector
    device=choose_device('auto')
    if device.type=='mps':torch.mps.empty_cache()
    if args.variant=='shared':
        checkpoint=root/'shared'/f'{args.stage}.pt';model,info=load_checkpoint(checkpoint)
    else:
        checkpoint=root/'per_limb'/('pipeline.json' if args.stage=='finetuned' else 'frozen_pipeline.json')
        model,info=load_pipeline(checkpoint)
    if (not info.get('requires_crop_pipeline') or
            info['preprocessing']['crop_config_sha256']!=file_hash(root/'crop_config.json')):
        raise ValueError('Classifier belongs to a different preprocessing pipeline')
    model.to(device).eval()
    with torch.inference_mode():
        p=model(ImageNormalizer().to(device)(pixels.to(device))).softmax(-1)[0].cpu().tolist()
    result={'image':str(args.image.resolve()),'image_sha256':meta['image_sha256'],'variant':args.variant,
            'stage':args.stage,'crop':decision,'crop_config_sha256':file_hash(root/'crop_config.json'),
            'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
            'inference_device':str(device),
            'prediction':{limb:CLASSES[max(range(4),key=lambda j:p[i][j])] for i,limb in enumerate(LIMBS)},
            'probabilities':{limb:dict(zip(CLASSES,p[i])) for i,limb in enumerate(LIMBS)},'probabilities_are_calibrated':False}
    if args.json_output:
        target=args.json_output.resolve()
        if target.exists() or target.is_relative_to((PROJECT/'train').resolve()) or any(target.is_relative_to(p) for p in [OLD_SHARED,OLD_INDEPENDENT]):
            raise ValueError('Output must be a new file outside canonical data and previous runs')
        write_json(target,result)
    print(json.dumps(result,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name,function in [('build',build),('train-shared',train_shared),('train-per-limb',train_independent),('evaluate',evaluate),('predict',predict)]:
        sub=subs.add_parser(name);sub.add_argument('--root',type=Path,default=DEFAULT_ROOT);sub.set_defaults(function=function)
        if name=='build':sub.add_argument('--detector-device',choices=['mps','cpu','cuda'],default='mps')
        if name=='predict':
            sub.add_argument('--image',type=Path,required=True);sub.add_argument('--variant',choices=['shared','per_limb'],default='shared')
            sub.add_argument('--stage',choices=['frozen','finetuned'],default='finetuned');sub.add_argument('--json-output',type=Path)
    return p


if __name__=='__main__':
    args=parser().parse_args();args.root=args.root.resolve();args.function(args)
