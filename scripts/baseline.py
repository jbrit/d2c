#!/usr/bin/env python3
"""Prepare, train, evaluate and run the image-only four-limb baseline.

All run artifacts are separate from the canonical dataset. See README.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.metadata
import io
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from audit_images import load_image, make_preview
from baseline_data import (audited_records, canonical_snapshot, encode_labels, file_hash,
                           grouped_split, majority_predictions, metrics, read_csv,
                           safe_run_directory, verify_inputs, write_json)
from baseline_model import (ImageNormalizer, LimbModel, choose_device, extract_features,
                            image_loader, image_logits, load_checkpoint, loss_fn,
                            metric_from_logits, save_checkpoint, seed_everything)
from clean_dataset import CLASSES, LIMBS, csv_bytes


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT / 'runs/baseline_v1'


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare(args):
    root = args.dataset_root.resolve()
    run = args.run_dir.resolve()
    safe_run_directory(run, root)
    if run.exists():
        raise FileExistsError(f'Run directory already exists: {run}. Reuse its other commands or choose a new run directory.')
    before = canonical_snapshot(root)
    records, identity = audited_records(root)
    assignment, split = grouped_split(records, args.seed, args.validation_fraction, args.split_candidates)
    review = root / 'cleaned/annotation_review/suspected_label_issues.csv'
    flags = {}
    if review.is_file():
        identity[str(review)] = file_hash(review)
        for r in read_csv(review):
            flags.setdefault(int(r['source_record']), set()).add(r['assessment'])
    print(f'Checking original and unannotated preview hashes for {len(records)} images...', flush=True)
    for index, r in enumerate(records):
        if file_hash(Path(r['filepath'])) != r['image_sha256']:
            raise ValueError(f'Original image changed since audit: {r["filepath"]}')
        preview = Path(r['preview_filepath'])
        with Image.open(preview) as im:
            im.load()
            if im.size != (448, 448) or im.mode != 'RGB':
                raise ValueError(f'Invalid preview: {preview}')
        r['preview_sha256'] = file_hash(preview)
        r['split'] = assignment[r['source_record']]
        r['annotation_review_flags'] = sorted(flags.get(r['source_record'], set()))
        if (index + 1) % 100 == 0:
            print(f'verified {index+1}/{len(records)}', flush=True)
    if before != canonical_snapshot(root):
        raise ValueError('Canonical file inventory changed during preparation')
    verify_inputs(identity)
    run.mkdir(parents=True)
    write_json(run / 'records.json', records)
    write_json(run / 'split_summary.json', split)
    split_rows = [{k: r[k] for k in ('source_record','casualty_id','split','filepath',*LIMBS)} for r in records]
    (run / 'split_manifest.csv').write_bytes(csv_bytes(list(split_rows[0]), split_rows))
    context = {'dataset_root': str(root), 'classes': list(CLASSES), 'limbs': list(LIMBS),
               'file_hashes': identity, 'canonical_snapshot': before,
               'records_sha256': file_hash(run / 'records.json'),
               'split_manifest_sha256': file_hash(run / 'split_manifest.csv'),
               'split_summary_sha256': file_hash(run / 'split_summary.json'),
               'input_fingerprint': fingerprint({'files': identity, 'records': records}),
               'labels_modified': 0, 'videos_used': 0, 'all_annotation_review_candidates_retained': True,
               'preprocessing': {'source': 'unannotated image-audit PNG previews', 'size': 448,
                                 'whole_frame': True, 'padding_rgb': [114,114,114],
                                 'normalization_mean': [.485,.456,.406],
                                 'normalization_std': [.229,.224,.225],
                                 'augmentation': 'none; no crops or flips'},
               'reproducibility': 'Fixed source identity and casualty assignment. Seeded Python/NumPy/Torch. Bitwise equality across devices or library releases is not guaranteed.'}
    write_json(run / 'inputs.json', context)
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'validation']
    predicted = majority_predictions(train, len(val))
    majority = metrics(encode_labels(val), predicted, [r['casualty_id'] for r in val])
    majority['class_chosen_from_training_only'] = {limb: CLASSES[predicted[0][i]] for i, limb in enumerate(LIMBS)}
    write_json(run / 'majority_metrics.json', majority)
    write_predictions(run / 'majority_predictions.jsonl', val, predicted)
    print(json.dumps({'run_directory':str(run),'train_images':len(train),'validation_images':len(val),
                      'train_casualties':len(split['train']['casualties']),
                      'validation_casualties':len(split['validation']['casualties']),
                      'unsupported_validation_classes':split['unsupported_validation_classes'],
                      'majority_validation_macro_f1':majority['mean_limb_macro_f1']},indent=2),flush=True)


def load_run(run: Path, verify_previews=True):
    run = run.resolve()
    context = json.loads((run / 'inputs.json').read_text())
    root = Path(context['dataset_root'])
    safe_run_directory(run, root)
    if context['classes'] != list(CLASSES) or context['limbs'] != list(LIMBS):
        raise ValueError('Run class or limb order mismatch')
    for name, key in [('records.json','records_sha256'),('split_manifest.csv','split_manifest_sha256'),
                      ('split_summary.json','split_summary_sha256')]:
        if file_hash(run / name) != context[key]:
            raise ValueError(f'Prepared run artifact was edited: {name}')
    verify_inputs(context['file_hashes'])
    if context['canonical_snapshot'] != canonical_snapshot(root):
        raise ValueError('Canonical file inventory/metadata changed since preparation')
    records = json.loads((run / 'records.json').read_text())
    if fingerprint({'files':context['file_hashes'],'records':records}) != context['input_fingerprint']:
        raise ValueError('Prepared input fingerprint mismatch')
    if verify_previews:
        for r in records:
            if file_hash(Path(r['preview_filepath'])) != r['preview_sha256']:
                raise ValueError(f'Preview changed since preparation: {r["preview_filepath"]}')
    return records, context


def write_predictions(path, records, predictions, probabilities=None):
    with path.open('w') as stream:
        for index, (r, pred) in enumerate(zip(records, predictions)):
            output = {'source_record':r['source_record'],'casualty_id':r['casualty_id'],
                      'filepath':r['filepath'],'annotation_review_flags':r['annotation_review_flags'],
                      'truth':{limb:r[limb] for limb in LIMBS},
                      'prediction':{limb:CLASSES[pred[i]] for i,limb in enumerate(LIMBS)},
                      'correct_limbs':sum(CLASSES[pred[i]]==r[limb] for i,limb in enumerate(LIMBS))}
            if probabilities is not None:
                output['probabilities'] = {limb:dict(zip(CLASSES,probabilities[index][i])) for i,limb in enumerate(LIMBS)}
            stream.write(json.dumps(output,allow_nan=False)+'\n')


def model_and_features(args, records, context):
    device=choose_device(args.device)
    seed_everything(args.seed)
    torch.hub.set_dir(str(args.run_dir.resolve()/'pretrained_weights'))
    model=LimbModel(pretrained=True)
    checkpoint=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    weights_hash=file_hash(checkpoint)
    feature_identity=fingerprint({'input_fingerprint':context['input_fingerprint'],
                                  'pretrained_sha256':weights_hash,
                                  'model_code_sha256':file_hash(PROJECT/'scripts/baseline_model.py'),
                                  'torch':torch.__version__,
                                  'torchvision':importlib.metadata.version('torchvision')})
    write_json(args.run_dir/'pretrained.json',{'weights':'ResNet18_Weights.IMAGENET1K_V1',
               'url':'https://download.pytorch.org/models/resnet18-f37072fd.pth',
               'sha256':weights_hash,'feature_fingerprint':feature_identity})
    features=extract_features(model,records,device,args.batch_size,args.run_dir,feature_identity)
    model.cpu()
    if device.type=='mps':torch.mps.empty_cache()
    return model,features.clone()


def smoke(args):
    records,context=load_run(args.run_dir)
    model,features=model_and_features(args,records,context)
    pool=[i for i,r in enumerate(records) if r['split']=='train']
    random.Random(args.seed).shuffle(pool)
    selected=[]
    covered=set()
    while len(selected)<min(args.smoke_images,len(pool)):
        remaining=[i for i in pool if i not in selected]
        chosen=max(remaining,key=lambda i:sum((limb,records[i][limb]) not in covered for limb in LIMBS))
        selected.append(chosen)
        covered.update((limb,records[chosen][limb]) for limb in LIMBS)
    heads=copy.deepcopy(model.heads).cpu()
    x=features[selected]
    y=torch.tensor(encode_labels([records[i] for i in selected]),dtype=torch.long)
    optimizer=AdamW(heads.parameters(),lr=.02,weight_decay=0)
    initial=None
    start=time.monotonic()
    for step in range(args.smoke_steps):
        logits=torch.stack([head(x) for head in heads],dim=1)
        loss=loss_fn(logits,y)
        if initial is None:initial=float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits=torch.stack([head(x) for head in heads],dim=1)
            final_loss=float(loss_fn(logits,y))
            exact=float((logits.argmax(-1)==y).all(dim=1).float().mean())
        if exact==1.0 and final_loss<.02:break
    result={'stage':'frozen-backbone head overfit on training images only',
            'images':len(selected),'source_records':[records[i]['source_record'] for i in selected],
            'steps':step+1,'initial_loss':initial,'final_loss':final_loss,
            'exact_match_accuracy':exact,'seconds':time.monotonic()-start,
            'passed':exact==1.0 and final_loss<.05,
            'input_fingerprint':context['input_fingerprint'],
            'actual_training_heads_modified':False}
    write_json(args.run_dir/'smoke.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('Tiny-batch overfit did not pass; inspect the pipeline before training')


def train_frozen(args,records,context,model,features):
    path=args.run_dir/'frozen.pt'
    if path.exists():
        model,data=load_checkpoint(path,context['input_fingerprint'])
        print(f'Reusing saved frozen checkpoint from epoch {data["epoch"]}',flush=True)
        return model
    model.cpu().configure_stage('frozen')
    train_indices=[i for i,r in enumerate(records) if r['split']=='train']
    val_indices=[i for i,r in enumerate(records) if r['split']=='validation']
    val=[records[i] for i in val_indices]
    all_labels=torch.tensor(encode_labels(records),dtype=torch.long)
    loader=DataLoader(TensorDataset(features[train_indices],all_labels[train_indices]),
                      batch_size=args.feature_batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))
    optimizer=AdamW(model.heads.parameters(),lr=args.head_lr,weight_decay=args.weight_decay)
    best_score=-1.;best_heads=None;best_epoch=0;stale=0
    start=time.monotonic()
    with (args.run_dir/'frozen_history.jsonl').open('w') as history:
        for epoch in range(1,args.head_epochs+1):
            total=0.;correct=0;seen=0
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits=model.classify(x)
                loss=loss_fn(logits,y)
                loss.backward();optimizer.step()
                total+=float(loss.detach())*len(y)
                correct+=int((logits.detach().argmax(-1)==y).sum())
                seen+=len(y)
            with torch.no_grad():
                val_logits=model.classify(features[val_indices])
                val_loss=float(loss_fn(val_logits,all_labels[val_indices]))
            score=metric_from_logits(val,val_logits)['mean_limb_macro_f1']
            item={'epoch':epoch,'train_loss':total/seen,'train_label_accuracy':correct/(4*seen),
                  'validation_loss':val_loss,'validation_macro_f1':score,
                  'seconds':time.monotonic()-start}
            history.write(json.dumps(item)+'\n');history.flush()
            if score>best_score+1e-8:
                best_score=score;best_epoch=epoch
                best_heads=copy.deepcopy(model.heads.state_dict());stale=0
            else:stale+=1
            if epoch==1 or epoch%5==0:
                print(f'frozen epoch {epoch}: train loss {total/seen:.4f}, val macro-F1 {score:.4f}, best {best_score:.4f}',flush=True)
            if stale>=args.head_patience:break
    model.heads.load_state_dict(best_heads)
    save_checkpoint(path,model,best_epoch,best_score,context['input_fingerprint'],'frozen')
    write_json(args.run_dir/'frozen_training.json',{'best_epoch':best_epoch,'best_validation_macro_f1':best_score,
               'epochs_completed':epoch,'seconds':time.monotonic()-start,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'backbone_weights_and_buffers_fixed':True})
    return model


def train_finetuned(args,records,context,model):
    path=args.run_dir/'finetuned.pt'
    if path.exists():
        load_checkpoint(path,context['input_fingerprint'])
        print('Reusing saved fine-tuned checkpoint',flush=True)
        return
    device=choose_device(args.device)
    model.to(device).configure_stage('finetuned')
    train=[r for r in records if r['split']=='train']
    val=[r for r in records if r['split']=='validation']
    normalizer=ImageNormalizer().to(device)
    optimizer=AdamW([{'params':model.backbone.layer4.parameters(),'lr':args.backbone_lr},
                     {'params':model.heads.parameters(),'lr':args.finetune_head_lr}],weight_decay=args.weight_decay)
    loader=image_loader(train,args.batch_size,shuffle=True,seed=args.seed)
    best_score=-1.;best_state=None;best_epoch=0;stale=0
    buffers={name:buf.detach().cpu().clone() for name,buf in model.backbone.named_buffers()}
    start=time.monotonic()
    with (args.run_dir/'finetuned_history.jsonl').open('w') as history:
        for epoch in range(1,args.finetune_epochs+1):
            model.configure_stage('finetuned')
            total=0.;correct=0;seen=0
            for pixels,y in loader:
                x=normalizer(pixels.to(device));y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x)
                loss=loss_fn(logits,y)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
                optimizer.step()
                total+=float(loss.detach())*len(y)
                correct+=int((logits.detach().argmax(-1)==y).sum())
                seen+=len(y)
            val_logits=image_logits(model,val,device,args.batch_size)
            val_loss=float(loss_fn(val_logits,torch.tensor(encode_labels(val))))
            score=metric_from_logits(val,val_logits)['mean_limb_macro_f1']
            item={'epoch':epoch,'train_loss':total/seen,'train_label_accuracy':correct/(4*seen),
                  'validation_loss':val_loss,'validation_macro_f1':score,
                  'seconds':time.monotonic()-start}
            history.write(json.dumps(item)+'\n');history.flush()
            if score>best_score+1e-8:
                best_score=score;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
            print(f'finetuned epoch {epoch}: train loss {total/seen:.4f}, val macro-F1 {score:.4f}, best {best_score:.4f} ({time.monotonic()-start:.1f}s)',flush=True)
            if stale>=args.finetune_patience:break
    if any(not torch.equal(buf,dict(model.backbone.named_buffers())[name].detach().cpu()) for name,buf in buffers.items()):
        raise RuntimeError('Backbone running statistics changed unexpectedly')
    model.load_state_dict(best_state)
    save_checkpoint(path,model,best_epoch,best_score,context['input_fingerprint'],'finetuned')
    write_json(args.run_dir/'finetuned_training.json',{'best_epoch':best_epoch,'best_validation_macro_f1':best_score,
               'epochs_completed':epoch,'seconds':time.monotonic()-start,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'unfrozen_backbone_blocks':['layer4'],'batchnorm_running_statistics_fixed':True,
               'warm_start':'frozen.pt; no reinitialization of heads'})


def train(args):
    records,context=load_run(args.run_dir)
    if not (args.run_dir/'smoke.json').exists():
        raise ValueError('Run the smoke command before training')
    smoke_result=json.loads((args.run_dir/'smoke.json').read_text())
    if not smoke_result['passed'] or smoke_result['input_fingerprint']!=context['input_fingerprint']:
        raise ValueError('Smoke check did not pass for this prepared input')
    config={key:getattr(args,key) for key in ['seed','device','batch_size','feature_batch_size','head_epochs',
             'finetune_epochs','head_patience','finetune_patience','head_lr','backbone_lr','finetune_head_lr','weight_decay']}
    config.update({'loss':'unweighted mean cross entropy over all four limbs',
                   'model_selection':'highest mean per-limb validation macro-F1 over supported classes; earliest epoch breaks ties',
                   'finetune_scope':'last ResNet block plus heads; BatchNorm running statistics fixed',
                   'validation_is_development_holdout':True,'image_augmentation':'none'})
    config_path=args.run_dir/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Training configuration differs from this run; choose a new run directory')
    write_json(config_path,config)
    code_dir=args.run_dir/'code';code_dir.mkdir(exist_ok=True)
    code_hashes={}
    for path in [Path(__file__),PROJECT/'scripts/baseline_data.py',PROJECT/'scripts/baseline_model.py',
                 PROJECT/'scripts/audit_images.py',PROJECT/'scripts/clean_dataset.py']:
        target=code_dir/path.name
        if target.exists() and file_hash(target)!=file_hash(path):
            raise ValueError('Training source code differs from saved run; use a new run directory')
        shutil.copyfile(path,target);code_hashes[path.name]=file_hash(path)
    write_json(args.run_dir/'environment.json',{'python':sys.version,'executable':sys.executable,
               'platform':platform.platform(),'machine':platform.machine(),
               'device':str(choose_device(args.device)),
               'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow']},
               'code_sha256':code_hashes,'input_fingerprint':context['input_fingerprint']})
    seed_everything(args.seed)
    if (args.run_dir/'frozen.pt').exists():
        model,_=load_checkpoint(args.run_dir/'frozen.pt',context['input_fingerprint'])
    else:
        model,features=model_and_features(args,records,context)
        model=train_frozen(args,records,context,model,features)
    if args.stage=='all':train_finetuned(args,records,context,model)
    load_run(args.run_dir,verify_previews=False)


def evaluate(args):
    records,context=load_run(args.run_dir)
    seed_everything(42)
    device=choose_device(args.device)
    train_rows=[r for r in records if r['split']=='train']
    val=[r for r in records if r['split']=='validation']
    majority=majority_predictions(train_rows,len(val))
    results={'majority':metrics(encode_labels(val),majority,[r['casualty_id'] for r in val])}
    examples={}
    for stage in ['frozen','finetuned']:
        checkpoint=args.run_dir/f'{stage}.pt'
        if not checkpoint.exists():continue
        model,info=load_checkpoint(checkpoint,context['input_fingerprint'])
        start=time.monotonic()
        logits=image_logits(model,val,device,args.batch_size)
        probabilities=logits.softmax(-1).tolist()
        predictions=logits.argmax(-1).tolist()
        result=metric_from_logits(val,logits)
        result.update({'checkpoint_epoch':info['epoch'],'checkpoint_sha256':file_hash(checkpoint),
                       'evaluation_device':str(device),'evaluation_seconds':time.monotonic()-start,
                       'loss':float(loss_fn(logits,torch.tensor(encode_labels(val))))})
        results[stage]=result
        write_json(args.run_dir/f'{stage}_metrics.json',result)
        write_predictions(args.run_dir/f'{stage}_predictions.jsonl',val,predictions,probabilities)
        errors=sorted(zip(val,predictions),key=lambda pair:sum(CLASSES[pair[1][i]]==pair[0][limb] for i,limb in enumerate(LIMBS)))
        shown=[];seen=set()
        for row,pred in errors:
            if row['casualty_id'] not in seen and any(CLASSES[pred[i]]!=row[limb] for i,limb in enumerate(LIMBS)):
                shown.append((row,pred));seen.add(row['casualty_id'])
            if len(shown)==6:break
        examples[stage]=shown
        model.cpu();del model
        if device.type=='mps':torch.mps.empty_cache()
    if len(results)==1:raise ValueError('No learned checkpoints found; run train first')
    best=max((name for name in results if name!='majority'),key=lambda name:results[name]['mean_limb_macro_f1'])
    comparison={'input_fingerprint':context['input_fingerprint'],
                'split_summary_sha256':context['split_summary_sha256'],
                'selection_metric':'mean_limb_macro_f1; absent true classes omitted per limb',
                'selected_learned_stage':best,'evaluation_scope':'development casualty holdout used for checkpoint selection, not a final test set',
                'annotation_review_deferred':True,
                'flagged_validation_images':sum(bool(r['annotation_review_flags']) for r in val),
                'all_models_use_identical_validation_source_records':[r['source_record'] for r in val],
                'models':results}
    write_json(args.run_dir/'comparison.json',comparison)
    write_report(args.run_dir,context,comparison,examples)
    load_run(args.run_dir,verify_previews=False)
    write_json(args.run_dir/'preservation.json',{'canonical_csv_and_audit_inputs_unchanged':True,
               'canonical_file_inventory_sizes_mtimes_unchanged':True,
               'canonical_files':len(context['canonical_snapshot']),
               'original_images_hashed_against_audit_during_preparation':len(records),
               'labels_modified':0,'videos_used':0})
    print(json.dumps({'report':str(args.run_dir/'index.html'),'selected_stage':best,
          'metrics':{name:{k:m[k] for k in ['mean_limb_macro_f1','label_accuracy','exact_match_accuracy']}
                     for name,m in results.items()}},indent=2),flush=True)


def write_report(run,context,comparison,examples):
    split=json.loads((run/'split_summary.json').read_text())
    config=json.loads((run/'training_config.json').read_text())
    models=comparison['models']
    fmt=lambda x:'n.a.' if x is None else f'{100*x:.1f}%'
    esc=html.escape
    table=[]
    lines=['Image-only baseline comparison','',
           f'Training: {split["train"]["images"]} images / {len(split["train"]["casualties"])} casualties.',
           f'Validation: {split["validation"]["images"]} images / {len(split["validation"]["casualties"])} casualties.',
           'No casualty or exact image hash crosses the split. Existing labels and all review candidates retained.',
           '', 'Model             Macro-F1  Label accuracy  All-four correct']
    for name,m in models.items():
        table.append(f'<tr><th>{esc(name)}</th><td>{m["mean_limb_macro_f1"]:.3f}</td><td>{fmt(m["label_accuracy"])}</td><td>{fmt(m["exact_match_accuracy"])}</td><td>{fmt(m["casualty_mean_label_accuracy"])}</td></tr>')
        lines.append(f'{name:<17} {m["mean_limb_macro_f1"]:.3f}     {fmt(m["label_accuracy"]):>8}       {fmt(m["exact_match_accuracy"]):>8}')
    sections=[]
    for name,m in models.items():
        limb_rows=[];classes=[];matrices=[]
        for limb in LIMBS:
            data=m['by_limb'][limb]
            limb_rows.append(f'<tr><th>{limb}</th><td>{data["macro_f1"]:.3f}</td><td>{fmt(data["accuracy"])}</td><td>{data["supported_class_count"]}/4</td></tr>')
            for label,v in data['classes'].items():
                classes.append('<tr>'+''.join(f'<td>{esc(str(x))}</td>' for x in [limb,label,v['support'],v['predicted'],fmt(v['precision']),fmt(v['recall']),fmt(v['f1'])])+'</tr>')
            matrix=data['confusion_matrix_true_rows_predicted_columns']
            header=''.join(f'<th>{c}</th>' for c in CLASSES)
            body=''.join('<tr><th>'+CLASSES[i]+'</th>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for i,row in enumerate(matrix))
            matrices.append(f'<h4>{limb}</h4><div class="scroll"><table><thead><tr><th>True / predicted</th>{header}</tr></thead><tbody>{body}</tbody></table></div>')
        cards=[]
        for row,pred in examples.get(name,[]):
            description='; '.join(f'{limb}: {row[limb]} → {CLASSES[pred[i]]}' for i,limb in enumerate(LIMBS))
            cards.append(f'<figure><img src="{esc(Path(row["preview_filepath"]).as_uri())}" alt="Source {row["source_record"]}"><figcaption><a href="{esc(Path(row["filepath"]).as_uri())}">Source {row["source_record"]}</a>, {esc(row["casualty_id"])}<p>{esc(description)}</p></figcaption></figure>')
        sections.append(f'''<section><h2>{esc(name)}</h2><table><thead><tr><th>Limb</th><th>Macro-F1</th><th>Accuracy</th><th>Classes with true support</th></tr></thead><tbody>{''.join(limb_rows)}</tbody></table>
        <details><summary>Per-class precision, recall and F1</summary><div class="scroll"><table><thead><tr><th>Limb</th><th>Class</th><th>True count</th><th>Predicted count</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{''.join(classes)}</tbody></table></div></details>
        <details><summary>Confusion matrices</summary>{''.join(matrices)}</details>
        <p><a href="{name}_predictions.jsonl">Every validation prediction</a> · <a href="{name}_metrics.json">Complete metrics</a></p>
        {('<details><summary>Examples with the most incorrect limbs, from different casualties</summary><div class="gallery">'+''.join(cards)+'</div></details>') if cards else ''}</section>''')
    policy=next(iter(models.values()))['metric_policy']
    unsupported='; '.join(f'{x["limb"]}/{x["class"]}: only casualty {x["sole_casualty"]}, kept in training' for x in split['unsupported_validation_classes'])
    source_hash=next(v for p,v in context['file_hashes'].items() if Path(p).name=='train_dataset.csv')
    note=f'''Validation was used for checkpoint selection, so these are development results. The split was selected using label coverage and group sizes before fitting any model. {unsupported}.
    Annotation review is deferred; {comparison['flagged_validation_images']} validation images have unresolved AI review flags, and all remain included with original labels. One exact duplicate pair remains within training. The shared whole-image backbone has no explicit limb detector or anatomy constraint. These scores do not establish left-arm-amputation performance on unseen casualties.'''
    lines+=['',policy,'',note,'',f'Selected learned stage: {comparison["selected_learned_stage"]}.',
            'Canonical CSV and original media unchanged. No automatic label corrections or video frames used.',
            f'Source CSV SHA-256: {source_hash}',f'Input fingerprint: {context["input_fingerprint"]}']
    (run/'summary.txt').write_text('\n'.join(lines)+'\n')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Baseline comparison</title>
    <style>body{{font:16px/1.5 system-ui,sans-serif;color:#17212b;background:#f5f7f9;max-width:1150px;margin:32px auto;padding:0 20px}}section,.intro{{background:white;border:1px solid #d6dce2;border-radius:8px;padding:20px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #d6dce2}}thead{{background:#edf2f6}}a{{color:#075faa}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:600}}.scroll{{overflow-x:auto}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}}figure{{margin:0}}img{{width:100%;max-width:448px}}figcaption{{font-size:13px}}code{{overflow-wrap:anywhere}}</style>
    <h1>Image-only baseline comparison</h1><div class="intro"><p>{split['train']['images']} training images from {len(split['train']['casualties'])} casualties. {split['validation']['images']} validation images from {len(split['validation']['casualties'])} separate casualties. All 952 available images retained.</p>
    <table><thead><tr><th>Model</th><th>Mean limb macro-F1</th><th>Label accuracy</th><th>All four correct</th><th>Mean casualty label accuracy</th></tr></thead><tbody>{''.join(table)}</tbody></table>
    <p>{esc(policy)}</p><p>{esc(note)}</p><p>Frozen: train four linear classifiers on fixed pretrained features. Fine-tuned: initialize from frozen checkpoint, update final ResNet block and heads, retain fixed BatchNorm statistics. Input: full-frame 448-pixel RGB previews and ImageNet normalization. No crop, flip, augmentation or label weighting.</p>
    <p><a href="split_manifest.csv">Fixed split manifest</a> · <a href="split_summary.json">Class coverage</a> · <a href="training_config.json">Training configuration</a> · <a href="comparison.json">Comparison JSON</a> · <a href="summary.txt">Text summary</a> · <a href="smoke.json">Tiny-batch check</a></p>
    <p>Source CSV SHA-256: <code>{source_hash}</code></p><p>Canonical source files and labels unchanged. Artifacts identify input and checkpoint hashes.</p></div>{''.join(sections)}</html>'''
    (run/'index.html').write_text(page)


def predict(args):
    seed_everything(42)
    model,info=load_checkpoint(args.checkpoint)
    device=choose_device(args.device)
    model.to(device).eval()
    original,metadata=load_image(args.image.resolve())
    try:preview,_=make_preview(original,448)
    finally:original.close()
    try:pixels=torch.from_numpy(np.array(preview,dtype=np.uint8,copy=True).transpose(2,0,1).copy()).unsqueeze(0)
    finally:preview.close()
    with torch.inference_mode():
        scores=model(ImageNormalizer().to(device)(pixels.to(device))).softmax(-1)[0].cpu()
    result={'image':str(args.image.resolve()),'image_sha256':metadata['image_sha256'],
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':file_hash(args.checkpoint),
            'stage':info['stage'],'input_fingerprint':info['input_fingerprint'],
            'prediction':{limb:CLASSES[int(scores[i].argmax())] for i,limb in enumerate(LIMBS)},
            'probabilities':{limb:dict(zip(CLASSES,scores[i].tolist())) for i,limb in enumerate(LIMBS)},
            'probabilities_are_calibrated':False}
    if args.json_output:
        target=args.json_output.resolve()
        source_roots=[(PROJECT/'train').resolve()]
        run_inputs=args.checkpoint.resolve().parent/'inputs.json'
        if run_inputs.is_file():
            source_roots.append(Path(json.loads(run_inputs.read_text())['dataset_root']).resolve())
        if target.exists() or any(target.is_relative_to(root) for root in source_roots):
            raise ValueError('Prediction output must be a new file outside the dataset tree')
        write_json(target,result)
    print(json.dumps(result,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare',help='Validate inputs, save casualty split and majority baseline')
    prep.add_argument('--dataset-root',type=Path,default=PROJECT/'train')
    prep.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    prep.add_argument('--seed',type=int,default=42)
    prep.add_argument('--validation-fraction',type=float,default=.2)
    prep.add_argument('--split-candidates',type=int,default=2000)
    prep.set_defaults(function=prepare)
    check=subs.add_parser('smoke',help='Extract frozen features and overfit a tiny training batch')
    check.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    check.add_argument('--seed',type=int,default=42)
    check.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    check.add_argument('--batch-size',type=int,default=8)
    check.add_argument('--smoke-images',type=int,default=16)
    check.add_argument('--smoke-steps',type=int,default=500)
    check.set_defaults(function=smoke)
    fit=subs.add_parser('train',help='Train linear heads, then optionally fine-tune the final ResNet block')
    fit.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    fit.add_argument('--seed',type=int,default=42)
    fit.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    fit.add_argument('--stage',choices=['frozen','all'],default='all')
    fit.add_argument('--batch-size',type=int,default=8)
    fit.add_argument('--feature-batch-size',type=int,default=64)
    fit.add_argument('--head-epochs',type=int,default=100)
    fit.add_argument('--head-patience',type=int,default=20)
    fit.add_argument('--finetune-epochs',type=int,default=8)
    fit.add_argument('--finetune-patience',type=int,default=3)
    fit.add_argument('--head-lr',type=float,default=.001)
    fit.add_argument('--backbone-lr',type=float,default=.0001)
    fit.add_argument('--finetune-head-lr',type=float,default=.0003)
    fit.add_argument('--weight-decay',type=float,default=.0001)
    fit.set_defaults(function=train)
    score=subs.add_parser('evaluate',help='Evaluate saved checkpoints on the same fixed casualty holdout')
    score.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    score.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    score.add_argument('--batch-size',type=int,default=8)
    score.set_defaults(function=evaluate)
    infer=subs.add_parser('predict',help='Predict four limb labels from an original image')
    infer.add_argument('--checkpoint',type=Path,required=True)
    infer.add_argument('--image',type=Path,required=True)
    infer.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    infer.add_argument('--json-output',type=Path)
    infer.set_defaults(function=predict)
    return p


if __name__ == '__main__':
    args=parser().parse_args()
    for name in ['batch_size','feature_batch_size','head_epochs','finetune_epochs','head_patience',
                 'finetune_patience','smoke_images','smoke_steps','split_candidates']:
        if hasattr(args,name) and getattr(args,name)<1:
            raise SystemExit(f'{name} must be positive')
    for name in ['head_lr','backbone_lr','finetune_head_lr','weight_decay']:
        if hasattr(args,name):
            value=getattr(args,name)
            if not math.isfinite(value) or value<0 or (name!='weight_decay' and value==0):
                raise SystemExit(f'{name} must be finite and '+('nonnegative' if name=='weight_decay' else 'positive'))
    if hasattr(args,'seed') and not 0<=args.seed<2**32:
        raise SystemExit('seed must be between 0 and 2**32-1')
    args.function(args)
