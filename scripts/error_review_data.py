"""Collect selected-checkpoint results without mutating any source experiment."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import torch

from baseline import load_run
from baseline_data import CLASSES, LIMBS, encode_labels, file_hash, majority_predictions, metrics, write_json
from baseline_model import choose_device, image_logits, load_checkpoint, seed_everything
from per_limb import tree_hashes
from per_limb_model import load_pipeline

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT/'runs/baseline_v1'
SEPARATE=PROJECT/'runs/per_limb_v1'
CROP=PROJECT/'runs/crop_v2'
REVIEW=PROJECT/'train/cleaned/annotation_review'


def model_specs():
    specs=[{'key':'majority','name':'Majority rule','input':'none','checkpoint':None,
            'validation':BASE/'majority_predictions.jsonl','metrics':BASE/'majority_metrics.json'}]
    for prefix,name,run,cropped,independent in [
            ('shared','Shared · full frame',BASE,False,False),
            ('per_limb','Separate limbs · full frame',SEPARATE,False,True),
            ('cropped_shared','Shared · cropped',CROP/'shared',True,False),
            ('cropped_per_limb','Separate limbs · cropped',CROP/'per_limb',True,True)]:
        for stage in ['frozen','finetuned']:
            key=prefix+'_'+stage
            stem=('per_limb_' if run==SEPARATE else '')+stage
            checkpoint=run/('pipeline.json' if stage=='finetuned' else 'frozen_pipeline.json') if independent else run/(stage+'.pt')
            specs.append({'key':key,'name':name+' · '+('frozen' if stage=='frozen' else 'fine-tuned'),
                          'input':'cropped' if cropped else 'full_frame','checkpoint':checkpoint,
                          'independent':independent,'stage':stage,
                          'validation':run/(stem+'_predictions.jsonl'),'metrics':run/(stem+'_metrics.json')})
    return specs


def pack_result(record,prediction,probabilities=None):
    if set(prediction)!=set(LIMBS) or any(label not in CLASSES for label in prediction.values()):
        raise ValueError('Invalid model prediction limb/class order')
    result={'prediction':dict(prediction),
            'correct_limbs':sum(prediction[limb]==record[limb] for limb in LIMBS)}
    if probabilities is not None:
        result['probabilities']=probabilities
    return result


def validate_saved(records, rows, stored_metrics):
    expected={r['source_record']:r for r in records}
    if len(rows)!=len(expected) or {r['source_record'] for r in rows}!=set(expected):
        raise ValueError('Validation predictions do not cover the exact held-out records')
    predictions=[];ordered=[]
    for row in rows:
        source=expected[row['source_record']]
        if (row['truth']!={limb:source[limb] for limb in LIMBS} or row['filepath']!=source['filepath'] or
                row['casualty_id']!=source['casualty_id']):
            raise ValueError('Prediction truth/source lineage mismatch')
        result=pack_result(source,row['prediction'],row.get('probabilities'))
        if result['correct_limbs']!=row['correct_limbs']:
            raise ValueError('Saved correctness count disagrees with supplied labels')
        predictions.append([CLASSES.index(row['prediction'][limb]) for limb in LIMBS]);ordered.append(source)
    actual=metrics(encode_labels(ordered),predictions,[r['casualty_id'] for r in ordered])
    for key,value in actual.items():
        if stored_metrics[key]!=value:
            raise ValueError(f'Prediction file and metrics disagree: {key}')


def source_snapshot(specs):
    files={}
    for spec in specs:
        for key in ['validation','metrics','checkpoint']:
            path=spec.get(key)
            if path is not None:files[str(path)]=file_hash(path)
        if spec.get('independent'):
            bundle=json.loads(spec['checkpoint'].read_text())
            for entry in bundle['checkpoints']:
                path=spec['checkpoint'].parent/entry['file'];sha=file_hash(path)
                if sha!=entry['sha256']:raise ValueError('Checkpoint bundle hash mismatch')
                files[str(path)]=sha
    for path in [BASE/'records.json',CROP/'shared/records.json',CROP/'crop_manifest.json',CROP/'crop_config.json',
                 REVIEW/'suspected_label_issues.csv',CROP/'crop_review.json']:
        files[str(path)]=file_hash(path)
    return files


def collect(output:Path,device_name='auto'):
    full,full_context=load_run(BASE)
    cropped,crop_context=load_run(CROP/'shared')
    if len(full)!=len(cropped):raise ValueError('Cropped/full-frame input counts differ')
    for a,b in zip(full,cropped):
        if any(a[k]!=b[k] for k in ['source_record','casualty_id','split','filepath','image_sha256',*LIMBS]):
            raise ValueError('Cropped/full-frame source/label/split mismatch')
    protected={str(run):tree_hashes(run) for run in [BASE,SEPARATE,CROP/'shared',CROP/'per_limb']}
    specs=model_specs();sources=source_snapshot(specs)
    identity={'source_files':sources,'records':file_hash(BASE/'records.json'),'crop_records':file_hash(CROP/'shared/records.json')}
    identity_path=output/'source_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Review source files changed; use a new output directory')
    write_json(identity_path,identity)
    protected_path=output/'protected_runs.json'
    if protected_path.exists() and json.loads(protected_path.read_text())!=protected:
        raise ValueError('Previous experiments changed during error review')
    write_json(protected_path,protected)
    cache=output/'predictions';cache.mkdir(exist_ok=True)
    by_id={r['source_record']:{'source_record':r['source_record'],'casualty_id':r['casualty_id'],
                              'split':r['split'],'filepath':r['filepath'],'image_sha256':r['image_sha256'],
                              'truth':{limb:r[limb] for limb in LIMBS},'models':{},
                              'annotation_review_flags':r['annotation_review_flags'],'crop':c['crop'],
                              'full_frame_input':r['preview_filepath'],'cropped_input':c['preview_filepath']}
           for r,c in zip(full,cropped)}
    seed_everything(42);device=choose_device(device_name)
    for spec in specs:
        key=spec['key'];saved=[json.loads(line) for line in spec['validation'].read_text().splitlines()]
        val=[r for r in full if r['split']=='validation']
        validate_saved(val,saved,json.loads(spec['metrics'].read_text()))
        for row in saved:
            by_id[row['source_record']]['models'][key]=pack_result(by_id[row['source_record']]['truth'],row['prediction'],row.get('probabilities'))
        path=cache/(key+'_train.json')
        input_records=cropped if spec['input']=='cropped' else full
        train=[r for r in input_records if r['split']=='train']
        if path.exists():
            training=json.loads(path.read_text())
            if [r['source_record'] for r in training]!=[r['source_record'] for r in train]:
                raise ValueError('Training inference cache source order mismatch')
            print(f'Reusing {key} training predictions',flush=True)
        else:
            print(f'Scoring {key}: {len(train)} training images; validation results reused unchanged',flush=True)
            if key=='majority':
                predicted=majority_predictions(train,len(train));probabilities=None
            else:
                if spec['independent']:model,_=load_pipeline(spec['checkpoint'])
                else:model,_=load_checkpoint(spec['checkpoint'])
                logits=image_logits(model,train,device,8)
                predicted=logits.argmax(-1).tolist();probabilities=logits.softmax(-1).tolist()
                model.cpu();del model
                if device.type=='mps':torch.mps.empty_cache()
            training=[]
            for index,(row,pred) in enumerate(zip(train,predicted)):
                prediction={limb:CLASSES[pred[i]] for i,limb in enumerate(LIMBS)}
                probability=None if probabilities is None else {limb:dict(zip(CLASSES,probabilities[index][i])) for i,limb in enumerate(LIMBS)}
                training.append({'source_record':row['source_record'],**pack_result(row,prediction,probability)})
            write_json(path,training)
        for row in training:
            by_id[row['source_record']]['models'][key]={k:v for k,v in row.items() if k!='source_record'}
        if any(key not in r['models'] for r in by_id.values()):raise ValueError('Incomplete model coverage')
    issues={}
    with (REVIEW/'suspected_label_issues.csv').open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            source=int(row['source_record']);record=by_id[source]
            if row['image_sha256']!=record['image_sha256'] or row['original_label']!=record['truth'][row['limb']]:
                raise ValueError('Annotation-review flag no longer matches canonical label')
            issues.setdefault(source,[]).append(row)
    records=[]
    for row in by_id.values():
        row['annotation_issues']=issues.get(row['source_record'],[])
        row['wrong_models']=sum(r['correct_limbs']<4 for r in row['models'].values())
        row['wrong_model_limbs']=sum(4-r['correct_limbs'] for r in row['models'].values())
        row['learned_wrong_models']=sum(r['correct_limbs']<4 for key,r in row['models'].items() if key!='majority')
        records.append(row)
    write_json(output/'records.json',records)
    exported=[{k:(str(v) if isinstance(v,Path) else v) for k,v in spec.items()} for spec in specs]
    write_json(output/'models.json',exported)
    check_preservation(output)
    return records,exported


def check_preservation(output):
    for name,sha in json.loads((output/'source_identity.json').read_text())['source_files'].items():
        if file_hash(Path(name))!=sha:raise ValueError(f'Review source changed: {name}')
    for name,hashes in json.loads((output/'protected_runs.json').read_text()).items():
        if tree_hashes(Path(name))!=hashes:raise ValueError(f'Protected model run changed: {name}')
    load_run(BASE,verify_previews=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=PROJECT/'runs/error_review_v1');p.add_argument('--device',default='auto')
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(PROJECT/'train') or any(args.output==r or args.output.is_relative_to(r) or r.is_relative_to(args.output) for r in [BASE,SEPARATE,CROP]):
        raise ValueError('Error review output must be outside data and source experiments')
    args.output.mkdir(parents=True,exist_ok=True)
    collect(args.output,args.device)
