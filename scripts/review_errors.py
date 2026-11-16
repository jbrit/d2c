#!/usr/bin/env python3
"""Offline original/crop gallery for every model-label error and annotation flag."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlencode

from PIL import Image, ImageOps

from audit_images import load_image
from baseline_data import CLASSES, LIMBS, file_hash, metrics, write_json
from error_review_data import (BASE, CROP, PROJECT, REVIEW, SEPARATE,
                               check_preservation, collect)

DEFAULT_OUTPUT=PROJECT/'runs/error_review_v1'
LIMB_NAMES={'arm_left':'Left arm','arm_right':'Right arm','leg_left':'Left leg','leg_right':'Right leg'}
SHORT_LIMBS={'arm_left':'AL','arm_right':'AR','leg_left':'LL','leg_right':'LR'}
esc=lambda value:html.escape(str(value),quote=True)


def relative(target,page):
    return quote(Path(os.path.relpath(Path(target),Path(page).parent)).as_posix(),safe='/')


def case_name(source):return f'source_{source:06d}.html'


def score(value):return '—' if value is None else f'{value*100:.2f}%'


def result_rows(record,models):
    rows=[]
    for model in models:
        key=model['key'];result=record['models'][key]
        if result['correct_limbs']!=sum(result['prediction'][l]==record['truth'][l] for l in LIMBS):
            raise ValueError('Gallery correctness differs from model output')
        for limb in LIMBS:
            truth=record['truth'][limb];prediction=result['prediction'][limb]
            p=result.get('probabilities',{}).get(limb)
            rows.append({'model':key,'model_name':model['name'],'limb':limb,'supplied_label':truth,
                         'predicted_label':prediction,'is_error':truth!=prediction,
                         'predicted_score':None if p is None else p[prediction],
                         'supplied_label_score':None if p is None else p[truth],
                         **{'p_'+c:None if p is None else p[c] for c in CLASSES}})
    return rows


def summarize(records,models):
    counts={}
    for split in ['validation','train']:
        selected=[r for r in records if r['split']==split]
        counts[split]={'images':len(selected),'any_model_error_images':sum(r['wrong_models']>0 for r in selected),
                       'any_learned_error_images':sum(r['learned_wrong_models']>0 for r in selected),
                       'annotation_flagged_images':sum(bool(r['annotation_issues']) for r in selected),
                       'wrong_model_limb_decisions':sum(r['wrong_model_limbs'] for r in selected)}
    counts['models']=len(models);counts['images']=len(records)
    counts['limb_decisions']=len(records)*len(models)*len(LIMBS)
    counts['labels_modified']=0
    return counts


def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def exports(output,records,models):
    all_rows=[];file_rows=[];class_rows=[]
    fields=['source_record','filename','split','casualty_id','original_filepath','case_page','crop_status','crop_reason',
            'model','model_name','limb','supplied_label','predicted_label','is_error','predicted_score','supplied_label_score',
            *['p_'+c for c in CLASSES],'annotation_assessment','annotation_proposed_label']
    for record in records:
        common={'source_record':record['source_record'],'filename':Path(record['filepath']).name,
                'split':record['split'],'casualty_id':record['casualty_id'],'original_filepath':record['filepath'],
                'case_page':'cases/'+case_name(record['source_record']),'crop_status':record['crop']['status'],
                'crop_reason':record['crop']['reason']}
        issues={x['limb']:x for x in record['annotation_issues']}
        for row in result_rows(record,models):
            issue=issues.get(row['limb'],{})
            all_rows.append({**common,**row,'annotation_assessment':issue.get('assessment',''),
                             'annotation_proposed_label':issue.get('proposed_label','')})
        file_rows.append({**common,'wrong_models':record['wrong_models'],
                          'wrong_model_limb_decisions':record['wrong_model_limbs'],
                          'learned_wrong_models':record['learned_wrong_models'],
                          'annotation_flag_count':len(record['annotation_issues']),**record['truth']})
    write_csv(output/'all_predictions.csv',all_rows,fields)
    wrong=[r for r in all_rows if r['is_error']]
    write_csv(output/'mistakes.csv',wrong,fields)
    file_fields=['source_record','filename','split','casualty_id','original_filepath','case_page',
                 'wrong_models','learned_wrong_models','wrong_model_limb_decisions','annotation_flag_count',
                 'crop_status','crop_reason',*LIMBS]
    write_csv(output/'files_with_errors.csv',[r for r in file_rows if r['wrong_models']],file_fields)
    write_csv(output/'all_files.csv',file_rows,file_fields)
    by_split={}
    for split in ['validation','train']:
        selected=[r for r in records if r['split']==split];by_split[split]={}
        for model in models:
            key=model['key']
            truth=[[CLASSES.index(r['truth'][l]) for l in LIMBS] for r in selected]
            pred=[[CLASSES.index(r['models'][key]['prediction'][l]) for l in LIMBS] for r in selected]
            result=metrics(truth,pred,[r['casualty_id'] for r in selected]);by_split[split][key]=result
            for limb,data in result['by_limb'].items():
                matrix=data['confusion_matrix_true_rows_predicted_columns']
                for index,label in enumerate(CLASSES):
                    values=data['classes'][label]
                    class_rows.append({'split':split,'model':key,'model_name':model['name'],'limb':limb,'class':label,
                                       'true_count':values['support'],'predicted_count':values['predicted'],
                                       'true_positives':matrix[index][index],'false_negatives':values['support']-matrix[index][index],
                                       'false_positives':values['predicted']-matrix[index][index],
                                       'precision':values['precision'],'recall':values['recall'],'f1':values['f1']})
    write_csv(output/'class_results.csv',class_rows,list(class_rows[0]))
    write_json(output/'split_metrics.json',by_split)
    return len(wrong),class_rows


def make_assets(output,records):
    directories=[output/'images/original',output/'images/cropped',output/'images/thumbs']
    for directory in directories:directory.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'image_manifest.json'
    existing=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    assets={}
    for index,record in enumerate(records):
        sid=record['source_record'];name=f'source_{sid:06d}.jpg'
        original_path=directories[0]/name;crop_path=directories[1]/name;thumb_path=directories[2]/name
        expected={'image_sha256':record['image_sha256'],'crop_box':record['crop']['crop_box'],
                  'display_max_side':1600,'display_jpeg_quality':92}
        old=existing.get(str(sid))
        if old and old['identity']==expected and all((output/p).is_file() and file_hash(output/p)==sha for p,sha in old['files'].items()):
            assets[str(sid)]=old;continue
        original,metadata=load_image(Path(record['filepath']))
        try:
            if metadata['image_sha256']!=record['image_sha256']:
                raise ValueError('Original image changed while generating gallery')
            size=list(original.size)
            with ImageOps.contain(original,(1600,1600),Image.Resampling.LANCZOS) as display:
                display.save(original_path,quality=92,subsampling=0)
            with ImageOps.contain(original,(320,320),Image.Resampling.LANCZOS) as thumb:
                thumb.save(thumb_path,quality=85)
            if record['crop']['status']=='cropped':
                with original.crop(tuple(record['crop']['crop_box'])) as crop:
                    with ImageOps.contain(crop,(1600,1600),Image.Resampling.LANCZOS) as display:
                        display.save(crop_path,quality=92,subsampling=0)
            else:
                crop_path=original_path
        finally:
            original.close()
        paths=[original_path,thumb_path]+([crop_path] if crop_path!=original_path else [])
        assets[str(sid)]={'identity':expected,'oriented_native_size':size,
                         'original':str(original_path.relative_to(output)),
                         'cropped':str(crop_path.relative_to(output)),
                         'thumb':str(thumb_path.relative_to(output)),
                         'files':{str(path.relative_to(output)):file_hash(path) for path in paths}}
        if (index+1)%50==0:
            write_json(manifest_path,assets)
            print(f'Prepared original/crop display images {index+1}/{len(records)}',flush=True)
    write_json(manifest_path,assets)
    return assets


def header(title,css='../review.css'):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><link rel="stylesheet" href="{css}"></head>')


def case_page(output,record,models,assets,previous,next_source,cautions):
    sid=record['source_record'];path=output/'cases'/case_name(sid);split=record['split'];asset=assets[str(sid)]
    crop=record['crop'];filename=Path(record['filepath']).name
    links=[f'<a href="../index.html">Validation errors</a>',f'<a href="../training.html">Training errors</a>',
           f'<a href="../all.html">All errors</a>',f'<a href="../label_results.html">Results by label</a>',
           f'<a href="../annotation_flags.html">Annotation flags</a>']
    navigation='<div class="navigation">'+(f'<a href="{case_name(previous)}">← Previous error image</a>' if previous else '<span></span>')+(f'<a href="{case_name(next_source)}">Next error image →</a>' if next_source else '<span></span>')+'</div>'
    matrix='<div class="scroll"><table class="matrix"><thead><tr><th>Model</th>'+''.join(f'<th>{LIMB_NAMES[l]}</th>' for l in LIMBS)+'</tr></thead><tbody>'
    matrix+='<tr><th>Supplied annotation</th>'+''.join(f'<td><strong>{esc(record["truth"][l])}</strong></td>' for l in LIMBS)+'</tr>'
    for model in models:
        key=model['key'];result=record['models'][key]
        matrix+=f'<tr><th>{esc(model["name"])}<br><small>{result["correct_limbs"]}/4 match supplied labels</small></th>'
        for limb in LIMBS:
            truth=record['truth'][limb];pred=result['prediction'][limb];bad=truth!=pred
            probabilities=result.get('probabilities',{}).get(limb)
            confidence=None if probabilities is None else probabilities[pred]
            matrix+=(f'<td class="{"bad" if bad else "good"}"><span class="status">{"MISMATCH" if bad else "MATCH"}</span>'
                     f'<span class="prediction">{esc(pred)}</span><small>{("Supplied: "+esc(truth)+"<br>") if bad else ""}'
                     f'{("Predicted score: "+score(confidence)) if confidence is not None else "Deterministic majority rule"}</small></td>')
        matrix+='</tr>'
    matrix+='</tbody></table></div>'
    score_sections=[]
    for limb in LIMBS:
        rows=[]
        for model in models:
            result=record['models'][model['key']];prediction=result['prediction'][limb];truth=record['truth'][limb]
            p=result.get('probabilities',{}).get(limb)
            values=[model['name'],truth,prediction,'Mismatch' if prediction!=truth else 'Match',
                    score(None if p is None else p[prediction]),score(None if p is None else p[truth]),
                    *[score(None if p is None else p[c]) for c in CLASSES]]
            rows.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in values)+'</tr>')
        score_sections.append(f'<details><summary>{LIMB_NAMES[limb]} · supplied {esc(record["truth"][limb])} · all class scores</summary><div class="scroll"><table class="score-table"><thead><tr>'
                              +''.join(f'<th>{esc(x)}</th>' for x in ['Model','Supplied','Predicted','Result','Predicted score','Supplied-class score',*CLASSES])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></details>')
    issues=record['annotation_issues'];issue_html=''
    if issues:
        rows=[]
        for issue in issues:
            evidence=REVIEW/issue['evidence_filepath']
            values=[LIMB_NAMES[issue['limb']],issue['original_label'],issue['proposed_label'],issue['assessment'],issue['reason'],issue['human_decision'] or 'Unresolved']
            rows.append('<tr>'+''.join(f'<td class="{"reason" if i==4 else ""}">{esc(v)}</td>' for i,v in enumerate(values))+f'<td><a href="{relative(evidence,path)}">Earlier evidence</a></td></tr>')
        issue_html='<section class="notice"><h2>Annotation-review flags</h2><p>These are earlier AI review proposals. Human review status is shown below. Supplied labels are used for every mismatch count; this gallery applies no proposed changes.</p><div class="scroll"><table class="annotation-table"><tr><th>Limb</th><th>Supplied</th><th>Proposed</th><th>Assessment</th><th>Reason</th><th>Human decision</th><th>Evidence</th></tr>'+''.join(rows)+'</table></div></section>'
    if sid in cautions:
        issue_html+='<section class="notice"><h2>Crop-review caution</h2><p>'+esc(cautions[sid])+'</p></section>'
    width,height=asset['oriented_native_size'];box=crop['crop_box']
    original_url='../'+asset['original'];crop_url='../'+asset['cropped']
    scope='Training image — used for fitting; this result is in-sample.' if split=='train' else 'Development validation image — held out from fitting; used for checkpoint selection.'
    page=header(f'{filename} · source {sid}')+'<body>'
    page+=f'<nav>{" ".join(links)}</nav>{navigation}<h1>{esc(filename)} · source {sid}</h1><p><span class="badge split-{split}">{esc(split)}</span> {esc(record["casualty_id"])} · {record["wrong_models"]}/9 variants have an error · {record["wrong_model_limbs"]}/36 model/limb mismatches</p>'
    page+=f'<p class="notice">{scope} Laterality is the casualty’s anatomical left/right.</p>'
    page+='<section><div class="pair"><figure><h2>Original image</h2>'
    page+=f'<a href="{original_url}"><img src="{original_url}" alt="Original oriented image, source {sid}"></a><figcaption>{width} × {height} native pixels · <a href="{relative(record["filepath"],path)}">Open original native file</a></figcaption></figure>'
    page+=f'<figure><h2>{"Padded casualty crop" if crop["status"]=="cropped" else "Full-frame fallback — no crop"}</h2><a href="{crop_url}"><img src="{crop_url}" alt="Crop region used by cropped variants, source {sid}"></a><figcaption>{box[2]-box[0]} × {box[3]-box[1]} native crop pixels · {crop["linear_magnification"]:.2f}× magnification · {esc(crop["reason"])}</figcaption></figure></div>'
    page+=f'<p>Display copies are limited to 1,600 pixels on the longest side. The original native file above retains full resolution. Crop coordinates after EXIF orientation: {esc(box)}; 30% padding on each side of the detector extent.</p>'
    page+=f'<p><a href="{relative(CROP/"overlays"/f"source_{sid:06d}.jpg",path)}">View crop boundary on original</a></p>'
    page+='<details class="comparison-inputs"><summary>Actual classifier inputs (448 × 448), full frame and cropped</summary><div class="pair">'
    for title,input_path in [('Full-frame model input',record['full_frame_input']),('Cropped-model input',record['cropped_input'])]:
        url=relative(input_path,path);page+=f'<figure><h3>{title}</h3><a href="{url}"><img src="{url}" alt="{title}"></a></figure>'
    page+='</div></details></section>'+issue_html
    page+='<section id="predictions"><h2>Every model’s result for every limb</h2><p>Red cells disagree with the supplied annotation; green cells match it. A mismatch does not by itself establish an annotation error.</p>'+matrix+'</section>'
    page+='<section><h2>Results for each class</h2><p>Scores are uncalibrated softmax outputs. The majority rule has no probability estimate.</p>'+''.join(score_sections)+'</section>'
    page+=f'<section><h2>File provenance</h2><p class="source-path">{esc(record["filepath"])}</p><p>Source ID is the one-based parsed CSV data-record number, excluding the header.</p><p class="source-path">Image SHA-256: {esc(record["image_sha256"])}</p><p><a href="{relative(PROJECT/"train/train_dataset.csv",path)}">Canonical source manifest</a> · <a href="../records.json">Complete review data</a></p></section>{navigation}</body></html>'
    path.write_text(page)


def controls(records,models,split):
    options=lambda choices,selected='all':''.join(f'<option value="{esc(key)}"'+(' selected' if key==selected else '')+f'>{esc(label)}</option>' for key,label in choices)
    select=lambda id,label,choices,selected='all':f'<label>{label}<select id="{id}">{options(choices,selected)}</select></label>'
    parts=[select('split','Image split',[('validation','Validation (development)'),('train','Training (in-sample)'),('all','Both splits')],split),
           select('model','Model',[('all','Any variant, including majority'),('learned','Any learned variant')]+[(m['key'],m['name']) for m in models]),
           select('limb','Limb',[('all','Any limb')]+list(LIMB_NAMES.items())),
           select('truth','Supplied label',[('all','Any supplied label')]+[(c,c) for c in CLASSES]),
           select('prediction','Predicted label',[('all','Any predicted label')]+[(c,c) for c in CLASSES]),
           select('casualty','Casualty',[('all','Any casualty')]+[(x,x) for x in sorted({r['casualty_id'] for r in records})]),
           select('issue','Review flags',[('all','Any review status'),('flagged','Has annotation-review flag'),('likely_mismatch','Likely annotation mismatch'),('uncertain','Uncertain annotation'),('crop','Has crop-review caution')]),
           '<label>Find file / source ID / casualty<input id="search" type="search" placeholder="e.g. 0184.jpg or 184"></label>',
           select('sort','Sort',[('errors','Most matching errors'),('source','Source record'),('confidence','Highest wrong prediction score'),('magnification','Largest crop magnification')],'errors'),
           select('perpage','Images per page',[('24','24'),('48','48'),('all','All matching images')],'24')]
    return ('<section id="interactive" hidden><div class="filters">'+''.join(parts)+'</div><div class="actions"><label class="checkbox"><input id="errors" type="checkbox" checked> Only prediction mismatches</label><button id="reset">Reset filters</button><button id="export-filtered">Export filtered comparisons</button></div></section>')


def index_page(output,records,models,assets,summary,split,cautions):
    filename={'validation':'index.html','train':'training.html','all':'all.html'}[split]
    compact=[];rows=[]
    for record in records:
        sid=record['source_record'];flat=result_rows(record,models);asset=assets[str(sid)]
        compact.append({'id':sid,'filename':Path(record['filepath']).name,'filepath':record['filepath'],
                        'split':record['split'],'casualty':record['casualty_id'],'gain':record['crop']['linear_magnification'],
                        'cropCaution':sid in cautions,'issues':[i['assessment'] for i in record['annotation_issues']],
                        'results':[[r['model'],r['limb'],r['supplied_label'],r['predicted_label'],r['predicted_score']] for r in flat]})
        case='cases/'+case_name(sid)
        labels=''.join(f'<div><b>{SHORT_LIMBS[l]}</b> {esc(record["truth"][l])}</div>' for l in LIMBS)
        flags=(f'<span class="badge flag">{len(record["annotation_issues"])} annotation flags</span>' if record['annotation_issues'] else '')
        if sid in cautions:flags+='<span class="badge flag">Crop caution</span>'
        wrong=[r for r in flat if r['is_error']]
        snippet='\n'.join(f'{r["model_name"]}: {r["limb"]} {r["supplied_label"]} → {r["predicted_label"]}' for r in wrong[:3])
        rows.append(f'<tr data-record="{sid}" data-split="{record["split"]}" class="{"no-errors" if not record["wrong_models"] else "has-errors"}"><td><a href="{case}"><img class="thumb" loading="lazy" src="{esc(asset["thumb"])}" alt="Original image {sid}"></a></td>'
                    f'<td><a class="filename" href="{case}">{esc(Path(record["filepath"]).name)}</a><br>Source {sid}<br><span class="badge split-{record["split"]}">{record["split"]}</span><br>{esc(record["casualty_id"])}<br>{flags}</td>'
                    f'<td class="labels">{labels}</td><td>{record["wrong_models"]}/9 variants<br>{record["wrong_model_limbs"]}/36 decisions<br><small>{esc(record["crop"]["status"])} · {record["crop"]["linear_magnification"]:.2f}×</small></td><td class="matching-results">{esc(snippet)}</td></tr>')
    data=json.dumps({'records':compact,'modelNames':{m['key']:m['name'] for m in models}},separators=(',',':'),allow_nan=False).replace('<',r'\u003c')
    page=header('Every model error · casualty image review','review.css')+f'<body data-default-split="{split}"><h1>Every model error</h1><p>Original image, padded crop, supplied labels and all nine saved variants’ predictions.</p>'
    page+='<nav><a href="index.html">Validation errors</a><a href="training.html">Training errors</a><a href="all.html">All errors</a><a href="label_results.html">Results by label</a><a href="annotation_flags.html">Annotation flags</a><a href="files_with_errors.csv">Files with any error (CSV)</a><a href="mistakes.csv">Every mismatch (CSV)</a><a href="all_predictions.csv">All predictions and class scores (CSV)</a></nav>'
    page+=f'<section class="intro"><div class="statline"><span>Validation: <b>{summary["validation"]["any_model_error_images"]}/{summary["validation"]["images"]}</b> images with an error</span><span>Training: <b>{summary["train"]["any_model_error_images"]}/{summary["train"]["images"]}</b> images with an error</span><span><b>9</b> selected model variants</span></div>'
    page+='<p>Includes the majority rule, full-frame and cropped shared models, and full-frame and cropped separate-limb pipelines, at frozen and fine-tuned stages. A file enters the error queue if any variant disagrees with any supplied limb label.</p><p class="notice">Validation is the same development holdout used for checkpoint selection. Training results are in-sample and shown separately. Annotation flags are unresolved AI review proposals; labels are unchanged.</p><p>AL/AR/LL/LR mean anatomical left arm/right arm/left leg/right leg. The crop column uses the final padded crop with full-frame fallback. Model scores are uncalibrated.</p></section>'
    page+=controls(records,models,split)
    page+='<section><div class="actions"><strong id="match-count">All error images for this split</strong><span id="page-count"></span><button id="previous">Previous page</button><button id="next">Next page</button></div><p id="empty" hidden>No matching images. Change the filters or clear “Only prediction mismatches”.</p><noscript><p>JavaScript is disabled. Static error-file links for this split remain available; use the navigation above for other splits and CSV exports.</p></noscript><div class="scroll"><table class="file-table"><thead><tr><th>Original</th><th>File and casualty</th><th>Supplied labels</th><th>Errors across all variants</th><th>Matching comparisons</th></tr></thead><tbody id="file-rows">'+''.join(rows)+'</tbody></table></div></section>'
    page+=f'<script id="review-data" type="application/json">{data}</script><script src="review.js"></script></body></html>'
    (output/filename).write_text(page)


def label_page(output,rows):
    parts=[]
    for split in ['validation','train']:
        body=[]
        for r in rows:
            if r['split']!=split:continue
            params={'split':split,'model':r['model'],'limb':r['limb'],'truth':r['class'],'errors':'1'}
            missed='index.html#'+urlencode(params)
            params.pop('truth');params['prediction']=r['class'];false_calls='index.html#'+urlencode(params)
            values=[r['model_name'],LIMB_NAMES[r['limb']],r['class'],r['true_count'],r['predicted_count'],r['true_positives'],score(r['precision']),score(r['recall']),score(r['f1'])]
            body.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in values)+f'<td><a href="{missed}">{r["false_negatives"]} missed</a><br><a href="{false_calls}">{r["false_positives"]} false calls</a></td></tr>')
        parts.append(f'<section><h2>{"Validation — development holdout" if split=="validation" else "Training — in-sample"}</h2><div class="scroll"><table class="metrics-table"><thead><tr>'+''.join(f'<th>{x}</th>' for x in ['Model','Limb','Class','Supplied count','Predicted count','Correct','Precision','Recall','F1','View mistakes'])+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div></section>')
    page=header('Results for each limb and class','review.css')+'<body><nav><a href="index.html">Validation error gallery</a><a href="training.html">Training error gallery</a><a href="class_results.csv">Download label results CSV</a></nav><h1>Results for every limb and class</h1><p>Each count refers to supplied labels. “Missed” links show false negatives for that class; “false calls” links show false positives. No true support means recall/F1 is unmeasured (—). Training results are in-sample.</p>'+''.join(parts)+'</body></html>'
    (output/'label_results.html').write_text(page)


def annotation_page(output,records,assets):
    rows=[]
    for record in sorted(records,key=lambda r:r['source_record']):
        for issue in record['annotation_issues']:
            sid=record['source_record'];case='cases/'+case_name(sid)
            cells=[LIMB_NAMES[issue['limb']],issue['original_label'],issue['proposed_label'],issue['assessment'],issue['reason'],issue['human_decision'] or 'Unresolved']
            rows.append(f'<tr><td><a href="{case}"><img class="thumb" loading="lazy" src="{assets[str(sid)]["thumb"]}" alt="Source {sid}"></a><br><a href="{case}">{esc(Path(record["filepath"]).name)} · {sid}</a><br>{record["split"]}</td>'+''.join(f'<td>{esc(v)}</td>' for v in cells)+'</tr>')
    page=header('Existing annotation-review flags','review.css')+'<body><nav><a href="index.html">Validation errors</a><a href="all.html">All model errors</a><a href="label_results.html">Results by label</a></nav><h1>Existing annotation-review flags</h1><p class="notice">These are prior AI review concerns, not confirmed corrections or a complete list of annotation errors. Every source label is unchanged. These cases are shown even if every model matches the supplied labels.</p><section class="scroll"><table class="file-table"><thead><tr><th>Original / file</th><th>Limb</th><th>Supplied</th><th>Proposed</th><th>Assessment</th><th>Earlier review reason</th><th>Human decision</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></section></body></html>'
    (output/'annotation_flags.html').write_text(page)


def build(args):
    output=args.output.resolve()
    if output.is_relative_to(PROJECT/'train') or any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in [BASE,SEPARATE,CROP]):
        raise ValueError('Gallery must be outside canonical data and source experiments')
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'records.json').exists():records,models=collect(output,args.device)
    else:
        check_preservation(output)
        records=json.loads((output/'records.json').read_text());models=json.loads((output/'models.json').read_text())
    records.sort(key=lambda r:(r['split']!='validation',-r['wrong_model_limbs'],r['source_record']))
    summary=summarize(records,models);wrong_count,class_rows=exports(output,records,models)
    summary['wrong_model_limb_decisions']=wrong_count
    write_json(output/'summary.json',summary)
    assets=make_assets(output,records)
    review=json.loads((CROP/'crop_review.json').read_text());cautions={}
    for note in review.get('remaining_cautions',[]):
        for source in re.findall(r'Source\s*(\d+)',note,flags=re.IGNORECASE):cautions[int(source)]=note
    (output/'cases').mkdir(exist_ok=True)
    for split in ['validation','train']:
        queue=[r['source_record'] for r in records if r['split']==split and r['wrong_models']]
        positions={sid:i for i,sid in enumerate(queue)}
        for record in records:
            if record['split']!=split:continue
            sid=record['source_record'];position=positions.get(sid)
            previous=queue[position-1] if position is not None and position>0 else None
            next_source=queue[position+1] if position is not None and position+1<len(queue) else None
            case_page(output,record,models,assets,previous,next_source,cautions)
    for split in ['validation','train','all']:index_page(output,records,models,assets,summary,split,cautions)
    label_page(output,class_rows)
    annotation_page(output,records,assets)
    shutil.copyfile(PROJECT/'scripts/review_errors.js',output/'review.js')
    shutil.copyfile(PROJECT/'scripts/review_errors.css',output/'review.css')
    code=output/'code';code.mkdir(exist_ok=True)
    for name in ['error_review_data.py','review_errors.py','review_errors.js','review_errors.css']:
        shutil.copyfile(PROJECT/'scripts'/name,code/name)
    write_json(output/'build.json',{'source_code_sha256':{p.name:file_hash(p) for p in code.iterdir() if p.is_file()},
                                   'display_max_side':1600,'model_inputs_size':448,'models_retrained':False,
                                   'validation_results_reused':True,'training_results':'new inference on selected checkpoints in eval mode',
                                   'annotation_proposals_applied':False})
    check_preservation(output)
    print(json.dumps({'gallery':str(output/'index.html'),**summary},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--device',default='auto')
    build(p.parse_args())
