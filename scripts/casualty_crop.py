"""Conservative, label-independent casualty crops in oriented native coordinates."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn, fasterrcnn_resnet50_fpn_v2

from audit_images import make_preview
from baseline_data import file_hash


POLICY = {'detector': 'fcos_resnet50_fpn', 'weights': 'COCO_V1',
          'detector_max_side': 1333, 'score_threshold': .65, 'margin_each_side': .3,
          'minimum_box_area_fraction': .02, 'ambiguous_second_area_fraction': .5,
          'ambiguous_intersection_over_smaller': .3, 'maximum_crop_area_fraction': .98,
          'output_size': 448, 'padding_rgb': [114, 114, 114],
          'fallback': 'full oriented original image; no dropped records',
          'selection': 'largest confident person box; separated similarly-sized people trigger fallback',
          'coverage_detector': 'fasterrcnn_resnet50_fpn_v2 COCO_V1; CPU; min_size480/max_size800',
          'coverage_score_threshold': .9, 'coverage_min_intersection_over_smaller': .3,
          'small_edge_maximum_box_area_fraction': .2, 'image_edge_tolerance_fraction': .01,
          'labels_used_for_crop': False}


def area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def choose_crop(size, detections, policy=POLICY):
    width, height = size
    full = [0, 0, width, height]
    result = {'crop_box': full, 'selected_box': None, 'selected_score': None,
              'status': 'fallback', 'reason': 'no_confident_person', 'area_fraction': 1.,
              'linear_magnification': 1.}
    candidates = []
    for item in detections:
        box, score = item['box'], item['score']
        if (len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]) or
                score < policy['score_threshold']):
            continue
        clipped = [max(0., min(width, box[0])), max(0., min(height, box[1])),
                   max(0., min(width, box[2])), max(0., min(height, box[3]))]
        if area(clipped) > 0:
            candidates.append({'box': clipped, 'score': score})
    candidates.sort(key=lambda x: (area(x['box']), x['score']), reverse=True)
    if not candidates:
        return result
    selected = candidates[0]; box = selected['box']
    result.update(selected_box=box, selected_score=selected['score'])
    if area(box) < width * height * policy['minimum_box_area_fraction']:
        result['reason'] = 'detection_too_small'
        return result
    edge=policy['image_edge_tolerance_fraction']
    if (area(box)<width*height*policy['small_edge_maximum_box_area_fraction'] and
            (box[0]<=width*edge or box[1]<=height*edge or box[2]>=width*(1-edge) or box[3]>=height*(1-edge))):
        result['reason']='small_edge_detection'
        return result
    for other in candidates[1:]:
        b = other['box']
        intersection = area([max(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), min(box[3], b[3])])
        if (area(b) >= area(box) * policy['ambiguous_second_area_fraction'] and
                intersection / min(area(b), area(box)) < policy['ambiguous_intersection_over_smaller']):
            result['reason'] = 'ambiguous_multiple_people'
            return result
    dx = (box[2] - box[0]) * policy['margin_each_side']
    dy = (box[3] - box[1]) * policy['margin_each_side']
    crop = [max(0, math.floor(box[0] - dx)), max(0, math.floor(box[1] - dy)),
            min(width, math.ceil(box[2] + dx)), min(height, math.ceil(box[3] + dy))]
    fraction = area(crop) / (width * height)
    if fraction >= policy['maximum_crop_area_fraction']:
        result['reason'] = 'nearly_full_frame'
        return result
    result.update(crop_box=crop, status='cropped', reason='confident_padded_detection',
                  area_fraction=fraction,
                  linear_magnification=max(width, height) / max(crop[2] - crop[0], crop[3] - crop[1]))
    return result


def crop_preview(original, decision, size=448):
    with original.crop(tuple(decision['crop_box'])) as cropped:
        return make_preview(cropped, size)[0]


def validated_crop(size, detections, coverage, policy=POLICY):
    """A second detector must confirm the target and can enlarge its extent."""
    first=choose_crop(size,detections,policy)
    if first['status']=='fallback':
        return first
    box=first['selected_box'];matched=[]
    credible=[item for item in coverage if item['score']>=policy['coverage_score_threshold'] and area(item['box'])>0]
    if credible:
        largest=max(credible,key=lambda x:area(x['box']))['box']
        intersection=area([max(box[0],largest[0]),max(box[1],largest[1]),min(box[2],largest[2]),min(box[3],largest[3])])
        if intersection/min(area(box),area(largest))<policy['coverage_min_intersection_over_smaller']:
            first.update(crop_box=[0,0,*size],status='fallback',reason='coverage_target_disagreement',area_fraction=1.,linear_magnification=1.)
            return first
    for item in credible:
        b=item['box']
        if item['score']<policy['coverage_score_threshold'] or area(b)<=0:
            continue
        intersection=area([max(box[0],b[0]),max(box[1],b[1]),min(box[2],b[2]),min(box[3],b[3])])
        if intersection/min(area(box),area(b))>=policy['coverage_min_intersection_over_smaller']:
            matched.append(item)
    if not matched:
        first.update(crop_box=[0,0,*size],status='fallback',reason='coverage_unconfirmed',area_fraction=1.,linear_magnification=1.)
        return first
    second=max(matched,key=lambda x:area(x['box']))
    b=second['box'];union=[min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3])]
    result=choose_crop(size,[{'box':union,'score':max(first['selected_score'],second['score'])}],policy)
    result.update(primary_box=box,coverage_box=b,coverage_score=second['score'])
    return result


class CoverageDetector:
    def __init__(self,weights_directory):
        torch.set_num_threads(2)
        self.weights_file=Path(weights_directory)/'checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
        self.model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=480,max_size=800)
        self.model.load_state_dict(torch.load(self.weights_file,map_location='cpu',weights_only=True))
        self.model.eval();self.sha256=file_hash(self.weights_file)

    @torch.inference_mode()
    def detect(self,original):
        with original.copy() as image:
            image.thumbnail((800,800),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.array(image,copy=True).transpose(2,0,1).copy()).float()/255
            output=self.model([x])[0]
            sx,sy=original.width/image.width,original.height/image.height
            return [{'box':[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],'score':s}
                    for b,l,s in zip(output['boxes'].tolist(),output['labels'].tolist(),output['scores'].tolist())
                    if l==1 and s>=.3]


class CasualtyDetector:
    def __init__(self, weights_directory: Path, device='cpu', download=True):
        self.device = torch.device(device)
        self.weights_directory = weights_directory.resolve()
        self.weights_file = self.weights_directory / 'checkpoints/fcos_resnet50_fpn_coco-99b0c9b7.pth'
        if download:
            torch.hub.set_dir(str(self.weights_directory))
            self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1, progress=False)
        else:
            self.model = fcos_resnet50_fpn(weights=None, weights_backbone=None)
            self.model.load_state_dict(torch.load(self.weights_file, map_location='cpu', weights_only=True))
        self.model.to(self.device).eval()
        self.sha256 = file_hash(self.weights_file)

    @torch.inference_mode()
    def detect(self, original):
        with original.copy() as image:
            image.thumbnail((POLICY['detector_max_side'],) * 2, Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True).transpose(2, 0, 1).copy())
            output = self.model([pixels.to(self.device).float() / 255])[0]
            # CPU conversion is batched; avoid thousands of synchronizations.
            boxes = output['boxes'].cpu().tolist()
            labels = output['labels'].cpu().tolist()
            scores = output['scores'].cpu().tolist()
            sx, sy = original.width / image.width, original.height / image.height
            return [{'box': [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], 'score': score}
                    for b, label, score in zip(boxes, labels, scores) if label == 1 and score >= .3]
