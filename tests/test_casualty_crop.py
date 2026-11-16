import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from casualty_crop import POLICY,choose_crop,validated_crop,crop_preview


class CropPolicyTests(unittest.TestCase):
    def test_small_person_fragment_at_frame_edge_keeps_full_image(self):
        result=choose_crop((1200,800),[{'box':[1080,100,1200,500],'score':.99}])
        self.assertEqual(result['reason'],'small_edge_detection')

    def test_second_detector_larger_separate_subject_rejects_background_person(self):
        primary=[{'box':[850,200,1050,500],'score':.8}]
        coverage=[{'box':[850,200,1050,500],'score':.99},{'box':[100,100,700,700],'score':.99}]
        result=validated_crop((1200,800),primary,coverage)
        self.assertEqual(result['reason'],'coverage_target_disagreement')

    def test_second_detector_expands_torso_box_to_cover_legs(self):
        primary=[{'box':[400,300,600,500],'score':.8}]
        coverage=[{'box':[350,250,900,550],'score':.99}]
        first=choose_crop((1200,800),primary)
        result=validated_crop((1200,800),primary,coverage)
        self.assertEqual(result['crop_box'],[185,160,1065,640])
        self.assertLessEqual(result['crop_box'][0],first['crop_box'][0])
        self.assertGreaterEqual(result['crop_box'][2],first['crop_box'][2])

    def test_second_detector_must_confirm_same_subject(self):
        primary=[{'box':[400,300,600,500],'score':.8}]
        for coverage,reason in [([], 'coverage_unconfirmed'),
                         ([{'box':[10,10,200,200],'score':.99}], 'coverage_target_disagreement'),
                         ([{'box':[400,300,600,500],'score':.7}], 'coverage_unconfirmed')]:
            result=validated_crop((1200,800),primary,coverage)
            self.assertEqual(result['reason'],reason)
            self.assertEqual(result['crop_box'],[0,0,1200,800])

    def test_margin_is_thirty_percent_of_box_on_each_side_in_native_coordinates(self):
        d=choose_crop((1000,800),[{'box':[300,200,700,600],'score':.9}])
        self.assertEqual(d['crop_box'],[180,80,820,720])
        self.assertEqual(d['status'],'cropped')
        self.assertAlmostEqual(d['linear_magnification'],1000/640)

    def test_edge_crop_is_clamped_and_encloses_entire_selected_box(self):
        d=choose_crop((1000,800),[{'box':[0,30,500,500],'score':.9}])
        self.assertEqual(d['crop_box'],[0,0,650,641])

    def test_missing_uncertain_invalid_or_tiny_detections_keep_whole_image(self):
        for boxes in [[],[{'box':[100,100,500,500],'score':.4}],
                      [{'box':[10,10,11,11],'score':.99}],
                      [{'box':[float('nan'),0,100,100],'score':.99}],
                      [{'box':[300,300,200,200],'score':.99}]]:
            d=choose_crop((1000,800),boxes)
            self.assertEqual(d['status'],'fallback')
            self.assertEqual(d['crop_box'],[0,0,1000,800])

    def test_ambiguous_separate_people_fall_back_but_nested_boxes_do_not(self):
        people=[{'box':[50,100,400,700],'score':.9},{'box':[550,100,900,700],'score':.95}]
        self.assertEqual(choose_crop((1000,800),people)['reason'],'ambiguous_multiple_people')
        nested=[{'box':[300,200,700,600],'score':.9},{'box':[320,220,680,580],'score':.95}]
        self.assertEqual(choose_crop((1000,800),nested)['status'],'cropped')

    def test_selects_largest_confident_box_without_mutating_detections(self):
        people=[{'box':[400,300,500,500],'score':.99},{'box':[200,100,700,600],'score':.8}]
        before=copy.deepcopy(people)
        d=choose_crop((1000,800),people)
        self.assertEqual(d['selected_box'],[200,100,700,600])
        self.assertEqual(people,before)

    def test_full_frame_box_and_fallback_preserve_original_preview_pixels(self):
        d=choose_crop((1000,800),[{'box':[0,0,1000,800],'score':.9}])
        self.assertEqual(d['reason'],'nearly_full_frame')
        original=Image.new('RGB',(1000,800),(10,40,80))
        before=np.array(original)
        from audit_images import make_preview
        expected,_=make_preview(original,448)
        result=crop_preview(original,d)
        self.assertTrue(np.array_equal(np.array(result),np.array(expected)))
        self.assertTrue(np.array_equal(np.array(original),before))
        self.assertEqual(result.size,(448,448))


if __name__=='__main__':unittest.main()
