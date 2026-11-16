import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_data import CLASSES,LIMBS
from error_review_data import pack_result,validate_saved
from review_errors import result_rows,summarize


class ErrorReviewTests(unittest.TestCase):
    def test_majority_has_no_invented_probability_and_counts_each_limb(self):
        truth=dict(zip(LIMBS,CLASSES))
        predicted=dict.fromkeys(LIMBS,'no_injury')
        packed=pack_result(truth,predicted)
        self.assertEqual(packed['correct_limbs'],1)
        self.assertNotIn('probabilities',packed)
        record={'truth':truth,'models':{'majority':packed}}
        rows=result_rows(record,[{'key':'majority','name':'Majority'}])
        self.assertEqual(sum(r['is_error'] for r in rows),3)
        self.assertTrue(all(r['predicted_score'] is None for r in rows))

    def test_class_scores_distinguish_predicted_score_from_supplied_class_score(self):
        truth=dict.fromkeys(LIMBS,'wound');pred=dict.fromkeys(LIMBS,'no_injury')
        probabilities={l:dict(zip(CLASSES,[.8,.1,.05,.05])) for l in LIMBS}
        result=pack_result(truth,pred,probabilities)
        rows=result_rows({'truth':truth,'models':{'model':result}},[{'key':'model','name':'Model'}])
        self.assertTrue(all(r['predicted_score']==.8 and r['supplied_label_score']==.1 for r in rows))
        self.assertTrue(all(r['p_wound']==.1 for r in rows))

    def test_inconsistent_saved_correctness_is_rejected(self):
        truth=dict.fromkeys(LIMBS,'no_injury')
        with self.assertRaisesRegex(ValueError,'correctness'):
            result_rows({'truth':truth,'models':{'x':{'prediction':truth,'correct_limbs':0}}},[{'key':'x','name':'X'}])

    def test_summary_separates_training_validation_and_deduplicates_image_errors(self):
        records=[{'split':'validation','wrong_models':4,'learned_wrong_models':3,'annotation_issues':[],
                  'wrong_model_limbs':9},
                 {'split':'validation','wrong_models':0,'learned_wrong_models':0,'annotation_issues':[{}],
                  'wrong_model_limbs':0},
                 {'split':'train','wrong_models':1,'learned_wrong_models':0,'annotation_issues':[],
                  'wrong_model_limbs':2}]
        summary=summarize(records,[{}]*9)
        self.assertEqual(summary['validation']['any_model_error_images'],1)
        self.assertEqual(summary['train']['any_model_error_images'],1)
        self.assertEqual(summary['validation']['wrong_model_limb_decisions'],9)
        self.assertEqual(summary['validation']['annotation_flagged_images'],1)
        self.assertEqual(summary['limb_decisions'],3*9*4)

    def test_validation_pairing_rejects_missing_or_misassigned_source_records(self):
        record={'source_record':1,'filepath':'image.jpg','casualty_id':'case',**dict.fromkeys(LIMBS,'no_injury')}
        with self.assertRaisesRegex(ValueError,'cover'):
            validate_saved([record],[],{})
        row={'source_record':1,'filepath':'other.jpg','casualty_id':'case',
             'truth':dict.fromkeys(LIMBS,'no_injury'),'prediction':dict.fromkeys(LIMBS,'no_injury'),'correct_limbs':4}
        with self.assertRaisesRegex(ValueError,'lineage'):
            validate_saved([record],[row],{})


if __name__=='__main__':unittest.main()
