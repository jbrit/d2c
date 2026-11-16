import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_data import CLASSES,LIMBS,encode_labels,grouped_split,majority_predictions,metrics,safe_run_directory


def fixture():
    result=[]
    for group in range(12):
        for view in range(2):
            result.append({'source_record':len(result)+1,'casualty_id':f'casualty_{group:02}',
                           'image_sha256':f'unique-{group}-{view}',
                           'arm_left':'amputation' if group==0 else ['no_injury','wound','not_testable'][group%3],
                           'arm_right':CLASSES[group%4],'leg_left':'no_injury','leg_right':'no_injury'})
    return result


class GroupedSplitTests(unittest.TestCase):
    def test_split_is_repeatable_disjoint_and_covers_supported_classes(self):
        records=fixture()
        assignment,summary=grouped_split(records,seed=17,fraction=.33)
        self.assertEqual((assignment,summary),grouped_split(records,seed=17,fraction=.33))
        groups={name:{r['casualty_id'] for r in records if assignment[r['source_record']]==name}
                for name in ['train','validation']}
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertIn('casualty_00',groups['train'])
        self.assertEqual(summary['validation']['labels']['arm_left']['amputation'],0)
        for limb in LIMBS:
            for label in CLASSES:
                support={r['casualty_id'] for r in records if r[limb]==label}
                if len(support)>=2:
                    for split in ['train','validation']:
                        self.assertGreater(summary[split]['labels'][limb][label],0)

    def test_duplicate_image_content_cannot_cross_split(self):
        records=fixture()
        assignment,_=grouped_split(records,seed=17,fraction=.33)
        left=next(r for r in records if assignment[r['source_record']]=='train')
        right=next(r for r in records if assignment[r['source_record']]=='validation')
        right['image_sha256']=left['image_sha256']
        with self.assertRaisesRegex(ValueError,'duplicate image bytes'):
            grouped_split(records,seed=17,fraction=.33)

    def test_insufficient_groups_fail_instead_of_leaking(self):
        with self.assertRaises(ValueError):grouped_split(fixture()[:2])

    def test_run_cannot_contain_or_be_inside_source(self):
        root=Path('/tmp/example/train')
        for run in [root,root/'cleaned/run',root.parent]:
            with self.assertRaises(ValueError):safe_run_directory(run,root)
        safe_run_directory(root.parent/'runs/one',root)


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_unsupported_classes(self):
        truth=[[0,1,2,3],[1,1,0,3]]
        predicted=[[0,0,2,3],[0,1,0,0]]
        result=metrics(truth,predicted,['one','two'])
        self.assertEqual(result['label_accuracy'],5/8)
        self.assertEqual(result['exact_match_accuracy'],0)
        self.assertAlmostEqual(result['mean_limb_macro_f1'],2/3)
        self.assertEqual(result['by_limb']['arm_left']['confusion_matrix_true_rows_predicted_columns'],
                         [[1,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.assertIsNone(result['by_limb']['arm_left']['classes']['amputation']['f1'])
        self.assertEqual(result['by_limb']['arm_right']['classes']['no_injury']['predicted'],1)
        self.assertEqual(result['casualty_mean_label_accuracy'],5/8)

    def test_majority_is_fitted_on_training_labels_only(self):
        train=[dict.fromkeys(LIMBS,'wound'),dict.fromkeys(LIMBS,'wound'),dict.fromkeys(LIMBS,'no_injury')]
        predicted=majority_predictions(train,5)
        self.assertEqual(predicted,[[1,1,1,1]]*5)
        predicted[0][0]=3
        self.assertEqual(predicted[1][0],1)

    def test_perfect_supported_classes_do_not_certify_absent_class(self):
        result=metrics([[0,0,0,0]],[[0,0,0,0]])
        self.assertEqual(result['mean_limb_macro_f1'],1)
        self.assertEqual(result['mean_limb_macro_f1_fixed4'],.25)
        self.assertIsNone(result['by_limb']['arm_left']['classes']['wound']['recall'])

    def test_malformed_labels_are_rejected(self):
        for truth,pred in [([],[]),([[0,0,0,0]],[]),([[0,0,0,4]],[[0,0,0,0]])]:
            with self.assertRaises(ValueError):metrics(truth,pred)


class TrainingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from baseline_model import LimbModel,loss_fn,seed_everything
        except ImportError as error:
            raise unittest.SkipTest(f'Use the training environment: {error}')
        cls.torch=torch;cls.Model=LimbModel;cls.loss_fn=staticmethod(loss_fn)
        seed_everything(42)

    def test_heads_update_while_backbone_and_batchnorm_remain_fixed(self):
        torch=self.torch;model=self.Model()
        model.configure_stage('frozen')
        before={k:v.clone() for k,v in model.backbone.state_dict().items()}
        heads=copy.deepcopy(model.heads.state_dict())
        optimizer=torch.optim.SGD(model.heads.parameters(),lr=.1)
        output=model(torch.randn(2,3,64,64))
        self.assertEqual(output.shape,(2,4,4))
        self.loss_fn(output,torch.tensor([[0,1,2,3],[3,2,1,0]])).backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,model.backbone.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(any(not torch.equal(v,model.heads.state_dict()[k]) for k,v in heads.items()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_finetuning_updates_final_block_but_not_early_layers_or_statistics(self):
        torch=self.torch;model=self.Model();model.configure_stage('finetuned')
        early=model.backbone.conv1.weight.detach().clone()
        late=model.backbone.layer4[0].conv1.weight.detach().clone()
        buffers={k:v.clone() for k,v in model.backbone.named_buffers()}
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        logits=model(torch.randn(2,3,64,64))
        self.loss_fn(logits,torch.tensor([[0,1,2,3],[3,2,1,0]])).backward();optimizer.step()
        self.assertTrue(torch.equal(early,model.backbone.conv1.weight))
        self.assertFalse(torch.equal(late,model.backbone.layer4[0].conv1.weight))
        self.assertTrue(all(torch.equal(v,dict(model.backbone.named_buffers())[k]) for k,v in buffers.items()))


if __name__=='__main__':unittest.main()
