import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from baseline_data import file_hash
    from baseline_model import LimbModel, seed_everything
    from clean_dataset import CLASSES, LIMBS
    from per_limb import limb_score
    from per_limb_model import PerLimbPipeline, SingleLimbModel, load_pipeline, load_single, save_single
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Use the training environment')
class IndependentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)

    def test_initial_predictions_match_corresponding_shared_head_without_sharing_storage(self):
        shared = LimbModel().eval()
        single = SingleLimbModel.from_shared(shared, 2).eval()
        pixels = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(single(pixels), shared(pixels)[:, 2]))
        self.assertNotEqual(single.backbone.conv1.weight.data_ptr(), shared.backbone.conv1.weight.data_ptr())
        self.assertNotEqual(single.head.weight.data_ptr(), shared.heads[2].weight.data_ptr())

    def test_training_one_limb_changes_its_final_block_only_and_leaves_other_model_untouched(self):
        shared = LimbModel()
        one = SingleLimbModel.from_shared(shared, 0)
        other = SingleLimbModel.from_shared(shared, 1)
        untouched = {k: v.clone() for k, v in other.state_dict().items()}
        early = one.backbone.conv1.weight.detach().clone()
        late = one.backbone.layer4[0].conv1.weight.detach().clone()
        buffers = {k: v.clone() for k, v in one.backbone.named_buffers()}
        one.configure_stage('finetuned')
        optimizer = torch.optim.SGD([p for p in one.parameters() if p.requires_grad], lr=.1)
        output = one(torch.randn(2, 3, 64, 64))
        self.assertEqual(output.shape, (2, 4))
        F.cross_entropy(output, torch.tensor([1, 2])).backward(); optimizer.step()
        self.assertTrue(torch.equal(early, one.backbone.conv1.weight))
        self.assertFalse(torch.equal(late, one.backbone.layer4[0].conv1.weight))
        self.assertTrue(all(torch.equal(v, dict(one.backbone.named_buffers())[k]) for k, v in buffers.items()))
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in untouched.items()))

    def test_head_training_keeps_whole_backbone_fixed(self):
        model = SingleLimbModel(); model.configure_stage('frozen')
        before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        optimizer = torch.optim.SGD(model.head.parameters(), lr=.1)
        F.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 3])).backward(); optimizer.step()
        self.assertTrue(all(torch.equal(v, model.backbone.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_combination_routes_each_models_scores_to_its_anatomical_limb_without_voting(self):
        class Constant(nn.Module):
            def __init__(self, index):
                super().__init__(); self.index = index
            def forward(self, x):
                return F.one_hot(torch.full((len(x),), self.index), 4).float() * 10
        model = PerLimbPipeline([Constant(i) for i in range(4)])
        output = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(output.shape, (2, 4, 4))
        self.assertEqual(output.argmax(-1).tolist(), [[0, 1, 2, 3]] * 2)
        self.assertTrue(torch.allclose(output.softmax(-1).sum(-1), torch.ones(2, 4)))

    def test_limb_selection_ignores_other_targets_and_unsupported_class(self):
        records = [dict(zip(LIMBS, ['no_injury', 'wound', 'amputation', 'not_testable'])),
                   dict(zip(LIMBS, ['wound', 'amputation', 'not_testable', 'no_injury']))]
        logits = torch.tensor([[10., 0, 0, 0], [0., 10, 0, 0]])
        self.assertEqual(limb_score(records, logits, 0), 1.)
        records[0]['leg_right'] = 'amputation'
        self.assertEqual(limb_score(records, logits, 0), 1.)
        self.assertAlmostEqual(limb_score(records, torch.zeros(2, 4), 0), 1/3)

    def test_checkpoint_rejects_wrong_limb_or_input_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'one.pt'
            model = SingleLimbModel()
            save_single(path, model, LIMBS[0], 'finetuned', 2, .5, 'split')
            loaded, _ = load_single(path, LIMBS[0], 'split')
            self.assertTrue(torch.equal(loaded.head.weight, model.head.weight))
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_single(path, LIMBS[1], 'split')
            with self.assertRaisesRegex(ValueError, 'split'):
                load_single(path, LIMBS[0], 'another')

    def test_bundle_rejects_reordered_classes_and_tampered_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'pipeline.json'
            info = {'architecture': 'four_independent_resnet18', 'limbs': list(LIMBS),
                    'classes': list(reversed(CLASSES)), 'checkpoints': [{}] * 4}
            path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'order'):
                load_pipeline(path)
            (root / 'one.pt').write_bytes(b'tampered')
            info.update({'classes': list(CLASSES), 'input_fingerprint': 'split', 'stage': 'finetuned',
                         'checkpoints': [{'limb': limb, 'file': 'one.pt', 'sha256': 'invalid'} for limb in LIMBS]})
            path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_pipeline(path)


if __name__ == '__main__':
    unittest.main()
