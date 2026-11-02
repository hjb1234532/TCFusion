"""Self-contained CPU checks; no datasets or published weights are needed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import pair_paths
from losses import FidelityLoss
from runtime import build_model, fusion_only, setup


class ReleaseTests(unittest.TestCase):
    def test_original_forward_and_gradients(self):
        setup(1234, 2)
        model = build_model().train()
        ir, vis = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
        full = model(ir, vis, return_tcfd_info=True)
        self.assertEqual(len(full), 7)
        for scale in full[-1].values():
            self.assertEqual(len(scale['u_steps_ir']), 3)
            self.assertEqual(len(scale['u_steps_vis']), 3)
        expected = full[5]
        actual = fusion_only(model, ir, vis)
        self.assertTrue(torch.equal(expected, actual))
        criterion = FidelityLoss()
        criterion(expected, ir, vis).backward()
        expected_gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        criterion(actual, ir, vis).backward()
        actual_gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        self.assertEqual(expected_gradients.keys(), actual_gradients.keys())
        for name in expected_gradients:
            torch.testing.assert_close(expected_gradients[name], actual_gradients[name], rtol=0, atol=0)

    def test_train_resume_and_native_size_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for kind in ['ir', 'vis']:
                (root / kind).mkdir()
            rng = np.random.RandomState(17)
            for i in range(2):
                for kind in ['ir', 'vis']:
                    Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / kind / ('%d.png' % i))
            def run(*args):
                result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)
            common = ['train.py', '--ir-dir', root / 'ir', '--vis-dir', root / 'vis', '--epochs', 2,
                      '--batch-size', 2, '--workers', 0, '--device', 'cpu', '--threads', 2]
            run(*common, '--output', root / 'full')
            run(*common, '--output', root / 'resumed', '--stop-after-epoch', 1)
            checkpoint = root / 'resumed/model_latest.pth'
            run(*common, '--output', root / 'resumed', '--resume', checkpoint)
            a = torch.load(root / 'full/model_latest.pth', map_location='cpu')
            b = torch.load(checkpoint, map_location='cpu')
            self.assertEqual(b['epoch'], 2)
            for name in a['state_dict']:
                self.assertTrue(torch.equal(a['state_dict'][name], b['state_dict'][name]), name)
            self.assertEqual(a['summary']['loss'], b['summary']['loss'])
            for kind in ['ir', 'vis']:
                (root / ('test_' + kind)).mkdir()
                Image.fromarray(rng.randint(0, 256, (35, 37, 3), dtype=np.uint8)).save(root / ('test_' + kind) / 'odd.png')
            run('infer.py', '--checkpoint', checkpoint, '--ir-dir', root / 'test_ir',
                '--vis-dir', root / 'test_vis', '--output', root / 'fused', '--device', 'cpu')
            with Image.open(root / 'fused/odd.png') as image:
                self.assertEqual(image.size, (37, 35))
                self.assertEqual(image.mode, 'RGB')
            meta = json.loads((root / 'fused/inference.json').read_text())
            self.assertEqual(meta['count'], 1)
            self.assertEqual(meta['images'][0]['pad_bottom_right'], [1, 3])
            # Mismatched names and ambiguous duplicate stems must never silently pair.
            (root / 'test_vis/odd.png').rename(root / 'test_vis/other.png')
            with self.assertRaises(ValueError):
                pair_paths(root / 'test_ir', root / 'test_vis')
            Image.new('RGB', (12, 12)).save(root / 'test_ir/odd.jpg')
            with self.assertRaises(ValueError):
                pair_paths(root / 'test_ir', root / 'test_vis')


if __name__ == '__main__':
    unittest.main()
