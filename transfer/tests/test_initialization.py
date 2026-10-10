import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from transfer import common, protocol_utils as protocol, runtime, test as inference, train


class TinyDetector(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = torch.nn.Module()
        self.network.feature = torch.nn.Linear(3, 4)
        for name in ('bbox_cls', 'bbox_reg', 'bbox_centerness'):
            setattr(self.network, name, torch.nn.Linear(4, classes))


class InitializationTests(unittest.TestCase):
    def test_all_shared_tensors_copy_and_seeded_heads_stay_identical(self):
        torch.manual_seed(91)
        source = TinyDetector(7)
        torch.manual_seed(17)
        target = TinyDetector(3)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        copied = common.initialize_from_se3d(target, 'dsgn_event', source_state=source.state_dict())
        reset = common.RESET_PREFIXES['dsgn_event']
        self.assertEqual(set(copied), {name for name in before if not name.startswith(reset)})
        for name, value in target.state_dict().items():
            expected = before[name] if name.startswith(reset) else source.state_dict()[name]
            self.assertTrue(torch.equal(value, expected), name)
        incomplete = dict(source.state_dict())
        del incomplete['network.feature.weight']
        with self.assertRaisesRegex(ValueError, 'lacks shared'):
            common.initialize_from_se3d(target, 'dsgn_event', source_state=incomplete)

    def test_source_seed_step_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pth'
            torch.save(dict(model=TinyDetector(7).state_dict(), step=214368, epoch=8,
                            config=dict(model='dsgn_event', seed=20260909, labels='label')), path)
            args = SimpleNamespace(init='se3d', source=path, model='dsgn_event', source_sha256=protocol.sha256(path),
                                   source_step=214368, source_seed=20260909)
            _, identity = train.source_checkpoint(args)
            self.assertEqual((identity['source_seed'], identity['source_step']), (20260909, 214368))
            for field, changed in [('source_sha256', 'wrong'), ('source_seed', 20260910), ('source_step', 1071840)]:
                bad = copy.copy(args)
                setattr(bad, field, changed)
                with self.assertRaises(ValueError):
                    train.source_checkpoint(bad)

    def test_inference_uses_embedded_subset_anchors(self):
        anchors = json.loads(common.ANCHORS.read_text())
        anchors['training_anchor_dimensions']['Vehicle']['height'] = 9.25
        state = dict(schema=runtime.SCHEMA, anchors=anchors,
                     config=dict(anchor_payload_sha256=protocol.json_digest(anchors)))
        result = inference.checkpoint_anchors(state)
        cfg = common.configure(result)
        self.assertEqual(cfg.RPN3D.ANCHORS_HEIGHT[0], 9.25)
        with self.assertRaisesRegex(ValueError, 'Explicit anchors differ'):
            inference.checkpoint_anchors(state, common.ANCHORS)
        tampered = copy.deepcopy(state)
        tampered['anchors']['training_anchor_dimensions']['Vehicle']['height'] = 2.5
        with self.assertRaisesRegex(ValueError, 'Embedded anchors differ'):
            inference.checkpoint_anchors(tampered)
        common.configure()


if __name__ == '__main__':
    unittest.main()
