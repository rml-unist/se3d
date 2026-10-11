import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transfer import common, protocol_utils as protocol, train


class ProtocolTests(unittest.TestCase):
    def test_training_uses_complete_frozen_pool_and_anchors(self):
        args = train.arguments(['--model', 'se_cff', '--init', 'scratch', '--dsec-root', '/unused/dsec',
                                '--labels-root', '/unused/labels', '--output', '/unused/run'])
        audited = []

        def audit(rows, *unused):
            audited.append(rows)
            return dict(rows_sha256=protocol.json_digest(rows), files_sha256='test files')

        with patch.object(protocol, 'audit_input_files', audit):
            base, rows, anchors_path, anchors, inputs = train.prepare_inputs(args)
        self.assertEqual({split: len(values) for split, values in base['splits'].items()},
                         common.SPLIT_SIZES)
        self.assertEqual(rows, base['splits']['train'])
        self.assertEqual(audited, [rows, base['splits']['validation']])
        self.assertEqual(anchors_path, common.ANCHORS)
        self.assertEqual(anchors, json.loads(common.ANCHORS.read_text()))
        common.configure(anchors)
        dataset = common.dataset('train', args.dsec_root, args.labels_root, protocol_path=args.protocol)
        self.assertEqual(len(dataset), 3906)
        self.assertEqual(dataset.rows, rows)
        self.assertEqual(inputs['train']['rows_sha256'], protocol.json_digest(dataset.rows))

    def test_duplicate_frames_and_overlapping_chunks_are_rejected(self):
        base = protocol.load_protocol()
        duplicate = copy.deepcopy(base)
        duplicate['splits']['train'].append(duplicate['splits']['train'][0])
        overlap = copy.deepcopy(base)
        overlap['splits']['validation'][0]['chunk'] = overlap['splits']['train'][0]['chunk']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'protocol.json'
            for changed, message in [(duplicate, 'duplicate'), (overlap, 'multiple splits')]:
                with self.subTest(message=message):
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, message):
                        protocol.load_protocol(path)

    def test_input_audit_binds_annotations_calibration_and_disparity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dsec, labels = root / 'dsec', root / 'labels'
            row = dict(sequence='sequence', annotation_path='chunk/labels.pkl',
                       disparity_path='sequence/disparity/000000.png')
            frozen = dict(source_hashes={})
            paths = [(labels, row['annotation_path']),
                     (dsec, 'sequence/calibration/cam_to_cam.yaml'),
                     (dsec, 'sequence/calibration/cam_to_lidar.yaml'),
                     (dsec, row['disparity_path'])]
            for directory, relative in paths:
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
                frozen['source_hashes'][relative] = protocol.sha256(path)
            initial = protocol.audit_input_files([row], frozen, dsec, labels)
            self.assertEqual(initial['frames'], 1)
            self.assertEqual(len(initial['files']), 4)
            for directory, relative in paths[:3]:
                with self.subTest(relative=relative):
                    path = directory / relative
                    original = path.read_bytes()
                    path.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'differs from frozen protocol'):
                        protocol.audit_input_files([row], frozen, dsec, labels)
                    path.write_bytes(original)
            (dsec / row['disparity_path']).write_bytes(b'changed depth')
            changed = protocol.audit_input_files([row], frozen, dsec, labels)
            self.assertNotEqual(changed['files_sha256'], initial['files_sha256'])
            self.assertEqual(changed['rows_sha256'], initial['rows_sha256'])


if __name__ == '__main__':
    unittest.main()
