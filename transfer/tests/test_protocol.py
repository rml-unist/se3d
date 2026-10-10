import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from transfer import protocol_utils as protocol


class SubsetProtocolTests(unittest.TestCase):
    def test_frozen_nested_counts_and_heldout_exclusion(self):
        base = protocol.load_protocol()
        digest = protocol.sha256(protocol.PROTOCOL)
        prior = set()
        for fraction, chunks, frames in [(10, 13, 403), (25, 32, 992), (50, 63, 1953)]:
            subset = protocol.make_subset(base, digest, fraction)
            rows = protocol.validate_subset(subset, base, digest)
            self.assertEqual((subset['selected_chunks'], len(rows)), (chunks, frames))
            current = {protocol.frame_key(row) for row in rows}
            self.assertTrue(prior.issubset(current))
            prior = current
            heldout = {row['chunk'] for split in ['validation', 'test'] for row in base['splits'][split]}
            self.assertFalse(heldout.intersection(subset['chunks']))
            self.assertEqual(protocol.json_digest(subset),
                             protocol.json_digest(protocol.make_subset(base, digest, float(fraction))))

    def test_subset_edit_and_protocol_change_rejected(self):
        base = protocol.load_protocol()
        digest = protocol.sha256(protocol.PROTOCOL)
        subset = protocol.make_subset(base, digest, 10)
        tampered = copy.deepcopy(subset)
        tampered['train_indices'][0] = len(base['splits']['train']) - 1
        with self.assertRaisesRegex(ValueError, 'hash-prefix'):
            protocol.validate_subset(tampered, base, digest)
        with self.assertRaises(ValueError):
            protocol.validate_subset(subset, base, 'different protocol')

    def test_subset_anchors_read_only_selected_files_and_use_frozen_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dsec, labels = root / 'dsec', root / 'labels'
            rows = []
            for chunk in ('a', 'b', 'c', 'd'):
                for frame in range(3):
                    rows.append(dict(chunk=chunk, sequence='seq', frame=frame, timestamp_us=frame,
                                     annotation_path=chunk + '/labels.pkl', disparity_path='unused.png'))
            base = dict(splits=dict(train=rows, validation=[dict(chunk='val')], test=[dict(chunk='test')]),
                        source_hashes={})
            selected = protocol.make_subset(base, 'temporary', 25)['chunks'][0]
            annotation_path = labels / selected / 'labels.pkl'
            annotation_path.parent.mkdir(parents=True)
            annotations = []
            for frame, (height, width, length, y) in enumerate([(2, 2, 4, 1), (4, 6, 8, 3), (0, 0, 0, 0)]):
                boxes = np.array([[0, y, 10, length, width, height, 0]], dtype=np.float64) if height else np.empty((0, 7))
                annotations.append(dict(time_stamp=frame, image=dict(image_0_extrinsic=np.eye(4)),
                                        annos=dict(gt_boxes_lidar=boxes,
                                                   name=np.array(['Vehicle'] if height else [], dtype='<U16'))))
            annotation_path.write_bytes(pickle.dumps(annotations))
            base['source_hashes'][selected + '/labels.pkl'] = protocol.sha256(annotation_path)
            calibration = dsec / 'seq' / 'calibration'
            calibration.mkdir(parents=True)
            projection = np.eye(4)
            projection[3, 2] = 2
            camera = dict(extrinsics=dict(R_rect0=np.eye(3).tolist(), R_rect1=np.eye(3).tolist(), T_10=np.eye(4).tolist()),
                          intrinsics=dict(camRect0=dict(camera_matrix=[100, 100, 320, 240])),
                          disparity_to_depth=dict(cams_03=projection.tolist()))
            for name, value in [('cam_to_cam.yaml', camera), ('cam_to_lidar.yaml', dict(T_lidar_camRect1=np.eye(4).tolist()))]:
                path = calibration / name
                path.write_text(yaml.safe_dump(value))
                base['source_hashes']['seq/calibration/' + name] = protocol.sha256(path)
            digest = protocol.json_digest(base)
            subset = protocol.make_subset(base, digest, 25)
            # No files for the other three training chunks, validation or test exist.
            anchors = protocol.derive_subset_anchors(base, digest, subset, 'subset hash', dsec, labels)
            vehicle = anchors['training_anchor_dimensions']['Vehicle']
            self.assertEqual(vehicle, dict(height=3., width=4., length=6., center_y=2., annotations=2))
            fallback = json.loads(protocol.FALLBACK.read_text())['training_anchor_dimensions']
            for name in ('Pedestrian', 'Cyclist'):
                self.assertEqual(anchors['training_anchor_dimensions'][name], dict(fallback[name], annotations=0))
            self.assertEqual(anchors['provenance']['fallback_classes'], ['Pedestrian', 'Cyclist'])
            self.assertEqual(anchors['training_audit']['empty_frames'], 1)
            self.assertEqual(anchors['training_audit']['per_class']['Vehicle']['frames'], 2)
            audited = anchors['provenance']['selected_annotation_and_calibration_files']
            self.assertEqual(len(audited), 3)
            self.assertIn('labels/' + selected + '/labels.pkl', audited)
            annotation_path.write_bytes(b'changed after freeze')
            with self.assertRaisesRegex(ValueError, 'Annotation differs'):
                protocol.derive_subset_anchors(base, digest, subset, 'subset hash', dsec, labels)


if __name__ == '__main__':
    unittest.main()
