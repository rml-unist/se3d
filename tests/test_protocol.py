import argparse
import unittest
import warnings

from se3d.protocol import add_labels_argument, label_name
from tools.check_protocol import check


class AnnotationProtocolTests(unittest.TestCase):
    def test_historical_cli_names_resolve_to_public_directories(self):
        parser = argparse.ArgumentParser()
        add_labels_argument(parser)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            self.assertEqual(parser.parse_args(['--labels', 'corrected']).labels, 'label')
            self.assertEqual(parser.parse_args(['--labels', 'original']).labels, 'label_original')
        self.assertEqual(len(emitted), 2)
        self.assertEqual(parser.parse_args([]).labels, 'label')

    def test_internal_paths_are_not_silently_accepted(self):
        for value in ('label_5', 'label_3', '../label', '/tmp/label'):
            with self.assertRaises(argparse.ArgumentTypeError):
                label_name(value)

    def test_published_split_and_anchor_invariants(self):
        self.assertTrue(check()['passed'])


if __name__ == '__main__':
    unittest.main()
