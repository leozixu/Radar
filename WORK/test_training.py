"""Regression checks for bounded training selection and offline labels."""

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from train_tracker import LabelLookup, label_samples, main


class TrainingBoundariesTest(unittest.TestCase):
    def test_tuning_and_user_holdout_rejected_before_any_read(self):
        for out_of_scope in (71, 85, 86, 100, 101, 200):
            with self.subTest(packet=out_of_scope), patch("train_tracker.read_packet") as read:
                with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                    main(["--phase", "calibrate", "--packets", "1", str(out_of_scope)])
                read.assert_not_called()

    def test_association_identity_needs_history_support(self):
        pids = np.arange(24).reshape(6, 4)
        pids[5, 1:] = -1
        prior = np.array([
            [9, 4, 4, 4], [4, 4, 9, 9], [0, 0, 0, 0], [4, 4, 4, 4], [4, 4, 4, 4], [4, -1, -1, -1],
        ])
        current = np.array([4, 4, 0, -1, 9, 4])
        result = label_samples("association", pids, prior, current)
        np.testing.assert_array_equal(result, [0, -1, 2, -1, 1, 0])

    def test_unknown_segment_is_omitted_and_padding_is_not_an_observation(self):
        pids = np.array([[10, 11, -1], [12, 14, -1], [10, -1, -1], [10, 11, -1], [12, 14, -1]])
        lookup = LabelLookup(np.array([10, 11, 12, 14, 100]), np.array([7, 7, 7, -1, 0]))
        classes = label_samples("birth", pids, lookup(pids), np.array([7, 7, 7, 12, 0]))
        np.testing.assert_array_equal(classes, [0, -1, -1, 1, -1])


if __name__ == "__main__":
    unittest.main()
