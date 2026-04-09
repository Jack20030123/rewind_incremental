import unittest

import numpy as np

from utils.progress_utils import compute_frame_diff_progress


class OpticalFlowProgressTest(unittest.TestCase):
    def test_frame_diff_progress_is_cumulative_and_normalized(self):
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        frames[1] = 32
        frames[2] = 96
        frames[3] = 128

        progress, motion_signal = compute_frame_diff_progress(frames)

        self.assertEqual(progress.shape[0], frames.shape[0])
        self.assertEqual(motion_signal.shape[0], frames.shape[0])
        self.assertAlmostEqual(float(progress[0]), 0.0, places=6)
        self.assertAlmostEqual(float(progress[-1]), 1.0, places=6)
        self.assertTrue(np.all(progress[1:] >= progress[:-1]))

    def test_static_frames_produce_flat_progress(self):
        frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        progress, motion_signal = compute_frame_diff_progress(frames)

        self.assertTrue(np.allclose(progress, 0.0))
        self.assertTrue(np.allclose(motion_signal, 0.0))

    def test_rewind_labels_decrease_when_sequence_is_reversed(self):
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        frames[1] = 64
        frames[2] = 96
        frames[3] = 160

        progress, _ = compute_frame_diff_progress(frames)
        rewind_suffix = progress[::-1][1:3]
        rewind_progress = np.concatenate([progress, rewind_suffix], axis=0)

        self.assertGreater(rewind_progress[3], rewind_progress[4])
        self.assertGreater(rewind_progress[4], rewind_progress[5])


if __name__ == "__main__":
    unittest.main()
