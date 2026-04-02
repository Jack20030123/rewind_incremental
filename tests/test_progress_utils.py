import unittest

import numpy as np
import torch

from utils.progress_utils import compute_dino_goal_progress, compute_directional_penalty


class ProgressUtilsTest(unittest.TestCase):
    def test_forward_progress_increases_toward_goal(self):
        sequence = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.4, 0.9, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        progress, distances, _ = compute_dino_goal_progress(sequence, goal_k=2)

        self.assertGreater(progress[-1], progress[0])
        self.assertLess(distances[-1], distances[0])
        self.assertAlmostEqual(float(progress.max()), 1.0, places=6)

    def test_rewind_suffix_decreases_progress(self):
        forward = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.4, 0.9, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        rewind_sequence = np.concatenate([forward, forward[::-1][1:3]], axis=0)
        progress, _, _ = compute_dino_goal_progress(
            rewind_sequence,
            goal_source_embeddings=forward,
            goal_k=2,
        )

        self.assertGreater(progress[3], progress[4])
        self.assertGreater(progress[4], progress[5])

    def test_freeze_duplicate_frames_keep_progress_flat(self):
        forward = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.4, 0.9, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        frozen_sequence = np.array(
            [
                forward[0],
                forward[1],
                forward[1],
                forward[2],
                forward[3],
            ],
            dtype=np.float32,
        )
        progress, _, _ = compute_dino_goal_progress(
            frozen_sequence,
            goal_source_embeddings=forward,
            goal_k=2,
        )

        self.assertAlmostEqual(float(progress[1]), float(progress[2]), places=6)

    def test_directional_penalty_only_applies_on_away_reward_increase(self):
        rewards = torch.tensor([[0.1, 0.3, 0.2]], dtype=torch.float32)
        distances = torch.tensor([[0.2, 0.5, 0.4]], dtype=torch.float32)

        loss, violation_rate, away_rate = compute_directional_penalty(
            rewards,
            distances,
            tau_away=0.05,
            margin=0.0,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(violation_rate.item(), 1.0, places=6)
        self.assertAlmostEqual(away_rate.item(), 0.5, places=6)

    def test_directional_penalty_allows_nonincreasing_reward_on_away_steps(self):
        rewards = torch.tensor([[0.4, 0.2, 0.1]], dtype=torch.float32)
        distances = torch.tensor([[0.2, 0.5, 0.7]], dtype=torch.float32)

        loss, violation_rate, _ = compute_directional_penalty(
            rewards,
            distances,
            tau_away=0.05,
            margin=0.0,
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(violation_rate.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
