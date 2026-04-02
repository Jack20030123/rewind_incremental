import numpy as np
import torch


def _l2_normalize_np(array, eps=1e-8):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=eps, a_max=None)
    return array / norms


def compute_dino_goal_progress(sequence_embeddings, goal_source_embeddings=None, goal_k=3, eps=1e-8):
    sequence_embeddings = np.asarray(sequence_embeddings, dtype=np.float32)
    if sequence_embeddings.ndim != 2:
        raise ValueError("sequence_embeddings must have shape [T, D]")

    if goal_source_embeddings is None:
        goal_source_embeddings = sequence_embeddings
    goal_source_embeddings = np.asarray(goal_source_embeddings, dtype=np.float32)
    if goal_source_embeddings.ndim != 2:
        raise ValueError("goal_source_embeddings must have shape [T, D]")

    normalized_sequence = _l2_normalize_np(sequence_embeddings, eps=eps)
    normalized_goal_source = _l2_normalize_np(goal_source_embeddings, eps=eps)

    goal_k = max(1, min(int(goal_k), normalized_goal_source.shape[0]))
    goal_embedding = normalized_goal_source[-goal_k:].mean(axis=0)
    goal_embedding = _l2_normalize_np(goal_embedding[None], eps=eps)[0]

    goal_distances = np.linalg.norm(normalized_sequence - goal_embedding[None], axis=-1)
    min_distance = float(goal_distances.min())
    max_distance = float(goal_distances.max())

    if max_distance - min_distance < eps:
        if max_distance < eps:
            progress = np.ones_like(goal_distances, dtype=np.float32)
        else:
            progress = np.zeros_like(goal_distances, dtype=np.float32)
    else:
        progress = 1.0 - ((goal_distances - min_distance) / (max_distance - min_distance))

    return progress.astype(np.float32), goal_distances.astype(np.float32), goal_embedding.astype(np.float32)


def compute_directional_penalty(reward_predictions, goal_distances, tau_away=0.0, margin=0.0):
    if reward_predictions.ndim == 3 and reward_predictions.shape[-1] == 1:
        reward_predictions = reward_predictions.squeeze(-1)
    if reward_predictions.ndim != 2:
        raise ValueError("reward_predictions must have shape [B, T] or [B, T, 1]")
    if goal_distances.ndim != 2:
        raise ValueError("goal_distances must have shape [B, T]")

    reward_deltas = reward_predictions[:, 1:] - reward_predictions[:, :-1]
    distance_deltas = goal_distances[:, 1:] - goal_distances[:, :-1]
    away_mask = distance_deltas > tau_away

    if away_mask.numel() == 0:
        zero = reward_predictions.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    away_count = away_mask.sum()
    away_rate = away_mask.float().mean()
    if away_count.item() == 0:
        zero = reward_predictions.sum() * 0.0
        return zero, zero.detach(), away_rate.detach()

    penalty = torch.relu(reward_deltas - margin) * away_mask.float()
    loss = penalty.sum() / away_count.clamp(min=1)
    violation_rate = ((reward_deltas > margin) & away_mask).float().sum() / away_count
    return loss, violation_rate.detach(), away_rate.detach()
