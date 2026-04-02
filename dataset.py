import h5py
import random
import torch as th
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.progress_utils import compute_dino_goal_progress





class ReWiNDVideoDataset(Dataset):

    def __init__(self, args, h5_file, sample_neg=False):
        # h5_file = h5py.File(h5_file, "r")
        self.h5_file = h5_file
        self.args = args
        self.keys = list(self.h5_file.keys())
        self.sample_neg = sample_neg

    def _sample_trajectory_name(self, data_group):
        traj_lists = list(data_group.keys())
        traj_lists = [traj for traj in traj_lists if "lang" not in traj]
        return random.choice(traj_lists)

    def _compute_progress_target(self, video_frames, linear_progress, goal_source_frames):
        if self.args.progress_target_type == "linear":
            progress = np.asarray(linear_progress, dtype=np.float32)
            goal_distance = np.zeros_like(progress, dtype=np.float32)
        elif self.args.progress_target_type == "dino_goal_distance":
            progress, goal_distance, _ = compute_dino_goal_progress(
                sequence_embeddings=video_frames,
                goal_source_embeddings=goal_source_frames,
                goal_k=self.args.goal_k,
            )
        else:
            raise ValueError(f"Unsupported progress target type: {self.args.progress_target_type}")
        return progress, goal_distance


    def sample_text_feature(self, data_group):

        lang_embedding = np.array(data_group["minilm_lang_embedding"])
        # lang_embedding = np.expand_dims(lang_embedding, axis=0) # extract lang_embedding from the group

        len_lang_embedding = lang_embedding.shape[0]
        if len_lang_embedding > 1:
            lang_embedding = lang_embedding[random.randint(0, len_lang_embedding-1)]

        lang_embedding = th.from_numpy(lang_embedding).float()
        return lang_embedding
    

    def sample_negative_text_feature(self, key):
        random_key = random.choice(self.keys)
        while random_key == key:
            random_key = random.choice(self.keys)
        data_group = self.h5_file[random_key]
        lang_embedding = np.array(data_group["minilm_lang_embedding"])
        # lang_embedding = np.expand_dims(lang_embedding, axis=0)
        len_lang_embedding = lang_embedding.shape[0]
        if len_lang_embedding > 1:
            lang_embedding = lang_embedding[random.randint(0, len_lang_embedding-1)]

        lang_embedding = th.from_numpy(lang_embedding).float()
        return lang_embedding


    def sample_video_feature(self, data_group):
        random_name = self._sample_trajectory_name(data_group)
        progress_dataset = np.asarray(data_group[random_name]) # all video data
        
        start_idx = random.randint(0, len(progress_dataset)-3)
        end_idx = random.randint(start_idx+3, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        full_length = len(full_frames)
        video_progress = np.arange(0, video_frames.shape[0], dtype=np.float32) + 1
        video_progress = video_progress / full_length
        video_progress, goal_distance = self._compute_progress_target(
            video_frames=video_frames,
            linear_progress=video_progress,
            goal_source_frames=video_frames,
        )

        video_frames = th.from_numpy(video_frames).float()

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)
            video_progress = np.expand_dims(video_progress, axis=1)
            video_progress = self.padding_video(video_progress, self.args.max_length).detach().cpu().numpy()
            video_progress = np.squeeze(video_progress, axis=1)
            goal_distance = np.expand_dims(goal_distance, axis=1)
            goal_distance = self.padding_video(goal_distance, self.args.max_length).detach().cpu().numpy()
            goal_distance = np.squeeze(goal_distance, axis=1)
        return video_frames, video_progress, np.ones(video_progress.shape[0]), goal_distance


    def sample_reverse_video_feature(self, data_group):
        random_name = self._sample_trajectory_name(data_group)

        progress_dataset = np.asarray(data_group[random_name]) # all video data

        start_idx = random.randint(0, len(progress_dataset)//2)

        end_idx = random.randint(len(progress_dataset)//2, len(progress_dataset)) # end_idx start from len(progress_dataset)//2

        while end_idx - start_idx < 3:
            start_idx = random.randint(0, len(progress_dataset)//2)
            end_idx = random.randint(len(progress_dataset)//2, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        progress_idx = np.arange(0, video_frames.shape[0], dtype=np.float32) + 1
        progress = progress_idx / len(full_frames)
        forward_frames = video_frames.copy()

        # rewind the video
        # reverse_frame = video_frames[::-1][1:]
        # reverse_progress = progress[::-1][1:]

        # random start rewind
        selected_end_point = random.randint(2, len(video_frames))
        reverse_frame = video_frames[::-1][1:selected_end_point]
        reverse_progress = progress[::-1][1:selected_end_point]

        video_frames = np.concatenate([video_frames, reverse_frame], axis=0)
        progress = np.concatenate([progress, reverse_progress], axis=0)
        progress, goal_distance = self._compute_progress_target(
            video_frames=video_frames,
            linear_progress=progress,
            goal_source_frames=forward_frames,
        )

        video_frames = th.from_numpy(video_frames).float()

        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)

            progress = np.expand_dims(progress, axis=1)
            progress = self.padding_video(progress, self.args.max_length).detach().cpu().numpy()
            progress = np.squeeze(progress, axis=1)
            goal_distance = np.expand_dims(goal_distance, axis=1)
            goal_distance = self.padding_video(goal_distance, self.args.max_length).detach().cpu().numpy()
            goal_distance = np.squeeze(goal_distance, axis=1)
            if len(video_frames.shape) == 1:
                import pdb; pdb.set_trace()
            return video_frames, progress, np.ones(progress.shape[0]), goal_distance
        else:
            return video_frames, progress, np.ones(progress.shape[0]), goal_distance
        
    # ReWiND + freeze (2/8)
    def sample_reverse_uniform_frozen_video_feature(self, data_group, freeze_ratio=0.4):
        # 1. pick a random trajectory
        random_name = self._sample_trajectory_name(data_group)

        progress_dataset = np.asarray(data_group[random_name])  # (N, D)

        # 2. sample forward segment
        start_idx = random.randint(0, len(progress_dataset) // 2)
        end_idx = random.randint(len(progress_dataset) // 2, len(progress_dataset))

        while end_idx - start_idx < 3:
            start_idx = random.randint(0, len(progress_dataset) // 2)
            end_idx = random.randint(len(progress_dataset) // 2, len(progress_dataset))

        video_frames = np.array(progress_dataset)[start_idx:end_idx]
        full_frames = np.array(progress_dataset)[start_idx:]
        forward_frames = video_frames.copy()

        # 3. forward progress
        progress_idx = np.arange(0, video_frames.shape[0], dtype=np.float32) + 1
        progress = progress_idx / len(full_frames)

        # 4. rewind
        selected_end_point = random.randint(2, len(video_frames))
        reverse_frame = video_frames[::-1][1:selected_end_point]
        reverse_progress = progress[::-1][1:selected_end_point]

        video_frames = np.concatenate([video_frames, reverse_frame], axis=0)
        progress = np.concatenate([progress, reverse_progress], axis=0)

        # 5. UNIFORM FREEZING (per-frame)
        # Insert duplicates into the post-rewind sequence instead of overwriting
        # frames in place. This preserves the original temporal order and delays
        # later frames, while keeping the final sequence length unchanged.
        original_length = video_frames.shape[0]
        frozen_frames = [video_frames[0]]
        frozen_progress = [progress[0]]

        for t in range(1, original_length):
            if len(frozen_frames) >= original_length:
                break

            if random.random() < freeze_ratio:
                frozen_frames.append(frozen_frames[-1].copy())
                frozen_progress.append(frozen_progress[-1])
                if len(frozen_frames) >= original_length:
                    break

            frozen_frames.append(video_frames[t])
            frozen_progress.append(progress[t])

        video_frames = np.stack(frozen_frames[:original_length], axis=0)
        progress = np.asarray(frozen_progress[:original_length])
        progress, goal_distance = self._compute_progress_target(
            video_frames=video_frames,
            linear_progress=progress,
            goal_source_frames=forward_frames,
        )

        # 6. convert to tensor
        video_frames = th.from_numpy(video_frames).float()

        # 7. optional padding
        if self.args.subsample_video:
            video_frames = self.padding_video(video_frames, self.args.max_length)

            progress = np.expand_dims(progress, axis=1)
            progress = self.padding_video(progress, self.args.max_length).detach().cpu().numpy()
            progress = np.squeeze(progress, axis=1)
            goal_distance = np.expand_dims(goal_distance, axis=1)
            goal_distance = self.padding_video(goal_distance, self.args.max_length).detach().cpu().numpy()
            goal_distance = np.squeeze(goal_distance, axis=1)

        return video_frames, progress, np.ones(progress.shape[0]), goal_distance




    def padding_video(self, video_frames, max_length):
        video_length = len(video_frames)
        if type(video_frames) == np.ndarray:
            video_frames = th.tensor(video_frames)
        if video_length < max_length:
            # padding last frame
            padding_length = max_length - video_length
            # first_frame = video_frames[0].unsqueeze(0)
            last_frame = video_frames[-1].unsqueeze(0)
            padding_frames = last_frame.repeat(padding_length, 1)
            video_frames = th.cat([video_frames, padding_frames], dim=0)
            # video_frames = th.cat([padding_frames, video_frames], dim=0)
        
        elif video_length > max_length:
            frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
            video_frames = video_frames[frame_idx]

        return video_frames
    
    def __len__(self):
        if self.args.extra_data_ratio == 1:
            return self.args.batch_size * 100
        return int(self.args.batch_size * 100 * (1 - self.args.extra_data_ratio)) + 1


    def __getitem__(self, idx):
        # select a random key
        key_id = random.randint(0, len(self.keys)-1)
        key = self.keys[key_id]
        data_group = self.h5_file[key]

        if self.args.rewind:
            random_num = random.random()
            if random_num < self.args.rewind_ratio:
                # freeze
                if self.args.use_freeze:
                    video_array, progress, class_label, goal_distance = \
                        self.sample_reverse_uniform_frozen_video_feature(
                            data_group, self.args.freeze_ratio
                        )
                else:
                    video_array, progress, class_label, goal_distance = \
                        self.sample_reverse_video_feature(data_group)
            else:
                video_array, progress, class_label, goal_distance = self.sample_video_feature(data_group)
                
        else:
            
            video_array, progress, class_label, goal_distance = self.sample_video_feature(data_group)

        # sample text sample
        if self.sample_neg:
            if random.random() < 0.2:
                text_array = self.sample_negative_text_feature(key)
                progress = np.zeros(progress.shape)
                goal_distance = np.zeros(goal_distance.shape)
                class_label = np.zeros(class_label.shape)
            else:
                text_array = self.sample_text_feature(data_group)
        else:
            text_array = self.sample_text_feature(data_group)


        output_dict = {
            "text_array": text_array,
            "video_array": video_array,
            "progress": progress,
            "goal_distance": goal_distance,
            "class_label": class_label
        }
        return  output_dict
    
