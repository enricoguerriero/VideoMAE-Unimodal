"""
data_process.py

VIDEO-ONLY port of the multimodal thesis' DataProcessor (the accelerometer and
audio branches are removed). It cuts a video into labeled 3-second clips exactly
as the thesis did, so regenerated clips are identical to the ones the MoViNet
video base model was trained on.

IMPORTANT — this is the OPTIONAL "regenerate from raw" path. The recommended
path is to index the already-processed clips with build_manifest.py. Use this
only when you need to re-cut clips from raw videos + CLEANED annotation files.

Kept verbatim from the thesis (video path only):
    * split_video(): 3 s window, 1 s stride, from 0 to min(video_dur, last_annotation_end).
    * label thresholds: strong (stim/vent) = 0.50, suction = 0.25,
      non_target = 0.20, weak = 0.20.
    * single-label taxonomy with purity rules; ambiguous clips routed to
      no_overlap / no_label / partial / target_overlap buckets.
    * frames resized to 256x192; filename
      {case}_interval_{n}_start_{ms}_end_{ms}{tag}_{labelnum}.mp4.

Offsets are NOT used: the video path works in raw video/annotation milliseconds;
`offset_video`/`offset_acc` only ever affected the (removed) accelerometer branch.

Expected input layout (per site):
    <base>/Unprocessed_data/videos/<case_id>.mp4
    <base>/Unprocessed_data/anot_files/<case_id>.txt   (5-col tab-separated:
        Event, Start_ms, End_ms, Duration_ms, Original_Event_Name)
"""

import os
from collections import Counter

import cv2
import pandas as pd


class VideoDataProcessor:
    def __init__(self, video_file, annotation_file, segment_size, shift,
                 date_of_recording, folder_name, for_predict=False):
        self.video_file = video_file
        self.annotation_file = annotation_file
        self.segment_size = segment_size
        self.shift = shift
        self.date_of_recording = date_of_recording
        self.for_predict = for_predict
        self.folder_name = folder_name

        # Label thresholds (identical to the thesis).
        self.STRONG_THRESHOLD = 0.5      # stimulation / ventilation
        self.suction_threshold = 0.25    # suction
        self.non_target_threshold = 0.20 # explicit non-target
        self.weak_threshold = 0.20       # purity guard for "other" target leakage

        self.BasePath = os.path.dirname(self.folder_name)
        os.makedirs(self.folder_name, exist_ok=True)
        os.makedirs(os.path.join(self.folder_name, "videos", "no_label"), exist_ok=True)

        self.stimulation_intervals = None
        self.ventilation_intervals = None
        self.suction_intervals = None
        self.non_target_intervals = None
        self.other_intervals = None
        self.video_length = None

    # ------------------------------------------------------------------ intervals
    def merge_intervals(self, intervals):
        if not intervals:
            return []
        sorted_iv = sorted(intervals)
        merged = [sorted_iv[0]]
        for start, end in sorted_iv[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def load_annotation_data(self):
        path = os.path.join(self.BasePath, "Unprocessed_data", "anot_files", self.annotation_file)
        with open(path, "r") as file:
            lines = file.readlines()
        data = [line.strip().split("\t") for line in lines]

        df_an = pd.DataFrame(data, columns=["Event", "Start", "End", "Duration", "Original_Event_Name"])
        df_an = df_an[df_an["Original_Event_Name"] != "Newborn visible in video frame"].reset_index(drop=True)
        df_an.drop("Original_Event_Name", axis=1, inplace=True)
        df_an["Start"] = df_an["Start"].astype(int)
        df_an["End"] = df_an["End"].astype(int)

        map_labels = {"Ignored label": 4, "Suction": 3, "Ventilation": 2, "Stimulation": 1, "Non-target": 0}
        label_df = df_an.copy()
        label_df["Event"] = label_df["Event"].map(map_labels)
        df_others = label_df[label_df["Event"] == 4].reset_index(drop=True)
        df_filtered = label_df[label_df["Event"] != 4].reset_index(drop=True)

        if self.stimulation_intervals is None:
            self.stimulation_intervals, self.ventilation_intervals = [], []
            self.suction_intervals, self.non_target_intervals, self.other_intervals = [], [], []
            for _, row in df_others.iterrows():
                if row["Event"] == 4:
                    self.other_intervals.append((row["Start"], row["End"]))
            for _, row in df_filtered.iterrows():
                iv = (row["Start"], row["End"])
                if row["Event"] == 1:
                    self.stimulation_intervals.append(iv)
                elif row["Event"] == 2:
                    self.ventilation_intervals.append(iv)
                elif row["Event"] == 3:
                    self.suction_intervals.append(iv)
                elif row["Event"] == 0:
                    self.non_target_intervals.append(iv)
            self.stimulation_intervals = self.merge_intervals(self.stimulation_intervals)
            self.ventilation_intervals = self.merge_intervals(self.ventilation_intervals)
            self.suction_intervals = self.merge_intervals(self.suction_intervals)
            self.non_target_intervals = self.merge_intervals(self.non_target_intervals)
            self.other_intervals = self.merge_intervals(self.other_intervals)

        return (df_filtered, df_an, self.stimulation_intervals, self.ventilation_intervals,
                self.suction_intervals, self.non_target_intervals, self.other_intervals)

    # ------------------------------------------------------------------ video
    def load_video_data(self):
        path = os.path.join(self.BasePath, "Unprocessed_data", "videos", self.video_file)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_ms = int((total_frames / fps) * 1000) if fps else 0
        self.video_length = video_duration_ms
        return video_duration_ms, cap, fps, frame_width, frame_height

    def split_video(self):
        clips = []
        start_time, segment_duration = 0, self.segment_size * 1000
        end_time = start_time + segment_duration
        video_duration, _, _, _, _ = self.load_video_data()

        path = os.path.join(self.BasePath, "Unprocessed_data", "anot_files", self.annotation_file)
        with open(path, "r") as file:
            lines = file.readlines()
        data = [line.strip().split("\t") for line in lines]
        df_an = pd.DataFrame(data, columns=["Event", "Start", "End", "Duration", "Original_Event_Name"])
        df_an = df_an[df_an["Original_Event_Name"] != "Newborn visible in video frame"].reset_index(drop=True)
        df_an["End"] = df_an["End"].astype(int)
        effective_duration = min(video_duration, df_an["End"].max()) if len(df_an) else video_duration

        while start_time < effective_duration and end_time <= video_duration:
            clips.append((start_time, end_time))
            start_time += self.shift * 1000
            end_time += self.shift * 1000
        return clips

    # ------------------------------------------------------------------ labeling
    @staticmethod
    def overlap_ms(clip_start, clip_end, intervals):
        total = 0
        for s, e in intervals:
            if clip_start < e and clip_end > s:
                total += min(clip_end, e) - max(clip_start, s)
        return total

    def _overlap_suffix(self, stim, vent, suct, length_clip):
        parts = []
        if stim > 0:
            parts.append(f"_stim{stim / length_clip:.2f}")
        if vent > 0:
            parts.append(f"_vent{vent / length_clip:.2f}")
        if suct > 0:
            parts.append(f"_suct{suct / length_clip:.2f}")
        return "".join(parts)

    def label_all_clips(self):
        video_clips = self.split_video()
        (_, _, stim_iv, vent_iv, suct_iv, nt_iv, other_iv) = self.load_annotation_data()
        labeled, length_clip = [], self.segment_size * 1000

        for clip_start, clip_end in video_clips:
            stim = self.overlap_ms(clip_start, clip_end, stim_iv)
            vent = self.overlap_ms(clip_start, clip_end, vent_iv)
            suct = self.overlap_ms(clip_start, clip_end, suct_iv)
            nt = self.overlap_ms(clip_start, clip_end, nt_iv)
            other = self.overlap_ms(clip_start, clip_end, other_iv)

            if not self.for_predict:
                stim_strong = stim >= length_clip * self.STRONG_THRESHOLD
                vent_strong = vent >= length_clip * self.STRONG_THRESHOLD
                suct_strong = suct >= length_clip * self.suction_threshold
                strong_count = int(stim_strong) + int(vent_strong) + int(suct_strong)
                stim_weak = stim >= length_clip * self.weak_threshold
                vent_weak = vent >= length_clip * self.weak_threshold
                suct_weak = suct >= length_clip * self.weak_threshold
                stim_any, vent_any, suct_any = stim > 0, vent > 0, suct > 0

                if strong_count >= 2:
                    combo = "+".join(sorted(
                        [n for n, f in (("stimulation", stim_strong), ("ventilation", vent_strong),
                                        ("suction", suct_strong)) if f]))
                    label = f"Target overlap:{combo}"
                elif stim_strong and not vent_weak and not suct_weak:
                    label = "Stimulation"
                elif vent_strong and not stim_weak and not suct_weak and other == 0:
                    label = "Ventilation"
                elif suct_strong and not stim_weak and not vent_weak:
                    label = "Suction"
                elif nt >= length_clip * self.non_target_threshold and not stim_any and not vent_any and not suct_any and other == 0:
                    label = "Non-target"
                else:
                    any_count = int(stim_any) + int(vent_any) + int(suct_any)
                    if not stim_any and not vent_any and not suct_any and nt == 0 and other == 0:
                        label = "No overlap"
                    elif any_count == 1:
                        which = "Stimulation" if stim_any else ("Ventilation" if vent_any else "Suction")
                        label = f"Partial:{which}"
                    elif any_count >= 2:
                        combo = "+".join(sorted(
                            [n for n, f in (("stimulation", stim_any), ("ventilation", vent_any),
                                            ("suction", suct_any)) if f]))
                        label = f"Target overlap partial:{combo}"
                    else:
                        label = "No label"
            else:
                max_overlap = max(stim, vent, suct)
                if max_overlap >= length_clip * self.STRONG_THRESHOLD:
                    label = ("Stimulation" if max_overlap == stim else
                             "Ventilation" if max_overlap == vent else "Suction")
                else:
                    label = "Non-target"

            tag = self._overlap_suffix(stim, vent, suct, length_clip)
            labeled.append((clip_start, clip_end, label, tag))
        return labeled

    # ------------------------------------------------------------------ saving
    def save_clips(self):
        labeled = self.label_all_clips()
        _, cap, fps, _, _ = self.load_video_data()
        for index, (clip_start, clip_end, label, tag) in enumerate(labeled):
            cap.set(cv2.CAP_PROP_POS_MSEC, clip_start)
            current, frames = clip_start, []
            while current <= clip_end:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(frame, (256, 192)))
                current += (self.shift / fps) * 1000 if fps else 0

            if label == "Stimulation":
                label_num, out_dir = 1, "videos/stimulation/"
            elif label == "Ventilation":
                label_num, out_dir = 2, "videos/ventilation/"
            elif label == "Suction":
                label_num, out_dir = 3, "videos/suction/"
            elif label == "Non-target":
                label_num, out_dir = 0, "videos/non_target/"
            elif label == "No overlap":
                label_num, out_dir = 4, "videos/no_overlap/"
            elif label == "No label":
                label_num, out_dir = 5, "videos/no_label/"
            elif label.startswith("Partial:"):
                label_num, out_dir = 6, f"videos/partial/{label.split(':')[1].lower()}/"
            elif label.startswith("Target overlap:"):
                label_num, out_dir = 7, f"videos/target_overlap/{label.split(':')[1]}/"
            elif label.startswith("Target overlap partial:"):
                label_num, out_dir = 8, f"videos/partial/{label.split(':')[1]}/"
            else:
                label_num, out_dir = 5, "videos/no_label/"

            out_dir = os.path.join(self.folder_name, out_dir)
            os.makedirs(out_dir, exist_ok=True)
            if not frames:
                continue
            fname = f"{self.date_of_recording}_interval_{index + 1}_start_{clip_start}_end_{clip_end}{tag}_{label_num}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(out_dir, fname), fourcc, fps, (256, 192))
            for f in frames:
                out.write(f)
            out.release()
        cap.release()
        cv2.destroyAllWindows()

    def run_video_only(self):
        self.save_clips()
