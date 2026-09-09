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

# cv2 and pandas are imported INSIDE the methods that need them, not here.
# `label_window` / `bucket_for_label` below are pure and stdlib-only, and
# src/data/recut_site.py --dry-run imports them to project a bucket census in an
# environment that may have neither installed — the same reason annotations.py
# and scripts/audit_source_data.py are stdlib-only. Cutting clips still needs
# both; that import just happens at the point of use.


#: The thesis' constants, as module-level defaults so the pure `label_window`
#: below carries them and the class no longer owns the only copy.
STRONG_THRESHOLD = 0.5       # stimulation / ventilation
SUCTION_THRESHOLD = 0.25     # suction is brief; the thesis used a lower bar
NON_TARGET_THRESHOLD = 0.20  # explicit non-target
WEAK_THRESHOLD = 0.20        # purity guard for "other" target leakage


def label_window(stim, vent, suct, nt, other, length_clip,
                 strong=STRONG_THRESHOLD, suction=SUCTION_THRESHOLD,
                 non_target=NON_TARGET_THRESHOLD, weak=WEAK_THRESHOLD,
                 for_predict=False) -> str:
    """Per-activity overlap (ms) within one window -> the thesis' label string.

    Extracted VERBATIM from `label_all_clips`, which now delegates to it. It is
    a pure function of the five overlap totals, so anything that already knows
    the annotation intervals can predict the label WITHOUT decoding video —
    which is what lets `src/data/recut_site.py --dry-run` project an exact
    bucket census before spending hours in ffmpeg. A second copy of this
    cascade would be free to drift from the one that cuts the clips; there is
    deliberately only one.
    """
    if for_predict:
        max_overlap = max(stim, vent, suct)
        if max_overlap >= length_clip * strong:
            return ("Stimulation" if max_overlap == stim else
                    "Ventilation" if max_overlap == vent else "Suction")
        return "Non-target"

    stim_strong = stim >= length_clip * strong
    vent_strong = vent >= length_clip * strong
    suct_strong = suct >= length_clip * suction
    strong_count = int(stim_strong) + int(vent_strong) + int(suct_strong)
    stim_weak = stim >= length_clip * weak
    vent_weak = vent >= length_clip * weak
    suct_weak = suct >= length_clip * weak
    stim_any, vent_any, suct_any = stim > 0, vent > 0, suct > 0

    if strong_count >= 2:
        combo = "+".join(sorted(
            [n for n, f in (("stimulation", stim_strong), ("ventilation", vent_strong),
                            ("suction", suct_strong)) if f]))
        return f"Target overlap:{combo}"
    if stim_strong and not vent_weak and not suct_weak:
        return "Stimulation"
    if vent_strong and not stim_weak and not suct_weak and other == 0:
        return "Ventilation"
    if suct_strong and not stim_weak and not vent_weak:
        return "Suction"
    if (nt >= length_clip * non_target and not stim_any and not vent_any
            and not suct_any and other == 0):
        return "Non-target"

    any_count = int(stim_any) + int(vent_any) + int(suct_any)
    if not stim_any and not vent_any and not suct_any and nt == 0 and other == 0:
        return "No overlap"
    if any_count == 1:
        which = "Stimulation" if stim_any else ("Ventilation" if vent_any else "Suction")
        return f"Partial:{which}"
    if any_count >= 2:
        combo = "+".join(sorted(
            [n for n, f in (("stimulation", stim_any), ("ventilation", vent_any),
                            ("suction", suct_any)) if f]))
        return f"Target overlap partial:{combo}"
    return "No label"


def windows(video_duration_ms, effective_duration_ms, segment_ms=3000, shift_ms=1000):
    """The (start, end) window pairs `split_video` emits, as a pure function.

    Extracted for the same reason as `label_window`: the clip COUNT is decided
    here and the LABEL there, so a dry run needs both to project a census that
    matches what the cut will write. Note the asymmetric stop condition, which
    is the thesis' and is kept exactly: a window must START before the last
    annotation ends, but must END within the video — so the unannotated tail of
    an episode is never cut, while its unannotated head is.
    """
    out = []
    start, end = 0, segment_ms
    while start < effective_duration_ms and end <= video_duration_ms:
        out.append((start, end))
        start += shift_ms
        end += shift_ms
    return out


def bucket_for_label(label: str):
    """Label string -> (bucket number, output subdirectory).

    The trailing `_N` of every clip filename and the directory it is filed
    under, both of which `build_manifest.py` reads back. Extracted from
    `save_clips` for the same reason as `label_window`: the dry run must
    project the SAME buckets the cut will write.
    """
    if label == "Stimulation":
        return 1, "videos/stimulation/"
    if label == "Ventilation":
        return 2, "videos/ventilation/"
    if label == "Suction":
        return 3, "videos/suction/"
    if label == "Non-target":
        return 0, "videos/non_target/"
    if label == "No overlap":
        return 4, "videos/no_overlap/"
    if label == "No label":
        return 5, "videos/no_label/"
    if label.startswith("Partial:"):
        return 6, f"videos/partial/{label.split(':')[1].lower()}/"
    if label.startswith("Target overlap:"):
        return 7, f"videos/target_overlap/{label.split(':')[1]}/"
    if label.startswith("Target overlap partial:"):
        return 8, f"videos/partial/{label.split(':')[1]}/"
    return 5, "videos/no_label/"


class VideoDataProcessor:
    def __init__(self, video_file, annotation_file, segment_size, shift,
                 date_of_recording, folder_name, for_predict=False, base_dir=None):
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

        # Where the RAW inputs live: <BasePath>/Unprocessed_data/{videos,anot_files}.
        # It must be passed in, because the input staging dir and the output clip
        # root are independent paths — in the documented layout they are SIBLINGS
        # (.../Data_processing and .../Processed_video_clips), so deriving the
        # input root from the output one read a directory that does not exist and
        # every case failed. `os.path.dirname(folder_name)` survives only as the
        # fallback for the old single-argument call.
        self.BasePath = str(base_dir) if base_dir is not None else os.path.dirname(self.folder_name)
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
        import pandas as pd
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
        import cv2
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
        import pandas as pd
        video_duration, _, _, _, _ = self.load_video_data()

        path = os.path.join(self.BasePath, "Unprocessed_data", "anot_files", self.annotation_file)
        with open(path, "r") as file:
            lines = file.readlines()
        data = [line.strip().split("\t") for line in lines]
        df_an = pd.DataFrame(data, columns=["Event", "Start", "End", "Duration", "Original_Event_Name"])
        df_an = df_an[df_an["Original_Event_Name"] != "Newborn visible in video frame"].reset_index(drop=True)
        df_an["End"] = df_an["End"].astype(int)
        effective_duration = min(video_duration, df_an["End"].max()) if len(df_an) else video_duration

        return windows(video_duration, effective_duration,
                       self.segment_size * 1000, self.shift * 1000)

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

            label = label_window(
                stim, vent, suct, nt, other, length_clip,
                strong=self.STRONG_THRESHOLD, suction=self.suction_threshold,
                non_target=self.non_target_threshold, weak=self.weak_threshold,
                for_predict=self.for_predict)

            tag = self._overlap_suffix(stim, vent, suct, length_clip)
            labeled.append((clip_start, clip_end, label, tag))
        return labeled

    # ------------------------------------------------------------------ saving
    def save_clips(self):
        import cv2
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
                # one cap.read() advances exactly ONE frame, so the cursor moves
                # by one frame period. Using `shift / fps` happened to agree only
                # because SHIFT == 1; any other stride truncated the clip.
                current += 1000.0 / fps if fps else 0

            label_num, out_dir = bucket_for_label(label)

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
