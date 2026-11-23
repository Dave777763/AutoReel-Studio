"""
Video assembly & Ken Burns rendering engine for AutoReel Studio.
Handles 9:16 (1080x1920 @ 30fps) formatting, automatic duration calculation,
smooth Ken Burns camera movements (zoom-in 1.00->1.12, zoom-out 1.12->1.00, subtle pans),
0.3s crossfade transitions between scenes, lower-third subtitle compositing,
and final H.264 + AAC 192kbps MP4 export without watermarks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from proglog import ProgressBarLogger

from subtitles import SubtitleChunk, SubtitleStyle, composite_subtitles_on_frame

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
DEFAULT_CROSSFADE = 0.3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

KEN_BURNS_MODES = [
    "zoom_in",            # 1.00 -> 1.12 centered
    "zoom_out",           # 1.12 -> 1.00 centered
    "pan_left_to_right",  # 1.06 -> 1.12 with subtle horizontal pan L -> R
    "pan_right_to_left",  # 1.12 -> 1.06 with subtle horizontal pan R -> L
    "pan_top_to_bottom",  # 1.05 -> 1.12 with subtle vertical pan T -> B
    "pan_bottom_to_top",  # 1.12 -> 1.05 with subtle vertical pan B -> T
]

KEN_BURNS_LABELS = {
    "auto": "Automático (Alternar movimientos)",
    "zoom_in": "Zoom In (1.00 → 1.12)",
    "zoom_out": "Zoom Out (1.12 → 1.00)",
    "pan_left_to_right": "Paneo Sutil Izquierda → Derecha + Zoom",
    "pan_right_to_left": "Paneo Sutil Derecha → Izquierda + Zoom",
    "pan_top_to_bottom": "Paneo Sutil Arriba → Abajo + Zoom",
    "pan_bottom_to_top": "Paneo Sutil Abajo → Arriba + Zoom",
}


@dataclass
class SceneItem:
    id: int
    path: str
    name: str
    media_type: str  # "image" or "video"
    duration: float  # effective visible duration in timeline (seconds)
    effect: str = "auto"  # Ken Burns mode for images ("auto" or specific key)
    fit_mode: str = "cover"  # "cover" (crop to 9:16) or "blur_pad" (blurred 9:16 background)
    native_duration: Optional[float] = None


class StreamlitRenderLogger(ProgressBarLogger):
    """Bridges MoviePy's internal proglog progress updates to Streamlit UI callbacks."""

    def __init__(self, ui_callback: Optional[Callable[[float, str], None]] = None):
        super().__init__()
        self.ui_callback = ui_callback

    def bars_callback(self, bar, attr, value, old_value=None):
        if not self.ui_callback:
            return
        bar_state = self.bars.get(bar, {})
        total = bar_state.get("total", 0) or 0
        if total > 0 and attr == "index":
            ratio = min(1.0, max(0.0, float(value) / float(total)))
            self.ui_callback(ratio, f"Renderizando fotogramas ({int(ratio * 100)}%)...")


def detect_media_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def get_audio_duration(audio_path: str) -> float:
    """Return exact duration of an audio file in seconds using MoviePy AudioFileClip."""
    from moviepy import AudioFileClip

    clip = AudioFileClip(audio_path)
    dur = float(clip.duration)
    clip.close()
    return round(dur, 2)


def get_video_native_duration(video_path: str) -> float:
    """Return native duration of a video file in seconds."""
    from moviepy import VideoFileClip

    try:
        clip = VideoFileClip(video_path)
        dur = float(clip.duration)
        clip.close()
        return round(dur, 2)
    except Exception:
        return 5.0


def compute_auto_durations(num_scenes: int, total_audio_duration: float) -> List[float]:
    """
    Distribute total audio duration evenly across all scenes so their sum matches
    total_audio_duration to 2 decimal places.
    """
    if num_scenes <= 0 or total_audio_duration <= 0:
        return []
    base = round(total_audio_duration / num_scenes, 2)
    durations = [max(0.5, base) for _ in range(num_scenes)]
    diff = round(total_audio_duration - sum(durations[:-1]), 2)
    if diff >= 0.4:
        durations[-1] = diff
    return durations


def compute_script_synced_durations(
    num_scenes: int,
    total_audio_duration: float,
    subtitles: Optional[List[SubtitleChunk]] = None,
) -> List[float]:
    """
    Synchronize scene cut points to natural sentence/phrase boundaries in the Spanish
    transcript (or numbered list markers and pause gaps), ensuring clean cuts (such as
    Scene 3 -> Scene 4 at ~20.50s) while summing to total_audio_duration.
    """
    if num_scenes <= 0 or total_audio_duration <= 0:
        return []
    if num_scenes == 1:
        return [round(total_audio_duration, 2)]

    # If no subtitles yet, fall back to even distribution
    if not subtitles:
        return compute_auto_durations(num_scenes, total_audio_duration)

    # Collect candidate cut boundaries at phrase ends, sentence punctuation, or pause gaps
    candidate_cuts: List[Tuple[float, float]] = []  # (timestamp, priority_weight)
    for idx in range(len(subtitles) - 1):
        curr_ch = subtitles[idx]
        next_ch = subtitles[idx + 1]
        gap = max(0.0, next_ch.start - curr_ch.end)
        mid_t = round((curr_ch.end + next_ch.start) / 2.0, 2)

        # Boost priority if next chunk starts a numbered item ("1.", "2.", "3.", "4.")
        # or if current chunk ends a sentence (".", "!", "?")
        next_txt = next_ch.text.strip()
        curr_txt = curr_ch.text.strip()
        weight = 1.0 + min(1.5, gap * 2.5)
        if curr_txt.endswith((".", "!", "?")):
            weight += 2.0
        if next_txt[:2] in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.") or next_txt[:1].isdigit():
            weight += 3.5

        candidate_cuts.append((mid_t, weight))

    if len(candidate_cuts) < (num_scenes - 1):
        return compute_auto_durations(num_scenes, total_audio_duration)

    # Select (num_scenes - 1) cut points closest to ideal uniform targets weighted by phrase boundaries
    selected_cuts: List[float] = []
    min_scene_dur = max(1.2, total_audio_duration / (num_scenes * 2.4))

    for cut_idx in range(1, num_scenes):
        ideal_t = (total_audio_duration * cut_idx) / num_scenes
        prev_cut = selected_cuts[-1] if selected_cuts else 0.0
        max_allowed = total_audio_duration - (num_scenes - cut_idx) * min_scene_dur

        best_t = None
        best_score = -1e9
        for cand_t, weight in candidate_cuts:
            if cand_t <= prev_cut + min_scene_dur or cand_t >= max_allowed:
                continue
            dist_penalty = abs(cand_t - ideal_t) / max(1.0, total_audio_duration / num_scenes)
            score = weight - (dist_penalty * 2.2)
            if score > best_score:
                best_score = score
                best_t = cand_t

        if best_t is None:
            best_t = round(max(prev_cut + min_scene_dur, min(ideal_t, max_allowed)), 2)
        selected_cuts.append(round(best_t, 2))

    # Build durations from cut points
    boundaries = [0.0] + selected_cuts + [round(total_audio_duration, 2)]
    durations = [round(max(0.5, boundaries[i + 1] - boundaries[i]), 2) for i in range(num_scenes)]
    # Fix any rounding drift on the final scene
    drift = round(total_audio_duration - sum(durations[:-1]), 2)
    if drift >= 0.5:
        durations[-1] = drift
    return durations


def ease_in_out_sine(u: float) -> float:
    """Smooth sinusoidal easing function in [0, 1] for natural camera motion."""
    u = max(0.0, min(1.0, u))
    return 0.5 * (1.0 - math.cos(math.pi * u))


def prepare_9_16_base_image(
    image_path: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    upscale_factor: float = 1.18,
    fit_mode: str = "cover",
) -> Image.Image:
    """
    Load an image (.webp, .png, .jpg), fix EXIF orientation, and prepare an oversampled
    9:16 RGB canvas so Ken Burns zoom/pan crops remain razor-sharp at 1080x1920.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    canvas_w = int(round(target_w * upscale_factor))
    canvas_h = int(round(target_h * upscale_factor))
    target_ratio = canvas_w / canvas_h
    src_w, src_h = img.size
    src_ratio = src_w / max(1, src_h)

    if fit_mode == "blur_pad" and abs(src_ratio - target_ratio) > 0.08:
        # Blurred 9:16 background + sharp centered foreground
        bg = ImageOps.fit(img, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=36))
        # Darken background slightly for contrast
        bg_arr = (np.array(bg, dtype=np.float32) * 0.72).astype(np.uint8)
        bg = Image.fromarray(bg_arr)

        fg = img.copy()
        fg.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        paste_x = (canvas_w - fg.width) // 2
        paste_y = (canvas_h - fg.height) // 2
        bg.paste(fg, (paste_x, paste_y))
        return bg

    # Default "cover": center crop to 9:16 at oversampled resolution
    return ImageOps.fit(
        img,
        (canvas_w, canvas_h),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def render_ken_burns_frame(
    base_img: Image.Image,
    progress: float,
    mode: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> np.ndarray:
    """
    Generate a single 1080x1920 RGB numpy frame at normalized time `progress` in [0, 1]
    using sub-pixel affine crop (`Image.Transform.EXTENT`) for smooth, jitter-free Ken Burns motion.
    """
    u = ease_in_out_sine(progress)
    bw, bh = base_img.size

    # Default parameters: zoom from z_start to z_end, normalized center (cx, cy) in [0, 1]
    if mode == "zoom_in":
        z_start, z_end = 1.00, 1.12
        cx_start, cx_end = 0.50, 0.50
        cy_start, cy_end = 0.50, 0.50
    elif mode == "zoom_out":
        z_start, z_end = 1.12, 1.00
        cx_start, cx_end = 0.50, 0.50
        cy_start, cy_end = 0.50, 0.50
    elif mode == "pan_left_to_right":
        z_start, z_end = 1.06, 1.12
        cx_start, cx_end = 0.15, 0.85
        cy_start, cy_end = 0.50, 0.50
    elif mode == "pan_right_to_left":
        z_start, z_end = 1.12, 1.06
        cx_start, cx_end = 0.85, 0.15
        cy_start, cy_end = 0.50, 0.50
    elif mode == "pan_top_to_bottom":
        z_start, z_end = 1.05, 1.12
        cx_start, cx_end = 0.50, 0.50
        cy_start, cy_end = 0.18, 0.82
    elif mode == "pan_bottom_to_top":
        z_start, z_end = 1.12, 1.05
        cx_start, cx_end = 0.50, 0.50
        cy_start, cy_end = 0.82, 0.18
    else:
        z_start, z_end = 1.00, 1.12
        cx_start, cx_end = 0.50, 0.50
        cy_start, cy_end = 0.50, 0.50

    zoom = z_start + (z_end - z_start) * u
    cx = cx_start + (cx_end - cx_start) * u
    cy = cy_start + (cy_end - cy_start) * u

    # Visible crop window size inside base_img
    crop_w = bw / zoom
    crop_h = bh / zoom

    max_shift_x = max(0.0, bw - crop_w)
    max_shift_y = max(0.0, bh - crop_h)

    x0 = max_shift_x * cx
    y0 = max_shift_y * cy
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    frame_img = base_img.transform(
        (target_w, target_h),
        Image.Transform.EXTENT,
        (x0, y0, x1, y1),
        resample=Image.Resampling.BILINEAR,
    )
    return np.array(frame_img, dtype=np.uint8)


class VideoClipAccessor:
    """
    Wraps a MoviePy VideoFileClip to provide fast, aspect-corrected 1080x1920 RGB frames
    at any local timestamp `t` (looping or trimming cleanly if duration differs from native).
    """

    def __init__(
        self,
        video_path: str,
        clip_duration: float,
        fit_mode: str = "cover",
        target_w: int = TARGET_WIDTH,
        target_h: int = TARGET_HEIGHT,
    ):
        from moviepy import VideoFileClip

        self.clip = VideoFileClip(video_path)
        self.native_duration = max(0.1, float(self.clip.duration or clip_duration))
        self.clip_duration = max(0.1, float(clip_duration))
        self.fit_mode = fit_mode
        self.target_w = target_w
        self.target_h = target_h

    def get_frame(self, t: float) -> np.ndarray:
        # Loop seamlessly if requested scene duration exceeds native clip length
        local_t = t % self.native_duration
        local_t = min(max(0.0, local_t), max(0.0, self.native_duration - 1e-3))
        raw_frame = self.clip.get_frame(local_t)
        img = Image.fromarray(raw_frame).convert("RGB")

        src_w, src_h = img.size
        if src_w == self.target_w and src_h == self.target_h:
            return raw_frame

        target_ratio = self.target_w / self.target_h
        src_ratio = src_w / max(1, src_h)

        if self.fit_mode == "blur_pad" and abs(src_ratio - target_ratio) > 0.08:
            bg = ImageOps.fit(img, (self.target_w, self.target_h), method=Image.Resampling.BILINEAR)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
            bg_arr = (np.array(bg, dtype=np.float32) * 0.72).astype(np.uint8)
            bg = Image.fromarray(bg_arr)

            fg = img.copy()
            fg.thumbnail((self.target_w, self.target_h), Image.Resampling.BILINEAR)
            px = (self.target_w - fg.width) // 2
            py = (self.target_h - fg.height) // 2
            bg.paste(fg, (px, py))
            return np.array(bg, dtype=np.uint8)

        fitted = ImageOps.fit(
            img,
            (self.target_w, self.target_h),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        return np.array(fitted, dtype=np.uint8)

    def close(self):
        try:
            self.clip.close()
        except Exception:
            pass


@dataclass
class TimelineSegment:
    scene: SceneItem
    start_time: float
    end_time: float
    render_duration: float
    resolved_effect: str
    base_image: Optional[Image.Image] = None
    video_accessor: Optional[VideoClipAccessor] = None

    def get_frame_at_local(self, local_t: float) -> np.ndarray:
        local_t = max(0.0, min(local_t, self.render_duration))
        if self.scene.media_type == "image" and self.base_image is not None:
            progress = local_t / max(0.001, self.render_duration)
            return render_ken_burns_frame(
                self.base_image,
                progress=progress,
                mode=self.resolved_effect,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
            )
        if self.scene.media_type == "video" and self.video_accessor is not None:
            return self.video_accessor.get_frame(local_t)
        return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    def close(self):
        if self.video_accessor is not None:
            self.video_accessor.close()


def build_timeline_segments(
    scenes: List[SceneItem],
    crossfade_sec: float = DEFAULT_CROSSFADE,
) -> Tuple[List[TimelineSegment], float]:
    """
    Construct overlapping timeline segments so each scene has its full effective duration
    while smoothly crossfading over `crossfade_sec` (0.3s) into the next scene.
    Wait-free math:
      - Scene i starts at sum(d_0 ... d_{i-1})
      - To allow a 0.3s crossfade into scene i+1 during [start_{i+1}, start_{i+1} + 0.3s],
        scene i remains active for `d_i + crossfade_sec` (except the last scene).
    """
    segments: List[TimelineSegment] = []
    cursor = 0.0
    n = len(scenes)

    for idx, sc in enumerate(scenes):
        eff_dur = max(0.4, float(sc.duration))
        is_last = idx == (n - 1)
        xfade = min(crossfade_sec, eff_dur * 0.45) if not is_last else 0.0
        render_dur = eff_dur + xfade

        resolved_effect = sc.effect
        if resolved_effect == "auto" or resolved_effect not in KEN_BURNS_MODES:
            resolved_effect = KEN_BURNS_MODES[idx % len(KEN_BURNS_MODES)]

        base_img = None
        vid_acc = None
        if sc.media_type == "image":
            base_img = prepare_9_16_base_image(
                sc.path,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
                fit_mode=sc.fit_mode,
            )
        elif sc.media_type == "video":
            vid_acc = VideoClipAccessor(
                sc.path,
                clip_duration=render_dur,
                fit_mode=sc.fit_mode,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
            )

        seg = TimelineSegment(
            scene=sc,
            start_time=cursor,
            end_time=cursor + render_dur,
            render_duration=render_dur,
            resolved_effect=resolved_effect,
            base_image=base_img,
            video_accessor=vid_acc,
        )
        segments.append(seg)
        cursor += eff_dur

    total_duration = max(0.5, cursor)
    return segments, total_duration


def compose_frame_at_time(
    t: float,
    segments: List[TimelineSegment],
    crossfade_sec: float,
    subtitles: Optional[List[SubtitleChunk]],
    subtitle_style: Optional[SubtitleStyle],
    subtitle_cache: dict,
) -> np.ndarray:
    """
    Evaluate the exact 1080x1920 RGB frame at global timestamp `t`:
    1. Identifies active timeline segment(s) at `t`
    2. Applies 0.3s alpha crossfade when transitioning between scene i and scene i+1
    3. Composites lower-third Spanish subtitles with dark stroke and highlighted keywords
    """
    active: List[TimelineSegment] = [
        seg for seg in segments if seg.start_time - 1e-4 <= t <= seg.end_time + 1e-4
    ]

    if not active and segments:
        active = [segments[-1] if t >= segments[-1].start_time else segments[0]]

    if len(active) == 1:
        seg = active[0]
        frame = seg.get_frame_at_local(t - seg.start_time)
    elif len(active) >= 2:
        # Transition overlap between outgoing seg_a and incoming seg_b
        seg_a, seg_b = active[-2], active[-1]
        frame_a = seg_a.get_frame_at_local(t - seg_a.start_time).astype(np.float32)
        frame_b = seg_b.get_frame_at_local(t - seg_b.start_time).astype(np.float32)
        overlap_len = max(0.01, seg_a.end_time - seg_b.start_time)
        alpha = max(0.0, min(1.0, (t - seg_b.start_time) / overlap_len))
        # Smooth cosine crossfade curve
        smooth_alpha = ease_in_out_sine(alpha)
        blended = frame_a * (1.0 - smooth_alpha) + frame_b * smooth_alpha
        frame = blended.astype(np.uint8)
    else:
        frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    if subtitles and subtitle_style:
        frame = composite_subtitles_on_frame(
            frame,
            t=t,
            chunks=subtitles,
            style=subtitle_style,
            cache=subtitle_cache,
        )

    return frame


def render_preview_frame(
    t: float,
    scenes: List[SceneItem],
    crossfade_sec: float = DEFAULT_CROSSFADE,
    subtitles: Optional[List[SubtitleChunk]] = None,
    subtitle_style: Optional[SubtitleStyle] = None,
) -> Image.Image:
    """
    Render a single 1080x1920 preview frame at timestamp `t` for live UI inspection
    in Streamlit before launching the full MP4 export.
    """
    segments, total_dur = build_timeline_segments(scenes, crossfade_sec=crossfade_sec)
    try:
        clamped_t = max(0.0, min(t, total_dur - 0.01))
        cache: dict = {}
        arr = compose_frame_at_time(
            clamped_t,
            segments=segments,
            crossfade_sec=crossfade_sec,
            subtitles=subtitles,
            subtitle_style=subtitle_style,
            subtitle_cache=cache,
        )
        return Image.fromarray(arr)
    finally:
        for seg in segments:
            seg.close()


def assemble_and_export_reel(
    audio_path: str,
    scenes: List[SceneItem],
    output_path: str,
    crossfade_sec: float = DEFAULT_CROSSFADE,
    subtitles: Optional[List[SubtitleChunk]] = None,
    subtitle_style: Optional[SubtitleStyle] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.12,
    preset: str = "medium",
    crf: int = 20,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Assemble the full 9:16 (1080x1920 @ 30fps) vertical reel and export to MP4
    with H.264 video (`libx264`, `yuv420p`) and AAC 192kbps audio without watermarks.
    Strictly respects the manual/assigned duration of each scene without redistributing
    durations on export, and optionally mixes a background music track (looped/trimmed
    to final_duration, 1.5s fade-in/out, ducked to bgm_volume, default 12% / -18 dB).
    """
    from moviepy import AudioFileClip, CompositeAudioClip, VideoClip, afx

    if not scenes:
        raise ValueError("La lista de escenas está vacía.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if progress_callback:
        progress_callback(0.02, "Preparando escenas 9:16 (1080x1920) y movimientos Ken Burns...")

    # Strictly honor each scene's assigned duration from `scenes`
    segments, timeline_duration = build_timeline_segments(scenes, crossfade_sec=crossfade_sec)
    voice_clip = AudioFileClip(audio_path)
    bgm_clip = None
    mixed_audio = None

    final_duration = round(max(0.5, float(timeline_duration)), 2)

    if voice_clip.duration > final_duration:
        voice_track = voice_clip.subclipped(0, final_duration)
    else:
        voice_track = voice_clip

    # Optional background music track with ducking + 1.5s fade-in / fade-out
    if bgm_path and os.path.exists(bgm_path) and bgm_volume > 0.0:
        bgm_clip = AudioFileClip(bgm_path)
        if bgm_clip.duration < final_duration:
            bgm_track = bgm_clip.with_effects([afx.AudioLoop(duration=final_duration)])
        else:
            bgm_track = bgm_clip.subclipped(0, final_duration)

        fade_dur = min(1.5, final_duration * 0.25)
        bgm_track = bgm_track.with_volume_scaled(float(bgm_volume)).with_effects(
            [
                afx.AudioFadeIn(fade_dur),
                afx.AudioFadeOut(fade_dur),
            ]
        )
        mixed_audio = CompositeAudioClip([voice_track, bgm_track]).with_duration(final_duration)
    else:
        mixed_audio = voice_track

    subtitle_cache: dict = {}

    def make_frame(t: float) -> np.ndarray:
        return compose_frame_at_time(
            t=t,
            segments=segments,
            crossfade_sec=crossfade_sec,
            subtitles=subtitles,
            subtitle_style=subtitle_style,
            subtitle_cache=subtitle_cache,
        )

    try:
        video_clip = VideoClip(frame_function=make_frame, duration=final_duration)
        video_clip = video_clip.with_fps(TARGET_FPS).with_audio(mixed_audio)

        logger = StreamlitRenderLogger(progress_callback) if progress_callback else "bar"

        if progress_callback:
            progress_callback(0.05, "Codificando MP4 final (H.264 1080x1920 @ 30fps + AAC 192kbps)...")

        temp_audio_path = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            "_temp_autoreel_audio.m4a",
        )

        video_clip.write_videofile(
            output_path,
            fps=TARGET_FPS,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset=preset,
            threads=max(2, os.cpu_count() or 4),
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-movflags", "+faststart",
                "-profile:v", "high",
                "-level", "4.1",
            ],
            logger=logger,
        )

        if progress_callback:
            progress_callback(1.0, "¡Exportación completada con éxito!")

        return output_path
    finally:
        for clip_obj in (voice_clip, bgm_clip):
            if clip_obj is not None:
                try:
                    clip_obj.close()
                except Exception:
                    pass
        for seg in segments:
            seg.close()
