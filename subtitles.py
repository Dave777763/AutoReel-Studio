"""
Subtitles engine for AutoReel Studio.
Uses faster-whisper for Spanish automatic speech recognition with word-level timestamps,
and Pillow for rendering modern lower-third dynamic captions with dark strokes and keyword highlighting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Common Spanish stopwords to exclude when auto-detecting keywords to highlight
SPANISH_STOPWORDS: Set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "segun", "según",
    "sin", "so", "sobre", "tras", "versus", "via", "vía", "y", "e", "ni", "que",
    "o", "u", "pero", "mas", "más", "sino", "aunque", "como", "porque", "pues",
    "si", "sí", "no", "ya", "muy", "tan", "tanto", "tanta", "tantos", "tantas",
    "yo", "tu", "tú", "el", "él", "ella", "ello", "nosotros", "nosotras",
    "vosotros", "vosotras", "ellos", "ellas", "usted", "ustedes", "mi", "mí",
    "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
    "me", "te", "se", "nos", "os", "le", "les", "este", "esta", "esto", "estos",
    "estas", "ese", "esa", "eso", "esos", "esas", "aquel", "aquella", "aquello",
    "aquellos", "aquellas", "es", "son", "fue", "era", "eran", "ser", "estar",
    "está", "están", "estaba", "ha", "han", "hay", "había", "hacer", "hace",
    "tiene", "tienen", "tenía", "puede", "pueden", "todo", "toda", "todos",
    "todas", "otro", "otra", "otros", "otras", "mismo", "misma", "cada", "cual",
    "quien", "quién", "cuando", "cuándo", "donde", "dónde", "como", "cómo",
}


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    is_keyword: bool = False


@dataclass
class SubtitleChunk:
    id: int
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)
    custom_text: Optional[str] = None
    custom_keywords: Optional[List[str]] = None

    @property
    def text(self) -> str:
        if self.custom_text is not None:
            return self.custom_text
        return " ".join(w.word for w in self.words).strip()

    @property
    def keywords_list(self) -> List[str]:
        if self.custom_keywords is not None:
            return self.custom_keywords
        return [clean_token(w.word) for w in self.words if w.is_keyword]


@dataclass
class SubtitleStyle:
    font_size: int = 66
    text_color: Tuple[int, int, int] = (255, 255, 255)
    highlight_color: Tuple[int, int, int] = (255, 215, 0)  # Vibrant gold/yellow
    active_word_color: Tuple[int, int, int] = (0, 245, 212)  # Cyan/mint accent
    stroke_color: Tuple[int, int, int] = (12, 12, 16)
    stroke_width: int = 7
    shadow_offset: Tuple[int, int] = (0, 5)
    y_position_ratio: float = 0.77  # Lower third (around 77% of 1920 = ~1478px)
    uppercase: bool = True
    highlight_mode: str = "keywords_and_active"  # "keywords_only", "active_word", "keywords_and_active"
    bg_pill: bool = True
    bg_opacity: int = 120


def clean_token(word: str) -> str:
    """Strip punctuation and lowercase for keyword matching."""
    return re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]", "", word).lower().strip()


# Common Spanish ASR corrections for AI tool names and technical acronyms
SPANISH_TRANSCRIPT_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bsube\s*pedef\s*es\b", re.IGNORECASE), "sube PDFs"),
    (re.compile(r"\bsubepedef\s*es\b", re.IGNORECASE), "sube PDFs"),
    (re.compile(r"\bsube\s*pedefes\b", re.IGNORECASE), "sube PDFs"),
    (re.compile(r"\bsubepedefes\b", re.IGNORECASE), "sube PDFs"),
    (re.compile(r"\bpedef\s*es\b", re.IGNORECASE), "PDFs"),
    (re.compile(r"\bpedefes\b", re.IGNORECASE), "PDFs"),
    (re.compile(r"\bpdf\s+es\b", re.IGNORECASE), "PDFs"),
    (re.compile(r"\bnotebook\s*l\s*m\b", re.IGNORECASE), "NotebookLM"),
    (re.compile(r"\bnotebook\s*lm\b", re.IGNORECASE), "NotebookLM"),
    (re.compile(r"\bgamma\s*app\b", re.IGNORECASE), "Gamma App"),
    (re.compile(r"\bgama\s*app\b", re.IGNORECASE), "Gamma App"),
    (re.compile(r"\bclean\s*up\s*pictures\b", re.IGNORECASE), "Cleanup Pictures"),
    (re.compile(r"\bcleanup\s*pictures\b", re.IGNORECASE), "Cleanup Pictures"),
    (re.compile(r"\badobe\s*podcast\s*enhance\b", re.IGNORECASE), "Adobe Podcast Enhance"),
    (re.compile(r"\badobe\s*podcast\b", re.IGNORECASE), "Adobe Podcast"),
    (re.compile(r"\bresumen\s+esprecisos\b", re.IGNORECASE), "resúmenes precisos"),
    (re.compile(r"\besprecisos\b", re.IGNORECASE), "precisos"),
]


def normalize_spanish_transcript(text: str) -> str:
    """
    Automatically fix frequent Whisper transcription errors in Spanish for technical terms
    and AI tool brands (e.g., 'subepedef es' -> 'sube PDFs', 'notebook lm' -> 'NotebookLM',
    'gamma app' -> 'Gamma App', 'clean up pictures' -> 'Cleanup Pictures',
    'adobe podcast enhance' -> 'Adobe Podcast Enhance').
    """
    if not text:
        return text
    normalized = text
    for pattern, replacement in SPANISH_TRANSCRIPT_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def normalize_word_timestamps(
    words: List[WordTimestamp],
    custom_keywords: Optional[Set[str]] = None,
) -> List[WordTimestamp]:
    """
    Apply multi-word & single-word phonetic corrections across a sequence of WordTimestamps
    while preserving accurate start/end timestamps before grouping into chunks.
    """
    if not words:
        return words

    # Multi-token patterns (normalized lowercase clean tokens -> replacement tokens)
    multi_rules: List[Tuple[Tuple[str, ...], List[str]]] = [
        (("subepedef", "es"), ["sube", "PDFs"]),
        (("sube", "pedef", "es"), ["sube", "PDFs"]),
        (("sube", "pedefes"), ["sube", "PDFs"]),
        (("sube", "pdf", "es"), ["sube", "PDFs"]),
        (("pedef", "es"), ["PDFs"]),
        (("pdf", "es"), ["PDFs"]),
        (("notebook", "l", "m"), ["NotebookLM"]),
        (("notebook", "lm"), ["NotebookLM"]),
        (("gamma", "app"), ["Gamma", "App"]),
        (("gama", "app"), ["Gamma", "App"]),
        (("clean", "up", "pictures"), ["Cleanup", "Pictures"]),
        (("cleanup", "pictures"), ["Cleanup", "Pictures"]),
        (("adobe", "podcast", "enhance"), ["Adobe", "Podcast", "Enhance"]),
        (("resumen", "esprecisos"), ["resúmenes", "precisos"]),
    ]

    result: List[WordTimestamp] = []
    i = 0
    n = len(words)

    while i < n:
        matched = False
        for pat_tokens, rep_tokens in multi_rules:
            k = len(pat_tokens)
            if i + k <= n:
                window_clean = tuple(clean_token(words[i + j].word) for j in range(k))
                if window_clean == pat_tokens:
                    span_start = words[i].start
                    span_end = words[i + k - 1].end
                    # Preserve trailing punctuation from last matched word
                    last_raw = words[i + k - 1].word
                    trailing_punct = ""
                    m_punct = re.search(r"([.,!?;:]+)$", last_raw)
                    if m_punct:
                        trailing_punct = m_punct.group(1)

                    dur = max(0.08, span_end - span_start)
                    step = dur / len(rep_tokens)
                    for r_idx, r_tok in enumerate(rep_tokens):
                        tok_str = r_tok + (trailing_punct if r_idx == len(rep_tokens) - 1 else "")
                        w_s = span_start + r_idx * step
                        w_e = w_s + step
                        result.append(
                            WordTimestamp(
                                word=tok_str,
                                start=w_s,
                                end=w_e,
                                is_keyword=True,
                            )
                        )
                    i += k
                    matched = True
                    break

        if matched:
            continue

        # Single-word normalizations
        w = words[i]
        norm_w = normalize_spanish_transcript(w.word)
        if norm_w != w.word:
            parts = norm_w.split()
            dur = max(0.08, w.end - w.start)
            step = dur / len(parts)
            for p_idx, p_tok in enumerate(parts):
                result.append(
                    WordTimestamp(
                        word=p_tok,
                        start=w.start + p_idx * step,
                        end=w.start + (p_idx + 1) * step,
                        is_keyword=True,
                    )
                )
        else:
            w.is_keyword = w.is_keyword or detect_is_keyword(w.word, custom_keywords)
            result.append(w)
        i += 1

    return result


def detect_is_keyword(word: str, custom_keywords: Optional[Set[str]] = None) -> bool:
    """
    Heuristic to detect Spanish keywords:
    - Matches user-provided custom keywords, OR
    - Contains numbers/symbols (%, $), OR
    - Non-stopword with length >= 5 characters, or emphasized punctuation (!, ?)
    """
    cleaned = clean_token(word)
    if not cleaned:
        return False
    if custom_keywords and cleaned in custom_keywords:
        return True
    if any(ch.isdigit() for ch in cleaned) or ("%" in word or "$" in word):
        return True
    if cleaned in SPANISH_STOPWORDS:
        return False
    if "!" in word or "¡" in word or "?" in word or "¿" in word:
        return len(cleaned) >= 4
    return len(cleaned) >= 6


def load_modern_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a bold, modern sans-serif font available on Windows/Linux/macOS."""
    candidates = [
        # Windows modern bold fonts
        r"C:\Windows\Fonts\montserrat-bold.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",     # Segoe UI Bold
        r"C:\Windows\Fonts\arialbd.ttf",      # Arial Bold
        r"C:\Windows\Fonts\calibrib.ttf",     # Calibri Bold
        r"C:\Windows\Fonts\impact.ttf",       # Impact
        r"C:\Windows\Fonts\trebucbd.ttf",     # Trebuchet MS Bold
        r"C:\Windows\Fonts\verdanab.ttf",     # Verdana Bold
        # Linux / macOS fallbacks
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def decode_audio_16khz_mono(audio_path: str) -> np.ndarray:
    """
    Decode any audio file (.mp3, .wav, .m4a, etc.) into a 1D float32 numpy array
    at 16,000 Hz mono using FFmpeg directly.
    Bypasses PyAV (`av.open(..., metadata_errors="ignore")`) incompatibility in PyAV >= 19.
    """
    import subprocess
    import shutil

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm16 = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def transcribe_spanish_audio(
    audio_path: str,
    model_size: str = "base",
    max_words_per_chunk: int = 4,
    max_chars_per_chunk: int = 26,
    custom_keywords_raw: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[SubtitleChunk]:
    """
    Transcribe audio in Spanish using faster-whisper with word-level timestamps,
    grouping words into dynamic short chunks suitable for 9:16 Reels/TikTok.
    """
    from faster_whisper import WhisperModel

    if progress_callback:
        progress_callback(f"Cargando modelo faster-whisper '{model_size}' (CPU/INT8)...")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if progress_callback:
        progress_callback("Decodificando audio a 16kHz mono con FFmpeg y transcribiendo en español...")

    audio_waveform = decode_audio_16khz_mono(audio_path)

    segments, _ = model.transcribe(
        audio_waveform,
        language="es",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=350),
    )

    custom_kw_set: Set[str] = {
        clean_token(k) for k in re.split(r"[,;\s]+", custom_keywords_raw) if clean_token(k)
    }

    all_words: List[WordTimestamp] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                token = w.word.strip()
                if not token:
                    continue
                is_kw = detect_is_keyword(token, custom_kw_set)
                all_words.append(
                    WordTimestamp(
                        word=token,
                        start=float(w.start),
                        end=float(max(w.end, w.start + 0.08)),
                        is_keyword=is_kw,
                    )
                )
        else:
            # Fallback if segment has no word-level breakdown
            raw_tokens = normalize_spanish_transcript(seg.text.strip()).split()
            if not raw_tokens:
                continue
            seg_dur = max(0.2, float(seg.end - seg.start))
            step = seg_dur / len(raw_tokens)
            for idx, tok in enumerate(raw_tokens):
                w_start = float(seg.start) + idx * step
                w_end = w_start + step
                all_words.append(
                    WordTimestamp(
                        word=tok,
                        start=w_start,
                        end=w_end,
                        is_keyword=detect_is_keyword(tok, custom_kw_set),
                    )
                )

    all_words = normalize_word_timestamps(all_words, custom_kw_set)

    chunks = group_words_into_chunks(
        all_words,
        max_words=max_words_per_chunk,
        max_chars=max_chars_per_chunk,
    )

    # Ensure each chunk has at least 1 highlighted keyword if it has >= 2 words
    for chunk in chunks:
        if chunk.words and not any(w.is_keyword for w in chunk.words):
            longest = max(chunk.words, key=lambda item: len(clean_token(item.word)))
            if len(clean_token(longest.word)) >= 4:
                longest.is_keyword = True

    if progress_callback:
        progress_callback(f"Transcripción completada: {len(chunks)} bloques de subtítulos generados.")

    return chunks


def group_words_into_chunks(
    words: List[WordTimestamp],
    max_words: int = 4,
    max_chars: int = 26,
    pause_threshold: float = 0.45,
) -> List[SubtitleChunk]:
    """Group word timestamps into readable, punchy lower-third chunks."""
    chunks: List[SubtitleChunk] = []
    current: List[WordTimestamp] = []

    for w in words:
        if not current:
            current.append(w)
            continue

        prev = current[-1]
        gap = w.start - prev.end
        candidate_text = " ".join(item.word for item in current + [w])
        ends_sentence = prev.word.endswith((".", "!", "?", ":", ";"))

        if (
            len(current) >= max_words
            or len(candidate_text) > max_chars
            or gap >= pause_threshold
            or ends_sentence
        ):
            chunks.append(
                SubtitleChunk(
                    id=len(chunks) + 1,
                    start=current[0].start,
                    end=current[-1].end,
                    words=current,
                )
            )
            current = [w]
        else:
            current.append(w)

    if current:
        chunks.append(
            SubtitleChunk(
                id=len(chunks) + 1,
                start=current[0].start,
                end=current[-1].end,
                words=current,
            )
        )

    return chunks


def parse_edited_chunks(rows: List[dict]) -> List[SubtitleChunk]:
    """
    Convert user-edited rows from Streamlit data editor back into SubtitleChunk objects,
    applying normalize_spanish_transcript and interpolating word timestamps evenly.
    """
    parsed: List[SubtitleChunk] = []
    for idx, row in enumerate(rows):
        raw_text = str(row.get("texto", "")).strip()
        text = normalize_spanish_transcript(raw_text)
        if not text:
            continue
        start = max(0.0, float(row.get("inicio", 0.0)))
        end = max(start + 0.15, float(row.get("fin", start + 1.0)))
        kw_raw = str(row.get("palabras_clave", "")).strip()
        kw_set = {clean_token(k) for k in re.split(r"[,;\s]+", kw_raw) if clean_token(k)}

        tokens = text.split()
        dur = end - start
        step = dur / max(1, len(tokens))
        words: List[WordTimestamp] = []
        for i, tok in enumerate(tokens):
            w_start = start + i * step
            w_end = w_start + step
            cleaned = clean_token(tok)
            is_kw = (cleaned in kw_set) if kw_set else detect_is_keyword(tok)
            words.append(
                WordTimestamp(
                    word=tok,
                    start=w_start,
                    end=w_end,
                    is_keyword=is_kw,
                )
            )
        parsed.append(
            SubtitleChunk(
                id=idx + 1,
                start=start,
                end=end,
                words=words,
            )
        )
    return parsed


def render_subtitle_rgba(
    chunk: SubtitleChunk,
    current_time: float,
    frame_size: Tuple[int, int] = (1080, 1920),
    style: Optional[SubtitleStyle] = None,
) -> Image.Image:
    """
    Render a SubtitleChunk into a full-frame RGBA transparent image (1080x1920)
    with modern typography, dark stroke/shadow, and highlighted keywords in the lower third.
    """
    if style is None:
        style = SubtitleStyle()

    width, height = frame_size
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not chunk.words:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = load_modern_font(style.font_size)
    max_line_width = int(width * 0.86)  # Keep safe margins on left and right

    # Build word items with display string and color
    word_items = []
    for w in chunk.words:
        disp = w.word.upper() if style.uppercase else w.word
        is_active = w.start - 0.03 <= current_time <= w.end + 0.05

        color = style.text_color
        scale_boost = False
        if style.highlight_mode == "keywords_only":
            if w.is_keyword:
                color = style.highlight_color
                scale_boost = True
        elif style.highlight_mode == "active_word":
            if is_active:
                color = style.highlight_color
                scale_boost = True
        else:  # "keywords_and_active"
            if w.is_keyword and is_active:
                color = style.active_word_color
                scale_boost = True
            elif w.is_keyword:
                color = style.highlight_color
                scale_boost = True
            elif is_active:
                color = style.active_word_color

        word_items.append((disp, color, scale_boost))

    # Measure space width
    space_bbox = draw.textbbox((0, 0), " ", font=font)
    space_w = max(14, int((space_bbox[2] - space_bbox[0]) * 1.15))

    # Wrap words into lines
    lines: List[List[Tuple[str, Tuple[int, int, int], bool, int, int]]] = []
    current_line: List[Tuple[str, Tuple[int, int, int], bool, int, int]] = []
    current_w = 0

    for disp, color, boost in word_items:
        bbox = draw.textbbox((0, 0), disp, font=font, stroke_width=style.stroke_width)
        w_px = bbox[2] - bbox[0]
        h_px = bbox[3] - bbox[1]

        if current_line and (current_w + space_w + w_px > max_line_width):
            lines.append(current_line)
            current_line = [(disp, color, boost, w_px, h_px)]
            current_w = w_px
        else:
            if current_line:
                current_w += space_w
            current_line.append((disp, color, boost, w_px, h_px))
            current_w += w_px

    if current_line:
        lines.append(current_line)

    line_height = int(style.font_size * 1.32)
    total_block_h = len(lines) * line_height
    base_y = int(height * style.y_position_ratio) - (total_block_h // 2)

    # Optional semi-transparent dark rounded backdrop pill for readability
    if style.bg_pill and lines:
        max_seen_w = 0
        for ln in lines:
            ln_w = sum(item[3] for item in ln) + space_w * max(0, len(ln) - 1)
            max_seen_w = max(max_seen_w, ln_w)

        pad_x, pad_y = 32, 20
        box_left = (width - max_seen_w) // 2 - pad_x
        box_top = base_y - pad_y
        box_right = (width + max_seen_w) // 2 + pad_x
        box_bottom = base_y + total_block_h + pad_y
        draw.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=26,
            fill=(10, 10, 14, style.bg_opacity),
        )

    # Draw each line centered horizontally
    for line_idx, ln in enumerate(lines):
        ln_w = sum(item[3] for item in ln) + space_w * max(0, len(ln) - 1)
        cursor_x = (width - ln_w) // 2
        cursor_y = base_y + line_idx * line_height

        for disp, color, boost, w_px, _ in ln:
            # Drop shadow
            sx, sy = style.shadow_offset
            draw.text(
                (cursor_x + sx, cursor_y + sy),
                disp,
                font=font,
                fill=(0, 0, 0, 210),
                stroke_width=style.stroke_width,
                stroke_fill=(0, 0, 0, 210),
            )

            # Main text with dark stroke
            draw.text(
                (cursor_x, cursor_y),
                disp,
                font=font,
                fill=(*color, 255),
                stroke_width=style.stroke_width,
                stroke_fill=(*style.stroke_color, 255),
            )
            cursor_x += w_px + space_w

    return canvas


def composite_subtitles_on_frame(
    frame_rgb: np.ndarray,
    t: float,
    chunks: List[SubtitleChunk],
    style: SubtitleStyle,
    cache: dict,
) -> np.ndarray:
    """
    Overlay active subtitle chunk at timestamp `t` onto an RGB numpy frame (H, W, 3).
    Caches rendered RGBA overlays by (chunk_id, active_word_index) for fast rendering.
    """
    active_chunk: Optional[SubtitleChunk] = None
    for ch in chunks:
        if ch.start <= t <= ch.end:
            active_chunk = ch
            break

    if active_chunk is None:
        return frame_rgb

    # Determine active word index for cache key
    active_word_idx = -1
    if style.highlight_mode in ("active_word", "keywords_and_active"):
        for idx, w in enumerate(active_chunk.words):
            if w.start - 0.03 <= t <= w.end + 0.05:
                active_word_idx = idx
                break

    cache_key = (active_chunk.id, active_word_idx)
    if cache_key not in cache:
        # Keep cache bounded
        if len(cache) > 48:
            cache.clear()
        rgba_img = render_subtitle_rgba(
            active_chunk,
            current_time=t,
            frame_size=(frame_rgb.shape[1], frame_rgb.shape[0]),
            style=style,
        )
        rgba_arr = np.array(rgba_img, dtype=np.uint8)
        # Precompute bounding box of non-zero alpha rows to speed up alpha blending
        alpha = rgba_arr[:, :, 3]
        non_empty_rows = np.where(alpha.max(axis=1) > 0)[0]
        if len(non_empty_rows) > 0:
            r_min, r_max = int(non_empty_rows[0]), int(non_empty_rows[-1]) + 1
            sub_rgb = rgba_arr[r_min:r_max, :, :3].astype(np.float32)
            sub_a = (rgba_arr[r_min:r_max, :, 3:4].astype(np.float32)) / 255.0
            cache[cache_key] = (r_min, r_max, sub_rgb, sub_a)
        else:
            cache[cache_key] = None

    cached = cache[cache_key]
    if cached is None:
        return frame_rgb

    r_min, r_max, sub_rgb, sub_a = cached
    out = frame_rgb.copy()
    roi = out[r_min:r_max, :, :].astype(np.float32)
    blended = roi * (1.0 - sub_a) + sub_rgb * sub_a
    out[r_min:r_max, :, :] = blended.astype(np.uint8)
    return out
