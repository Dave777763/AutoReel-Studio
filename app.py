"""
AutoReel Studio — Interfaz Web en Streamlit
Ensamblador profesional de videos verticales 9:16 (1080x1920, 30fps) sin marcas de agua.
Con movimientos cinematográficos Ken Burns, transiciones crossfade de 0.3s,
subtítulos automáticos en español con faster-whisper y exportación MP4 (H.264 + AAC 192kbps).
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import mimetypes
import os
import struct
import sys
import tempfile
import time
import wave
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from subtitles import (
    SubtitleChunk,
    SubtitleStyle,
    clean_token,
    load_modern_font,
    parse_edited_chunks,
    transcribe_spanish_audio,
)
from video_engine import (
    AUDIO_EXTENSIONS,
    DEFAULT_CROSSFADE,
    IMAGE_EXTENSIONS,
    KEN_BURNS_LABELS,
    KEN_BURNS_MODES,
    SceneItem,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_EXTENSIONS,
    assemble_and_export_reel,
    compute_auto_durations,
    compute_script_synced_durations,
    detect_media_type,
    get_audio_duration,
    get_video_native_duration,
    render_preview_frame,
)

UPLOADS_DIR = os.path.join(WORKSPACE_DIR, "workspace_media", "uploads")
DEMO_DIR = os.path.join(WORKSPACE_DIR, "workspace_media", "demo")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")

for d in (UPLOADS_DIR, DEMO_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuración de página y estilos CSS modernos
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoReel Studio · 9:16 Vertical Video Builder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    .studio-header {
        background: linear-gradient(135deg, #111827 0%, #1e1b4b 55%, #311042 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.2rem;
        color: #f9fafb;
    }
    .studio-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .studio-header p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.96rem;
    }
    .badge-row {
        margin-top: 0.65rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .studio-badge {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #f8fafc;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Inicialización del estado de sesión
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "audio_path": None,
        "audio_name": None,
        "audio_duration": 0.0,
        "bgm_path": None,
        "bgm_name": None,
        "bgm_volume_pct": 12,
        "scenes": [],  # List[dict] representing SceneItem fields
        "subtitles": [],  # List[SubtitleChunk]
        "last_exported_video": None,
        "crossfade_sec": DEFAULT_CROSSFADE,
        "whisper_model": "base",
        "sub_font_size": 66,
        "sub_text_color": "#FFFFFF",
        "sub_highlight_color": "#FFD700",
        "sub_active_color": "#00F5D4",
        "sub_stroke_color": "#0C0C10",
        "sub_stroke_width": 7,
        "sub_y_ratio": 0.77,
        "sub_uppercase": True,
        "sub_bg_pill": True,
        "sub_highlight_mode": "keywords_and_active",
        "custom_keywords": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ---------------------------------------------------------------------------
# Utilidades auxiliares
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (255, 255, 255)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def get_current_subtitle_style() -> SubtitleStyle:
    return SubtitleStyle(
        font_size=int(st.session_state["sub_font_size"]),
        text_color=hex_to_rgb(st.session_state["sub_text_color"]),
        highlight_color=hex_to_rgb(st.session_state["sub_highlight_color"]),
        active_word_color=hex_to_rgb(st.session_state["sub_active_color"]),
        stroke_color=hex_to_rgb(st.session_state["sub_stroke_color"]),
        stroke_width=int(st.session_state["sub_stroke_width"]),
        y_position_ratio=float(st.session_state["sub_y_ratio"]),
        uppercase=bool(st.session_state["sub_uppercase"]),
        highlight_mode=str(st.session_state["sub_highlight_mode"]),
        bg_pill=bool(st.session_state["sub_bg_pill"]),
    )


def dicts_to_scene_items(scene_dicts: List[dict]) -> List[SceneItem]:
    items: List[SceneItem] = []
    for idx, d in enumerate(scene_dicts):
        items.append(
            SceneItem(
                id=idx + 1,
                path=str(d["path"]),
                name=str(d["name"]),
                media_type=str(d["media_type"]),
                duration=max(0.4, float(d["duration"])),
                effect=str(d.get("effect", "auto")),
                fit_mode=str(d.get("fit_mode", "cover")),
                native_duration=d.get("native_duration"),
            )
        )
    return items


def sync_scene_durations_to_audio():
    """Recalcula las duraciones de todas las escenas para cubrir exactamente la duración del audio."""
    scenes = st.session_state["scenes"]
    audio_dur = float(st.session_state["audio_duration"] or 0.0)
    if not scenes:
        return
    target_total = audio_dur if audio_dur > 0 else len(scenes) * 3.0
    auto_durs = compute_auto_durations(len(scenes), target_total)
    for sc, dur in zip(scenes, auto_durs):
        sc["duration"] = dur


def create_demo_project() -> Tuple[str, List[str]]:
    """
    Genera un proyecto de demostración local (audio en español + 4 imágenes verticales de alta resolución)
    para probar el flujo completo de AutoReel Studio con un clic.
    """
    audio_file = os.path.join(DEMO_DIR, "locucion_demo_es.mp3")
    demo_text = (
        "Bienvenido a AutoReel Studio. Crea videos verticales increíbles con movimiento cinematográfico "
        "y subtítulos dinámicos automáticos listos para redes sociales."
    )

    # Intentar generar locución real en español con edge-tts; si no hay red, generar WAV sintético
    generated_tts = False
    try:
        import edge_tts

        async def _gen_tts():
            communicate = edge_tts.Communicate(demo_text, voice="es-MX-DaliaNeural")
            await communicate.save(audio_file)

        asyncio.run(_gen_tts())
        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
            generated_tts = True
    except Exception:
        generated_tts = False

    if not generated_tts:
        audio_file = os.path.join(DEMO_DIR, "audio_demo_es.wav")
        sr = 44100
        dur = 10.0
        n_samples = int(sr * dur)
        t = np.linspace(0, dur, n_samples, endpoint=False)
        # Acorde ambiental suave
        sig = (
            0.25 * np.sin(2 * np.pi * 220.0 * t)
            + 0.20 * np.sin(2 * np.pi * 277.18 * t)
            + 0.20 * np.sin(2 * np.pi * 329.63 * t)
        )
        envelope = np.minimum(1.0, t / 0.5) * np.minimum(1.0, (dur - t) / 0.8)
        pcm = (sig * envelope * 32767 * 0.6).astype(np.int16)
        with wave.open(audio_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    # Crear 4 imágenes verticales 1080x1920 con diseños modernos
    palettes = [
        ((15, 23, 42), (56, 189, 248), "ESCENA 01 · IMPACTO VISUAL", "Zoom In 1.00 → 1.12"),
        ((30, 27, 75), (244, 114, 182), "ESCENA 02 · RITMO DINÁMICO", "Zoom Out 1.12 → 1.00"),
        ((20, 83, 45), (74, 222, 128), "ESCENA 03 · PANEO CINEMÁTICO", "Paneo Sutil + Crossfade 0.3s"),
        ((67, 20, 7), (251, 191, 36), "ESCENA 04 · EXPORTACIÓN 9:16", "1080x1920 · 30 FPS · H.264"),
    ]
    image_paths: List[str] = []
    title_font = load_modern_font(58)
    sub_font = load_modern_font(38)

    for idx, (bg_top, accent, title, subtitle) in enumerate(palettes, start=1):
        img_path = os.path.join(DEMO_DIR, f"escena_demo_{idx:02d}.png")
        img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), bg_top)
        draw = ImageDraw.Draw(img)

        # Fondo geométrico con círculos y cuadrícula para apreciar el efecto Ken Burns
        for step_y in range(0, TARGET_HEIGHT, 120):
            draw.line([(0, step_y), (TARGET_WIDTH, step_y)], fill=(255, 255, 255, 22), width=2)
        for step_x in range(0, TARGET_WIDTH, 120):
            draw.line([(step_x, 0), (step_x, TARGET_HEIGHT)], fill=(255, 255, 255, 22), width=2)

        cx, cy = TARGET_WIDTH // 2, int(TARGET_HEIGHT * 0.42)
        for r in (420, 310, 200):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent, width=6)

        draw.rounded_rectangle(
            [90, cy - 110, TARGET_WIDTH - 90, cy + 110],
            radius=28,
            fill=(10, 12, 20),
            outline=accent,
            width=4,
        )
        tb = draw.textbbox((0, 0), title, font=title_font)
        tw = tb[2] - tb[0]
        draw.text(((TARGET_WIDTH - tw) // 2, cy - 65), title, font=title_font, fill=(255, 255, 255))

        sb = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = sb[2] - sb[0]
        draw.text(((TARGET_WIDTH - sw) // 2, cy + 15), subtitle, font=sub_font, fill=accent)

        img.save(img_path, "PNG")
        image_paths.append(img_path)

    return audio_file, image_paths


def load_media_into_session(audio_path: Optional[str], media_paths: List[str], append_scenes: bool = False):
    """Carga el archivo de audio y la lista de clips/imágenes en el estado de la sesión."""
    if audio_path and os.path.exists(audio_path):
        st.session_state["audio_path"] = audio_path
        st.session_state["audio_name"] = os.path.basename(audio_path)
        st.session_state["audio_duration"] = get_audio_duration(audio_path)

    existing = list(st.session_state["scenes"]) if append_scenes else []
    for p in media_paths:
        if not os.path.exists(p):
            continue
        mtype = detect_media_type(p)
        if mtype not in ("image", "video"):
            continue
        native_dur = get_video_native_duration(p) if mtype == "video" else None
        idx = len(existing)
        auto_mode = KEN_BURNS_MODES[idx % len(KEN_BURNS_MODES)] if mtype == "image" else "none"
        existing.append(
            {
                "orden": idx + 1,
                "name": os.path.basename(p),
                "path": os.path.abspath(p),
                "media_type": mtype,
                "duration": 3.0,
                "effect": auto_mode if mtype == "image" else "none",
                "fit_mode": "cover",
                "native_duration": native_dur,
            }
        )

    st.session_state["scenes"] = existing
    if existing:
        sync_scene_durations_to_audio()


# ---------------------------------------------------------------------------
# Componente Interactivo de Línea de Tiempo (Audio + Iconos Arrastrables)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def _encode_scene_thumbnail(path: str, media_type: str, mtime: float) -> str:
    """Genera una miniatura JPEG en base64 (88x120) para los iconos superiores de la línea de tiempo."""
    try:
        if media_type == "image" and os.path.exists(path):
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            thumb = ImageOps.fit(img, (88, 120), method=Image.Resampling.BILINEAR)
        elif media_type == "video" and os.path.exists(path):
            from moviepy import VideoFileClip

            clip = VideoFileClip(path)
            t_sample = min(0.5, max(0.0, float(clip.duration or 0.0) * 0.2))
            frame = clip.get_frame(t_sample)
            clip.close()
            img = Image.fromarray(frame).convert("RGB")
            thumb = ImageOps.fit(img, (88, 120), method=Image.Resampling.BILINEAR)
        else:
            thumb = Image.new("RGB", (88, 120), (30, 41, 59))
    except Exception:
        thumb = Image.new("RGB", (88, 120), (30, 41, 59))

    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=76)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@st.cache_data(show_spinner=False, max_entries=16)
def _encode_audio_data_uri(audio_path: str, mtime: float) -> Optional[str]:
    """Codifica el archivo de audio como Data URI para reproducción HTML5 y onda WebAudio."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        mime, _ = mimetypes.guess_type(audio_path)
        if not mime:
            ext = os.path.splitext(audio_path)[1].lower()
            mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
        with open(audio_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


TIMELINE_CSS = """
:host {
    display: block;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #f8fafc;
    user-select: none;
    -webkit-user-select: none;
}
.tl-card {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 16px;
    padding: 16px 18px 18px 18px;
    box-shadow: 0 12px 28px -8px rgba(0, 0, 0, 0.5);
}
.tl-top-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
}
.tl-controls-left {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.tl-play-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: #ffffff;
    border: none;
    border-radius: 999px;
    padding: 8px 18px;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
}
.tl-play-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.55);
}
.tl-play-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}
.tl-stop-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(30, 41, 59, 0.85);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
}
.tl-stop-btn:hover {
    background: rgba(51, 65, 85, 0.9);
    color: #ffffff;
}
.tl-time-badge {
    font-variant-numeric: tabular-nums;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38bdf8;
    padding: 6px 13px;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 700;
}
.tl-time-badge span.total {
    color: #94a3b8;
    font-weight: 500;
}
.tl-active-scene-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 5px 12px;
    font-size: 0.82rem;
    color: #e2e8f0;
}
.tl-active-scene-pill strong {
    color: #facc15;
}
.tl-hint {
    font-size: 0.78rem;
    color: #94a3b8;
}
.tl-body {
    display: grid;
    grid-template-columns: 1fr 135px;
    gap: 16px;
    align-items: stretch;
}
@media (max-width: 768px) {
    .tl-body { grid-template-columns: 1fr; }
}
.tl-preview-box {
    background: #090d16;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    text-align: center;
}
.tl-preview-img-wrap {
    width: 100%;
    aspect-ratio: 9 / 13;
    border-radius: 8px;
    overflow: hidden;
    background: #1e293b;
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.tl-preview-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.tl-preview-badge {
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(15, 23, 42, 0.85);
    color: #38bdf8;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 6px;
    border: 1px solid rgba(56, 189, 248, 0.4);
}
.tl-preview-caption {
    margin-top: 6px;
    font-size: 0.72rem;
    color: #cbd5e1;
    font-weight: 600;
    max-width: 120px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tl-preview-sub {
    font-size: 0.68rem;
    color: #94a3b8;
    font-variant-numeric: tabular-nums;
}
.tl-stack {
    position: relative;
    background: #090d16;
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 12px;
    padding: 12px 18px 14px 18px;
}
.tl-section-label {
    font-size: 0.71rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tl-pins-lane {
    position: relative;
    height: 92px;
    margin-bottom: 4px;
    border-bottom: 1px dashed rgba(148, 163, 184, 0.2);
}
.tl-pin {
    position: absolute;
    top: 4px;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: grab;
    z-index: 15;
    touch-action: none;
}
.tl-pin.locked { cursor: pointer; }
.tl-pin:hover { z-index: 25; filter: brightness(1.1); }
.tl-pin.dragging {
    cursor: grabbing;
    z-index: 40;
    filter: drop-shadow(0 8px 16px rgba(56, 189, 248, 0.55));
}
.tl-pin-card {
    width: 54px;
    height: 64px;
    border-radius: 10px;
    background: #1e293b;
    border: 2px solid #38bdf8;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.45);
    display: flex;
    flex-direction: column;
}
.tl-pin.active .tl-pin-card {
    border-color: #facc15;
    box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.35), 0 6px 14px rgba(0, 0, 0, 0.5);
}
.tl-pin-card img {
    width: 100%;
    height: 44px;
    object-fit: cover;
    display: block;
    pointer-events: none;
}
.tl-pin-num {
    position: absolute;
    top: 3px;
    left: 3px;
    background: rgba(15, 23, 42, 0.88);
    color: #f8fafc;
    font-size: 0.65rem;
    font-weight: 800;
    padding: 1px 5px;
    border-radius: 5px;
}
.tl-pin-grip {
    position: absolute;
    top: 3px;
    right: 3px;
    background: rgba(15, 23, 42, 0.82);
    color: #38bdf8;
    font-size: 0.62rem;
    padding: 1px 4px;
    border-radius: 4px;
}
.tl-pin-time {
    height: 18px;
    background: rgba(15, 23, 42, 0.96);
    color: #e2e8f0;
    font-size: 0.65rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    display: flex;
    align-items: center;
    justify-content: center;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
}
.tl-pin.dragging .tl-pin-time {
    background: #0284c7;
    color: #ffffff;
}
.tl-pin-pointer {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 7px solid #38bdf8;
    margin-top: -1px;
}
.tl-pin.active .tl-pin-pointer { border-top-color: #facc15; }
.tl-pin-stem {
    width: 2px;
    height: 14px;
    background: linear-gradient(180deg, #38bdf8 0%, rgba(56, 189, 248, 0.2) 100%);
}
.tl-pin.active .tl-pin-stem {
    background: linear-gradient(180deg, #facc15 0%, rgba(250, 204, 21, 0.3) 100%);
}
.tl-scenes-bar {
    position: relative;
    height: 34px;
    background: #111827;
    border-radius: 7px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 8px;
}
.tl-scene-seg {
    position: absolute;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 8px;
    box-sizing: border-box;
    border-right: 2px solid rgba(15, 23, 42, 0.9);
    font-size: 0.72rem;
    font-weight: 700;
    color: #f8fafc;
    overflow: hidden;
    white-space: nowrap;
    cursor: pointer;
}
.tl-scene-seg:hover { filter: brightness(1.15); }
.tl-scene-seg.active { box-shadow: inset 0 0 0 2px #facc15; }
.tl-scene-seg .seg-title { overflow: hidden; text-overflow: ellipsis; }
.tl-scene-seg .seg-dur {
    font-size: 0.66rem;
    opacity: 0.85;
    font-weight: 600;
    margin-left: 4px;
    font-variant-numeric: tabular-nums;
}
.tl-wave-wrap {
    position: relative;
    height: 68px;
    background: linear-gradient(180deg, #0b1324 0%, #0f172a 100%);
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.22);
    overflow: hidden;
    cursor: pointer;
    margin-bottom: 6px;
}
.tl-wave-canvas { width: 100%; height: 100%; display: block; }
.tl-cut-line {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1.5px;
    background: rgba(56, 189, 248, 0.55);
    pointer-events: none;
}
.tl-subs-bar {
    position: relative;
    height: 22px;
    background: rgba(15, 23, 42, 0.75);
    border-radius: 5px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    overflow: hidden;
    margin-bottom: 6px;
}
.tl-sub-chip {
    position: absolute;
    top: 2px;
    bottom: 2px;
    background: rgba(250, 204, 21, 0.16);
    border-left: 2px solid #facc15;
    border-radius: 3px;
    color: #fef08a;
    font-size: 0.62rem;
    padding: 0 4px;
    display: flex;
    align-items: center;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    cursor: pointer;
}
.tl-ruler {
    position: relative;
    height: 20px;
    font-size: 0.67rem;
    color: #94a3b8;
    font-variant-numeric: tabular-nums;
}
.tl-tick {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.tl-tick-mark {
    width: 1px;
    height: 5px;
    background: rgba(148, 163, 184, 0.45);
    margin-bottom: 2px;
}
.tl-playhead {
    position: absolute;
    top: 28px;
    bottom: 26px;
    width: 2px;
    background: #f43f5e;
    box-shadow: 0 0 8px rgba(244, 63, 94, 0.9);
    pointer-events: none;
    z-index: 30;
    transform: translateX(-50%);
}
.tl-playhead::before {
    content: '';
    position: absolute;
    top: -5px;
    left: -5px;
    width: 12px;
    height: 12px;
    background: #f43f5e;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(244, 63, 94, 1);
}
"""

TIMELINE_JS = """
export default function(component) {
    const { data, setStateValue, setTriggerValue, parentElement } = component;
    if (!data || !data.scenes || data.scenes.length === 0) {
        parentElement.innerHTML = '<div style="padding:12px;color:#94a3b8;">Sin escenas en la línea de tiempo.</div>';
        return;
    }

    const SEGMENT_COLORS = [
        'linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)',
        'linear-gradient(135deg, #6d28d9 0%, #7c3aed 100%)',
        'linear-gradient(135deg, #047857 0%, #059669 100%)',
        'linear-gradient(135deg, #b45309 0%, #d97706 100%)',
        'linear-gradient(135deg, #be185d 0%, #db2777 100%)',
        'linear-gradient(135deg, #0e7490 0%, #0891b2 100%)',
        'linear-gradient(135deg, #4338ca 0%, #4f46e5 100%)',
    ];

    const MIN_SCENE_DUR = 0.4;
    const totalDuration = Math.max(1.0, Number(data.total_duration || 10.0));
    const subtitles = Array.isArray(data.subtitles) ? data.subtitles : [];

    let scenes = data.scenes.map((sc, i) => ({
        idx: i,
        name: sc.name,
        media_type: sc.media_type,
        duration: Math.max(MIN_SCENE_DUR, Number(sc.duration || 3.0)),
        thumb: sc.thumb,
        effect_label: sc.effect_label || '',
        start: 0,
        end: 0,
    }));

    function recomputeStartsFromDurations() {
        let cum = 0.0;
        for (let i = 0; i < scenes.length; i++) {
            scenes[i].start = Number(cum.toFixed(2));
            cum += scenes[i].duration;
            scenes[i].end = Number(cum.toFixed(2));
        }
    }
    recomputeStartsFromDurations();

    const styleEl = document.createElement('style');
    styleEl.textContent = data.css || '';

    const root = document.createElement('div');
    root.className = 'tl-card';
    root.innerHTML = `
        <div class="tl-top-bar">
            <div class="tl-controls-left">
                <button class="tl-play-btn" id="tlPlayBtn" type="button">
                    <span id="tlPlayIcon">▶</span>
                    <span id="tlPlayText">Escuchar Audio</span>
                </button>
                <button class="tl-stop-btn" id="tlRestartBtn" type="button" title="Volver al inicio (0.00s)">
                    ⏮ 0.00s
                </button>
                <div class="tl-time-badge">
                    <span id="tlCurrTime">00.00s</span>
                    <span class="total"> / ${totalDuration.toFixed(2)}s</span>
                </div>
                <div class="tl-active-scene-pill" id="tlActivePill">
                    Escena activa: <strong id="tlActiveSceneLabel">#1</strong>
                </div>
            </div>
            <div class="tl-hint">
                <span>💡 Arrastra los iconos de arriba hacia la izquierda o derecha para elegir dónde empieza cada escena. Haz clic en la onda para saltar a ese segundo.</span>
            </div>
        </div>

        <div class="tl-body">
            <div class="tl-stack" id="tlStack">
                <div class="tl-section-label">
                    <span>📍 Iconos de Inicio de Escena (Arrastra para mover el punto de entrada)</span>
                    <span id="tlDragStatus" style="color:#38bdf8;"></span>
                </div>
                <div class="tl-pins-lane" id="tlPinsLane"></div>

                <div class="tl-section-label">
                    <span>🎬 Secuencia de Escenas</span>
                </div>
                <div class="tl-scenes-bar" id="tlScenesBar"></div>

                <div class="tl-section-label">
                    <span>🎵 Forma de Onda del Audio (${data.audio_name || 'Sin audio'})</span>
                </div>
                <div class="tl-wave-wrap" id="tlWaveWrap">
                    <canvas class="tl-wave-canvas" id="tlWaveCanvas"></canvas>
                    <div id="tlWaveCuts"></div>
                </div>

                ${subtitles.length > 0 ? `
                <div class="tl-section-label">
                    <span>💬 Subtítulos Sincronizados (clic para escuchar frase)</span>
                </div>
                <div class="tl-subs-bar" id="tlSubsBar"></div>
                ` : ''}

                <div class="tl-ruler" id="tlRuler"></div>
                <div class="tl-playhead" id="tlPlayhead" style="left: 18px;"></div>
            </div>

            <div class="tl-preview-box">
                <div class="tl-preview-img-wrap">
                    <img id="tlPreviewImg" src="${scenes[0].thumb}" alt="Preview" />
                    <div class="tl-preview-badge" id="tlPreviewBadge">#1</div>
                </div>
                <div>
                    <div class="tl-preview-caption" id="tlPreviewName">${scenes[0].name}</div>
                    <div class="tl-preview-sub" id="tlPreviewRange">${scenes[0].start.toFixed(2)}s → ${scenes[0].end.toFixed(2)}s</div>
                </div>
            </div>
        </div>
    `;
    parentElement.innerHTML = '';
    parentElement.appendChild(styleEl);
    parentElement.appendChild(root);

    const playBtn = root.querySelector('#tlPlayBtn');
    const playIcon = root.querySelector('#tlPlayIcon');
    const playText = root.querySelector('#tlPlayText');
    const restartBtn = root.querySelector('#tlRestartBtn');
    const currTimeEl = root.querySelector('#tlCurrTime');
    const activeSceneLabel = root.querySelector('#tlActiveSceneLabel');
    const dragStatusEl = root.querySelector('#tlDragStatus');
    const pinsLane = root.querySelector('#tlPinsLane');
    const scenesBar = root.querySelector('#tlScenesBar');
    const waveWrap = root.querySelector('#tlWaveWrap');
    const waveCanvas = root.querySelector('#tlWaveCanvas');
    const waveCuts = root.querySelector('#tlWaveCuts');
    const subsBar = root.querySelector('#tlSubsBar');
    const rulerEl = root.querySelector('#tlRuler');
    const playheadEl = root.querySelector('#tlPlayhead');
    const previewImg = root.querySelector('#tlPreviewImg');
    const previewBadge = root.querySelector('#tlPreviewBadge');
    const previewName = root.querySelector('#tlPreviewName');
    const previewRange = root.querySelector('#tlPreviewRange');

    const audio = new Audio();
    if (data.audio_data_uri) {
        audio.src = data.audio_data_uri;
        audio.preload = 'auto';
    } else {
        playBtn.disabled = true;
        playText.textContent = 'Sin Audio';
    }

    let currentTime = 0.0;
    let rafId = null;
    let peaks = null;

    function timeToPct(t) {
        return Math.max(0, Math.min(100, (t / totalDuration) * 100));
    }

    function getActiveSceneIndex(t) {
        for (let i = scenes.length - 1; i >= 0; i--) {
            if (t >= scenes[i].start - 0.005) {
                return i;
            }
        }
        return 0;
    }

    function updatePlayheadUI(t) {
        currentTime = Math.max(0, Math.min(totalDuration, t));
        currTimeEl.textContent = currentTime.toFixed(2).padStart(5, '0') + 's';

        const trackRect = waveWrap.getBoundingClientRect();
        const stackRect = root.querySelector('#tlStack').getBoundingClientRect();
        if (trackRect.width > 0) {
            const offsetLeft = (trackRect.left - stackRect.left) + (currentTime / totalDuration) * trackRect.width;
            playheadEl.style.left = `${offsetLeft}px`;
        }

        const activeIdx = getActiveSceneIndex(currentTime);
        const activeSc = scenes[activeIdx];
        if (activeSc) {
            activeSceneLabel.textContent = `#${activeIdx + 1} · ${activeSc.name}`;
            if (previewImg.getAttribute('src') !== activeSc.thumb) {
                previewImg.setAttribute('src', activeSc.thumb);
            }
            previewBadge.textContent = `#${activeIdx + 1}`;
            previewName.textContent = activeSc.name;
            previewRange.textContent = `${activeSc.start.toFixed(2)}s → ${activeSc.end.toFixed(2)}s`;
        }

        const allPins = pinsLane.querySelectorAll('.tl-pin');
        allPins.forEach((p, idx) => p.classList.toggle('active', idx === activeIdx));
        const allSegs = scenesBar.querySelectorAll('.tl-scene-seg');
        allSegs.forEach((s, idx) => s.classList.toggle('active', idx === activeIdx));

        drawWaveform();
    }

    function seekTo(t) {
        const clamped = Math.max(0, Math.min(totalDuration, t));
        if (data.audio_data_uri) {
            try { audio.currentTime = clamped; } catch (e) {}
        }
        updatePlayheadUI(clamped);
    }

    function renderRuler() {
        rulerEl.innerHTML = '';
        const step = totalDuration <= 12 ? 1 : totalDuration <= 30 ? 2 : totalDuration <= 60 ? 5 : 10;
        for (let s = 0; s <= totalDuration + 0.01; s += step) {
            const tick = document.createElement('div');
            tick.className = 'tl-tick';
            tick.style.left = `${timeToPct(s)}%`;
            tick.innerHTML = `<div class="tl-tick-mark"></div><span>${s.toFixed(0)}s</span>`;
            rulerEl.appendChild(tick);
        }
    }

    function renderSubtitles() {
        if (!subsBar || subtitles.length === 0) return;
        subsBar.innerHTML = '';
        subtitles.forEach((sub) => {
            const chip = document.createElement('div');
            chip.className = 'tl-sub-chip';
            const leftPct = timeToPct(sub.start);
            const widthPct = Math.max(1.0, timeToPct(sub.end) - leftPct);
            chip.style.left = `${leftPct}%`;
            chip.style.width = `${widthPct}%`;
            chip.title = `[${sub.start.toFixed(2)}s → ${sub.end.toFixed(2)}s] ${sub.text}`;
            chip.textContent = sub.text;
            chip.addEventListener('click', (e) => {
                e.stopPropagation();
                seekTo(sub.start);
            });
            subsBar.appendChild(chip);
        });
    }

    function renderScenesAndPins() {
        pinsLane.innerHTML = '';
        scenesBar.innerHTML = '';
        waveCuts.innerHTML = '';

        scenes.forEach((sc, idx) => {
            const leftPct = timeToPct(sc.start);
            const widthPct = Math.max(0.5, timeToPct(sc.end) - leftPct);

            const pin = document.createElement('div');
            pin.className = 'tl-pin' + (idx === 0 ? ' locked' : '');
            pin.style.left = `${leftPct}%`;
            pin.dataset.idx = String(idx);
            pin.title = idx === 0
                ? `Escena #1 (${sc.name}) inicia en 0.00s (Clic para escuchar desde el inicio)`
                : `Arrastra para mover dónde empieza la Escena #${idx + 1} (${sc.name})`;

            pin.innerHTML = `
                <div class="tl-pin-card">
                    <img src="${sc.thumb}" alt="#${idx + 1}" />
                    <span class="tl-pin-num">#${idx + 1}</span>
                    ${idx > 0 ? '<span class="tl-pin-grip">⇔</span>' : '<span class="tl-pin-grip">🔒</span>'}
                    <div class="tl-pin-time" data-role="time">${sc.start.toFixed(2)}s</div>
                </div>
                <div class="tl-pin-pointer"></div>
                <div class="tl-pin-stem"></div>
            `;

            if (idx === 0) {
                pin.addEventListener('click', () => seekTo(0));
            } else {
                attachPinDrag(pin, idx);
            }
            pinsLane.appendChild(pin);

            const seg = document.createElement('div');
            seg.className = 'tl-scene-seg';
            seg.style.left = `${leftPct}%`;
            seg.style.width = `${widthPct}%`;
            seg.style.background = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
            seg.title = `Escena #${idx + 1}: ${sc.name} (${sc.start.toFixed(2)}s → ${sc.end.toFixed(2)}s · Duración: ${sc.duration.toFixed(2)}s) — Clic para escuchar desde aquí`;
            seg.innerHTML = `
                <span class="seg-title">#${idx + 1} ${sc.name}</span>
                <span class="seg-dur">${sc.duration.toFixed(2)}s</span>
            `;
            seg.addEventListener('click', () => seekTo(sc.start));
            scenesBar.appendChild(seg);

            if (idx > 0) {
                const cutLine = document.createElement('div');
                cutLine.className = 'tl-cut-line';
                cutLine.style.left = `${leftPct}%`;
                waveCuts.appendChild(cutLine);
            }
        });

        updatePlayheadUI(currentTime);
    }

    function commitDurationsToPython() {
        const newDurations = scenes.map((s) => Number(s.duration.toFixed(2)));
        setStateValue('durations', newDurations);
        setTriggerValue('changed', {
            durations: newDurations,
            ts: Date.now(),
        });
    }

    function attachPinDrag(pinEl, idx) {
        let isDragging = false;
        let hasMoved = false;

        const onPointerDown = (e) => {
            if (e.button !== undefined && e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            isDragging = true;
            hasMoved = false;
            pinEl.classList.add('dragging');
            if (pinEl.setPointerCapture && e.pointerId !== undefined) {
                try { pinEl.setPointerCapture(e.pointerId); } catch (err) {}
            }
        };

        const onPointerMove = (e) => {
            if (!isDragging) return;
            hasMoved = true;
            const rect = scenesBar.getBoundingClientRect();
            if (rect.width <= 0) return;

            const relX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            const rawTime = (relX / rect.width) * totalDuration;

            const prevStart = scenes[idx - 1].start;
            const nextEnd = scenes[idx].end;
            const minAllowed = prevStart + MIN_SCENE_DUR;
            const maxAllowed = nextEnd - MIN_SCENE_DUR;

            const clampedStart = Number(Math.max(minAllowed, Math.min(maxAllowed, rawTime)).toFixed(2));

            scenes[idx - 1].end = clampedStart;
            scenes[idx - 1].duration = Number((clampedStart - prevStart).toFixed(2));
            scenes[idx].start = clampedStart;
            scenes[idx].duration = Number((nextEnd - clampedStart).toFixed(2));

            dragStatusEl.textContent = `Moviendo inicio de Escena #${idx + 1} → ${clampedStart.toFixed(2)}s (Escena #${idx}: ${scenes[idx - 1].duration.toFixed(2)}s | Escena #${idx + 1}: ${scenes[idx].duration.toFixed(2)}s)`;

            const leftPct = timeToPct(clampedStart);
            pinEl.style.left = `${leftPct}%`;
            const timeLabel = pinEl.querySelector('[data-role="time"]');
            if (timeLabel) timeLabel.textContent = `${clampedStart.toFixed(2)}s`;

            const allSegs = scenesBar.querySelectorAll('.tl-scene-seg');
            if (allSegs[idx - 1]) {
                const prevLeft = timeToPct(scenes[idx - 1].start);
                allSegs[idx - 1].style.width = `${Math.max(0.5, leftPct - prevLeft)}%`;
                const dEl = allSegs[idx - 1].querySelector('.seg-dur');
                if (dEl) dEl.textContent = `${scenes[idx - 1].duration.toFixed(2)}s`;
            }
            if (allSegs[idx]) {
                const nextRight = timeToPct(scenes[idx].end);
                allSegs[idx].style.left = `${leftPct}%`;
                allSegs[idx].style.width = `${Math.max(0.5, nextRight - leftPct)}%`;
                const dEl = allSegs[idx].querySelector('.seg-dur');
                if (dEl) dEl.textContent = `${scenes[idx].duration.toFixed(2)}s`;
            }

            const allCuts = waveCuts.querySelectorAll('.tl-cut-line');
            if (allCuts[idx - 1]) {
                allCuts[idx - 1].style.left = `${leftPct}%`;
            }

            seekTo(clampedStart);
        };

        const onPointerUp = () => {
            if (!isDragging) return;
            isDragging = false;
            pinEl.classList.remove('dragging');
            dragStatusEl.textContent = '';
            if (hasMoved) {
                commitDurationsToPython();
            } else {
                seekTo(scenes[idx].start);
            }
        };

        pinEl.addEventListener('pointerdown', onPointerDown);
        pinEl.addEventListener('pointermove', onPointerMove);
        pinEl.addEventListener('pointerup', onPointerUp);
        pinEl.addEventListener('pointercancel', onPointerUp);
    }

    let isScrubbingWave = false;
    waveWrap.addEventListener('pointerdown', (e) => {
        isScrubbingWave = true;
        if (waveWrap.setPointerCapture && e.pointerId !== undefined) {
            try { waveWrap.setPointerCapture(e.pointerId); } catch (err) {}
        }
        const rect = waveWrap.getBoundingClientRect();
        if (rect.width > 0) {
            const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            seekTo(ratio * totalDuration);
        }
    });
    waveWrap.addEventListener('pointermove', (e) => {
        if (!isScrubbingWave) return;
        const rect = waveWrap.getBoundingClientRect();
        if (rect.width > 0) {
            const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            seekTo(ratio * totalDuration);
        }
    });
    waveWrap.addEventListener('pointerup', () => { isScrubbingWave = false; });
    waveWrap.addEventListener('pointercancel', () => { isScrubbingWave = false; });

    async function decodeWaveform() {
        if (!data.audio_data_uri) return;
        try {
            const resp = await fetch(data.audio_data_uri);
            const arrayBuf = await resp.arrayBuffer();
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioCtx();
            const audioBuf = await ctx.decodeAudioData(arrayBuf);
            const rawData = audioBuf.getChannelData(0);
            const numBars = 240;
            const blockSize = Math.max(1, Math.floor(rawData.length / numBars));
            const computed = new Float32Array(numBars);
            let maxPeak = 0.01;
            for (let i = 0; i < numBars; i++) {
                let sum = 0;
                const start = i * blockSize;
                for (let j = 0; j < blockSize; j++) {
                    const v = Math.abs(rawData[start + j] || 0);
                    if (v > sum) sum = v;
                }
                computed[i] = sum;
                if (sum > maxPeak) maxPeak = sum;
            }
            for (let i = 0; i < numBars; i++) {
                computed[i] = Math.max(0.06, computed[i] / maxPeak);
            }
            peaks = computed;
            ctx.close();
            drawWaveform();
        } catch (err) {
            const numBars = 180;
            peaks = new Float32Array(numBars);
            for (let i = 0; i < numBars; i++) {
                peaks[i] = 0.25 + 0.55 * Math.abs(Math.sin(i * 0.23) * Math.cos(i * 0.07));
            }
            drawWaveform();
        }
    }

    function drawWaveform() {
        const dpr = window.devicePixelRatio || 1;
        const rect = waveCanvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        const w = Math.floor(rect.width * dpr);
        const h = Math.floor(rect.height * dpr);
        if (waveCanvas.width !== w || waveCanvas.height !== h) {
            waveCanvas.width = w;
            waveCanvas.height = h;
        }

        const ctx = waveCanvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        const arr = peaks;
        if (!arr || arr.length === 0) return;

        const n = arr.length;
        const barStep = w / n;
        const barW = Math.max(1.5 * dpr, barStep * 0.66);
        const midY = h / 2;
        const progressRatio = currentTime / totalDuration;

        for (let i = 0; i < n; i++) {
            const barRatio = i / n;
            const amp = arr[i];
            const barH = Math.max(3 * dpr, amp * (h * 0.84));
            const x = i * barStep + (barStep - barW) * 0.5;
            const y = midY - barH / 2;

            ctx.fillStyle = barRatio <= progressRatio ? '#38bdf8' : 'rgba(148, 163, 184, 0.35)';
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(x, y, barW, barH, 2 * dpr);
            } else {
                ctx.rect(x, y, barW, barH);
            }
            ctx.fill();
        }
    }

    function tickPlayhead() {
        if (!audio.paused) {
            updatePlayheadUI(audio.currentTime);
            rafId = requestAnimationFrame(tickPlayhead);
        }
    }

    playBtn.addEventListener('click', () => {
        if (!data.audio_data_uri) return;
        if (audio.paused) {
            if (audio.currentTime >= totalDuration - 0.05) {
                audio.currentTime = 0;
            }
            audio.play().then(() => {
                playIcon.textContent = '⏸';
                playText.textContent = 'Pausar Audio';
                rafId = requestAnimationFrame(tickPlayhead);
            }).catch(() => {});
        } else {
            audio.pause();
            playIcon.textContent = '▶';
            playText.textContent = 'Escuchar Audio';
            if (rafId) cancelAnimationFrame(rafId);
        }
    });

    restartBtn.addEventListener('click', () => seekTo(0));

    audio.addEventListener('pause', () => {
        playIcon.textContent = '▶';
        playText.textContent = 'Escuchar Audio';
        if (rafId) cancelAnimationFrame(rafId);
    });

    audio.addEventListener('ended', () => {
        playIcon.textContent = '▶';
        playText.textContent = 'Escuchar Audio';
        if (rafId) cancelAnimationFrame(rafId);
        updatePlayheadUI(totalDuration);
    });

    renderRuler();
    renderSubtitles();
    renderScenesAndPins();
    decodeWaveform();

    const resizeObserver = new ResizeObserver(() => updatePlayheadUI(currentTime));
    resizeObserver.observe(waveWrap);

    return () => {
        if (rafId) cancelAnimationFrame(rafId);
        try {
            audio.pause();
            audio.src = '';
        } catch (e) {}
        resizeObserver.disconnect();
    };
}
"""

_timeline_v2_renderer = st.components.v2.component(
    "autoreel_interactive_timeline",
    css=TIMELINE_CSS,
    js=TIMELINE_JS,
)


def render_interactive_scene_timeline(
    scenes: List[Dict[str, Any]],
    audio_path: Optional[str],
    audio_name: Optional[str],
    audio_duration: float,
    subtitles: Optional[List[Any]] = None,
    key: str = "interactive_scene_timeline",
) -> Optional[List[float]]:
    """
    Renderiza la línea de tiempo interactiva con reproductor de audio + iconos de escenas arrastrables
    y devuelve la lista actualizada de duraciones cuando el usuario mueve el inicio de una escena.
    """
    if not scenes:
        return None

    total_scenes_dur = round(sum(max(0.4, float(s["duration"])) for s in scenes), 2)
    total_dur = max(total_scenes_dur, float(audio_duration or 0.0), 1.0)

    audio_uri = None
    if audio_path and os.path.exists(audio_path):
        try:
            audio_mtime = os.path.getmtime(audio_path)
        except OSError:
            audio_mtime = 0.0
        audio_uri = _encode_audio_data_uri(audio_path, audio_mtime)

    scene_payload = []
    for sc in scenes:
        p = str(sc.get("path", ""))
        mtype = str(sc.get("media_type", "image"))
        try:
            mtime = os.path.getmtime(p) if os.path.exists(p) else 0.0
        except OSError:
            mtime = 0.0
        thumb_uri = _encode_scene_thumbnail(p, mtype, mtime)
        scene_payload.append(
            {
                "name": str(sc.get("name", "Escena")),
                "media_type": mtype,
                "duration": round(max(0.4, float(sc.get("duration", 3.0))), 2),
                "thumb": thumb_uri,
                "effect_label": str(sc.get("effect", "")),
            }
        )

    subs_payload = []
    if subtitles:
        for ch in subtitles:
            subs_payload.append(
                {
                    "start": round(float(getattr(ch, "start", 0.0)), 2),
                    "end": round(float(getattr(ch, "end", 0.0)), 2),
                    "text": str(getattr(ch, "text", "")),
                }
            )

    payload = {
        "css": TIMELINE_CSS,
        "audio_name": audio_name or "Sin audio cargado",
        "audio_data_uri": audio_uri,
        "total_duration": round(total_dur, 2),
        "scenes": scene_payload,
        "subtitles": subs_payload,
    }

    result = _timeline_v2_renderer(
        key=key,
        data=payload,
        on_changed_change=lambda: None,
    )

    if result and getattr(result, "changed", None):
        changed_obj = result.changed
        if isinstance(changed_obj, dict) and "durations" in changed_obj:
            new_durs = changed_obj["durations"]
            if isinstance(new_durs, list) and len(new_durs) == len(scenes):
                return [round(max(0.4, float(d)), 2) for d in new_durs]

    return None


# ---------------------------------------------------------------------------
# Cabecera Principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="studio-header">
        <h1>🎬 AutoReel Studio</h1>
        <p>Estudio de ensamblaje automático de videos verticales 9:16 sin marcas de agua con Ken Burns, subtítulos IA en español y exportación lista para Reels / TikTok.</p>
        <div class="badge-row">
            <span class="studio-badge">📐 9:16 · 1080×1920</span>
            <span class="studio-badge">🎞️ 30 FPS Constantes</span>
            <span class="studio-badge">🎥 Ken Burns 1.00↔1.12 + Paneos</span>
            <span class="studio-badge">✨ Crossfade 0.3s</span>
            <span class="studio-badge">🗣️ Subtítulos ES · faster-whisper</span>
            <span class="studio-badge">🚀 MP4 H.264 + AAC 192kbps</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Barra Lateral — Configuración Global y Proyecto Demo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ajustes del Estudio")

    st.subheader("Especificaciones de Salida")
    st.info(
        "**Resolución:** 1080 × 1920 (9:16)\n\n"
        "**Fotogramas:** 30 FPS\n\n"
        "**Códec Video:** H.264 (`libx264`, `yuv420p`)\n\n"
        "**Códec Audio:** AAC @ 192 kbps\n\n"
        "**Marca de agua:** Ninguna (100% limpio)"
    )

    st.session_state["crossfade_sec"] = st.slider(
        "Transición Crossfade entre escenas (s)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["crossfade_sec"]),
        step=0.05,
        help="Por defecto 0.3s según especificación cinematográfica.",
    )

    st.divider()
    st.subheader("🧪 Proyecto de Prueba Rápido")
    st.caption("¿Quieres probar todo el flujo al instante? Genera un set de audio en español + 4 escenas verticales.")
    if st.button("✨ Cargar Proyecto Demo en Español", width="stretch"):
        with st.spinner("Generando locución en español y escenas verticales 1080x1920..."):
            demo_audio, demo_imgs = create_demo_project()
            load_media_into_session(demo_audio, demo_imgs, append_scenes=False)
        st.success("¡Proyecto demo cargado! Revisa las pestañas para editar o exportar.")
        st.rerun()

    if st.button("🗑️ Limpiar Proyecto Actual", width="stretch"):
        st.session_state["audio_path"] = None
        st.session_state["audio_name"] = None
        st.session_state["audio_duration"] = 0.0
        st.session_state["bgm_path"] = None
        st.session_state["bgm_name"] = None
        st.session_state["scenes"] = []
        st.session_state["subtitles"] = []
        st.session_state["last_exported_video"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Resumen superior de métricas del proyecto
# ---------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
total_scenes_dur = round(sum(float(s["duration"]) for s in st.session_state["scenes"]), 2)
audio_dur = float(st.session_state["audio_duration"] or 0.0)

with col_m1:
    st.metric(
        "🎵 Audio Principal",
        st.session_state["audio_name"] or "Sin cargar",
        f"{audio_dur:.2f} s" if audio_dur > 0 else "0.00 s",
    )
with col_m2:
    n_imgs = sum(1 for s in st.session_state["scenes"] if s["media_type"] == "image")
    n_vids = sum(1 for s in st.session_state["scenes"] if s["media_type"] == "video")
    st.metric(
        "🖼️ Escenas en Timeline",
        f"{len(st.session_state['scenes'])} escenas",
        f"{n_imgs} img · {n_vids} video",
    )
with col_m3:
    diff_dur = round(total_scenes_dur - audio_dur, 2)
    delta_label = "Sincronizado con audio" if abs(diff_dur) <= 0.05 else f"{diff_dur:+.2f} s vs audio"
    st.metric("⏱️ Duración Total Video", f"{total_scenes_dur:.2f} s", delta_label)
with col_m4:
    st.metric(
        "💬 Subtítulos (ES)",
        f"{len(st.session_state['subtitles'])} bloques",
        f"Crossfade {st.session_state['crossfade_sec']:.2f}s",
    )

# Botón rápido de exportación directa en la parte superior cuando ya hay audio y escenas cargadas
if st.session_state["audio_path"] and st.session_state["scenes"]:
    with st.container():
        q_col1, q_col2 = st.columns([2.2, 1.2])
        with q_col1:
            st.success(
                f"✅ **Proyecto listo para ensamblar:** `{st.session_state['audio_name']}` "
                f"({audio_dur:.2f}s) + **{len(st.session_state['scenes'])} escenas** con Ken Burns y Crossfade 0.3s."
            )
        with q_col2:
            if st.button(
                "🚀 Generar Subtítulos y Exportar MP4 Ahora",
                type="primary",
                width="stretch",
                key="top_quick_export_btn",
            ):
                st.session_state["trigger_quick_export"] = True

if st.session_state.get("trigger_quick_export"):
    st.session_state["trigger_quick_export"] = False
    q_prog = st.progress(0.0)
    q_stat = st.empty()
    try:
        if not st.session_state["subtitles"]:
            q_stat.info("Paso 1/2: Generando subtítulos en español con faster-whisper...")
            st.session_state["subtitles"] = transcribe_spanish_audio(
                audio_path=st.session_state["audio_path"],
                model_size=st.session_state["whisper_model"],
                custom_keywords_raw=st.session_state["custom_keywords"],
                progress_callback=lambda msg: q_stat.info(msg),
            )
        out_name = f"AutoReel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_full = os.path.join(OUTPUT_DIR, out_name)
        scene_objs = dicts_to_scene_items(st.session_state["scenes"])
        style_obj = get_current_subtitle_style()

        def _q_on_prog(ratio: float, msg: str):
            q_prog.progress(min(1.0, max(0.0, float(ratio))))
            q_stat.info(msg)

        final_mp4 = assemble_and_export_reel(
            audio_path=st.session_state["audio_path"],
            scenes=scene_objs,
            output_path=out_full,
            crossfade_sec=float(st.session_state["crossfade_sec"]),
            subtitles=st.session_state["subtitles"],
            subtitle_style=style_obj,
            bgm_path=st.session_state.get("bgm_path"),
            bgm_volume=float(st.session_state.get("bgm_volume_pct", 12)) / 100.0,
            preset="fast",
            crf=20,
            progress_callback=_q_on_prog,
        )
        st.session_state["last_exported_video"] = final_mp4
        q_prog.progress(1.0)
        q_stat.success(f"🎉 ¡Video MP4 exportado con éxito en: `{final_mp4}`!")
    except Exception as exc:
        q_stat.error(f"Error al exportar: {exc}")

if st.session_state.get("last_exported_video") and os.path.exists(st.session_state["last_exported_video"]):
    exp_p = st.session_state["last_exported_video"]
    with st.expander("🎬 Ver / Descargar último video MP4 exportado", expanded=True):
        vc1, vc2 = st.columns([1, 1.3])
        with vc1:
            st.video(exp_p)
        with vc2:
            sz_mb = os.path.getsize(exp_p) / (1024 * 1024)
            st.markdown(
                f"**Archivo guardado en:** `{exp_p}`\n\n"
                f"**Tamaño:** `{sz_mb:.2f} MB` · **Formato:** `1080×1920 (9:16) @ 30fps · H.264 + AAC 192kbps`"
            )
            with open(exp_p, "rb") as f_mp4:
                st.download_button(
                    "⬇️ Descargar Video MP4",
                    data=f_mp4,
                    file_name=os.path.basename(exp_p),
                    mime="video/mp4",
                    type="primary",
                    width="stretch",
                    key="top_download_btn",
                )


# ---------------------------------------------------------------------------
# Pestañas de flujo de trabajo
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ Medios",
        "2️⃣ Timeline y Ken Burns",
        "3️⃣ Subtítulos IA",
        "4️⃣ Exportar MP4",
    ]
)


# ===========================================================================
# TAB 1: CARGA DE MEDIOS (SUBIDA WEB O CARPETA LOCAL)
# ===========================================================================
with tab1:
    st.subheader("Origen de los Archivos Multimedia")
    source_mode = st.radio(
        "Selecciona cómo deseas cargar tu audio (MP3/WAV) y tus escenas (.mp4, .webp, .png, .jpg):",
        options=[
            "📤 Subir archivos desde el navegador",
            "📁 Seleccionar desde una carpeta local en disco",
        ],
        horizontal=True,
    )

    if source_mode.startswith("📤"):
        c_up1, c_up2 = st.columns([1, 1])
        with c_up1:
            st.markdown("#### 1. Archivo de Audio Principal (`.mp3`, `.wav`)")
            uploaded_audio = st.file_uploader(
                "Sube la pista de voz o música que marcará el ritmo del Reel",
                type=["mp3", "wav"],
                key="uploader_audio",
            )
            if uploaded_audio is not None:
                saved_audio_path = os.path.join(UPLOADS_DIR, uploaded_audio.name)
                with open(saved_audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                if st.session_state["audio_path"] != saved_audio_path:
                    st.session_state["audio_path"] = saved_audio_path
                    st.session_state["audio_name"] = uploaded_audio.name
                    st.session_state["audio_duration"] = get_audio_duration(saved_audio_path)
                    if st.session_state["scenes"]:
                        sync_scene_durations_to_audio()
                    st.success(
                        f"Audio cargado: **{uploaded_audio.name}** ({st.session_state['audio_duration']:.2f} s)"
                    )

            if st.session_state["audio_path"] and os.path.exists(st.session_state["audio_path"]):
                st.audio(st.session_state["audio_path"])

            st.divider()
            st.markdown("#### 3. Música de Fondo Opcional con Atenuación (Ducking)")
            uploaded_bgm = st.file_uploader(
                "Sube una pista musical opcional (.mp3, .wav) — se hará loop/recorte automático con fade-in/out de 1.5s",
                type=["mp3", "wav"],
                key="uploader_bgm",
            )
            if uploaded_bgm is not None:
                saved_bgm_path = os.path.join(UPLOADS_DIR, f"bgm_{uploaded_bgm.name}")
                with open(saved_bgm_path, "wb") as f:
                    f.write(uploaded_bgm.getbuffer())
                st.session_state["bgm_path"] = saved_bgm_path
                st.session_state["bgm_name"] = uploaded_bgm.name
                st.success(f"Música de fondo cargada: **{uploaded_bgm.name}**")

            bgm_pct = int(st.session_state.get("bgm_volume_pct", 12))
            bgm_db = round(20.0 * math.log10(max(0.005, bgm_pct / 100.0)), 1) if bgm_pct > 0 else -60.0
            st.session_state["bgm_volume_pct"] = st.slider(
                f"Volumen de Música de Fondo (Ducking): {bgm_pct}% ({bgm_db:+.1f} dB)",
                min_value=0,
                max_value=50,
                value=bgm_pct,
                step=1,
                help="Por defecto 12% (-18.4 dB) para acompañar la locución sin opacar la voz.",
            )
            if st.session_state.get("bgm_path") and os.path.exists(st.session_state["bgm_path"]):
                col_bgm_a, col_bgm_b = st.columns([3, 1])
                with col_bgm_a:
                    st.audio(st.session_state["bgm_path"])
                with col_bgm_b:
                    if st.button("❌ Quitar BGM", width="stretch"):
                        st.session_state["bgm_path"] = None
                        st.session_state["bgm_name"] = None
                        st.rerun()

        with c_up2:
            st.markdown("#### 2. Lista de Clips de Video e Imágenes")
            uploaded_media = st.file_uploader(
                "Sube tus clips (.mp4) e imágenes (.webp, .png, .jpg) en el orden deseado",
                type=["mp4", "webp", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="uploader_scenes",
            )
            col_btn_add, col_btn_rep = st.columns(2)
            with col_btn_rep:
                if st.button(
                    "📥 Cargar como lista nueva de escenas",
                    disabled=not uploaded_media,
                    width="stretch",
                    type="primary",
                ):
                    saved_paths = []
                    for uf in uploaded_media:
                        dest = os.path.join(UPLOADS_DIR, uf.name)
                        with open(dest, "wb") as f:
                            f.write(uf.getbuffer())
                        saved_paths.append(dest)
                    load_media_into_session(st.session_state["audio_path"], saved_paths, append_scenes=False)
                    st.session_state["media_just_loaded_msg"] = (
                        f"✅ Se cargaron {len(saved_paths)} escenas al timeline y se sincronizaron con el audio. "
                        "Ya puedes hacer clic en el botón de abajo para exportar tu video MP4 o revisar la pestaña 2️⃣."
                    )
                    st.rerun()
            with col_btn_add:
                if st.button(
                    "➕ Añadir al final del timeline",
                    disabled=not uploaded_media,
                    width="stretch",
                ):
                    saved_paths = []
                    for uf in uploaded_media:
                        dest = os.path.join(UPLOADS_DIR, uf.name)
                        with open(dest, "wb") as f:
                            f.write(uf.getbuffer())
                        saved_paths.append(dest)
                    load_media_into_session(st.session_state["audio_path"], saved_paths, append_scenes=True)
                    st.session_state["media_just_loaded_msg"] = (
                        f"✅ Se añadieron {len(saved_paths)} escenas al timeline."
                    )
                    st.rerun()

            if st.session_state.get("media_just_loaded_msg"):
                st.success(st.session_state["media_just_loaded_msg"])

            if st.session_state["scenes"]:
                st.info(
                    f"🎞️ **Timeline actual:** {len(st.session_state['scenes'])} escenas cargadas "
                    f"({total_scenes_dur:.2f} s en total)."
                )
                if st.session_state["audio_path"]:
                    if st.button(
                        "🎬 Generar Subtítulos y Exportar Video MP4 Ahora",
                        type="primary",
                        width="stretch",
                        key="tab1_direct_export_btn",
                    ):
                        st.session_state["trigger_quick_export"] = True
                        st.rerun()

    else:
        st.markdown("#### Explorador de Carpeta Local")
        folder_input = st.text_input(
            "Ruta de la carpeta local con tus recursos multimedia:",
            value=DEMO_DIR if os.path.exists(DEMO_DIR) else WORKSPACE_DIR,
            help="Ingresa la ruta absoluta o relativa de una carpeta en tu equipo.",
        )

        if os.path.isdir(folder_input):
            all_files = sorted(os.listdir(folder_input))
            audio_candidates = [
                f for f in all_files if os.path.splitext(f)[1].lower() in {".mp3", ".wav"}
            ]
            media_candidates = [
                f
                for f in all_files
                if os.path.splitext(f)[1].lower() in {".mp4", ".webp", ".png", ".jpg", ".jpeg"}
            ]

            c_loc1, c_loc2, c_loc3 = st.columns(3)
            with c_loc1:
                selected_local_audio = st.selectbox(
                    "Selecciona el archivo de audio local (.mp3 / .wav):",
                    options=["(Mantener actual)"] + audio_candidates,
                )
            with c_loc2:
                selected_local_bgm = st.selectbox(
                    "Música de fondo opcional (.mp3 / .wav):",
                    options=["(Sin música de fondo)"] + audio_candidates,
                )
                st.session_state["bgm_volume_pct"] = st.slider(
                    "Volumen Música (% · 12% ≈ -18 dB):",
                    min_value=0,
                    max_value=50,
                    value=int(st.session_state.get("bgm_volume_pct", 12)),
                    key="local_bgm_slider",
                )
            with c_loc3:
                selected_local_media = st.multiselect(
                    "Selecciona y ordena los clips (.mp4) e imágenes (.webp, .png, .jpg) en el orden deseado:",
                    options=media_candidates,
                    default=media_candidates,
                )

            if st.button("📂 Importar Selección desde Carpeta Local", type="primary"):
                chosen_audio = (
                    os.path.join(folder_input, selected_local_audio)
                    if selected_local_audio != "(Mantener actual)"
                    else st.session_state["audio_path"]
                )
                if selected_local_bgm != "(Sin música de fondo)":
                    st.session_state["bgm_path"] = os.path.join(folder_input, selected_local_bgm)
                    st.session_state["bgm_name"] = selected_local_bgm
                chosen_media = [os.path.join(folder_input, m) for m in selected_local_media]
                load_media_into_session(chosen_audio, chosen_media, append_scenes=False)
                st.success(
                    f"Importados {len(chosen_media)} archivos multimedia desde '{folder_input}'."
                )
                st.rerun()
        else:
            st.warning("La ruta ingresada no corresponde a una carpeta existente en el sistema.")


# ===========================================================================
# TAB 2: TIMELINE, DURACIÓN POR ESCENA Y MOVIMIENTOS KEN BURNS
# ===========================================================================
with tab2:
    st.subheader("Secuencia de Escenas, Duración y Efectos Cinematográficos Ken Burns")

    if not st.session_state["scenes"]:
        st.info(
            "Aún no hay escenas cargadas. Sube tus imágenes/videos en la pestaña **1️⃣ Medios** "
            "o haz clic en **✨ Cargar Proyecto Demo en Español** en la barra lateral."
        )
    else:
        col_t1, col_t2, col_t3, col_t4 = st.columns([1.15, 1.25, 1.15, 1.25])
        with col_t1:
            if st.button(
                "⚖️ Ajustar Duraciones Uniformes al Audio",
                width="stretch",
                help="Reparte equitativamente los segundos de todas las escenas para cubrir exactamente la duración total del audio.",
            ):
                sync_scene_durations_to_audio()
                st.success("Duraciones sincronizadas uniformemente con el audio.")
                st.rerun()
        with col_t2:
            if st.button(
                "🎯 Sincronizar por frases del guion",
                width="stretch",
                type="primary",
                help="Alinea automáticamente los cortes de cada escena con las pausas y frases numeradas del guion transcrito (ej. corte Escena 3 → Escena 4 en ~20.50s).",
            ):
                if not st.session_state["subtitles"] and st.session_state["audio_path"]:
                    with st.spinner("Transcribiendo guion con faster-whisper para detectar cortes de frase..."):
                        st.session_state["subtitles"] = transcribe_spanish_audio(
                            audio_path=st.session_state["audio_path"],
                            model_size=st.session_state["whisper_model"],
                            custom_keywords_raw=st.session_state["custom_keywords"],
                        )
                target_total = audio_dur if audio_dur > 0 else len(st.session_state["scenes"]) * 3.0
                synced_durs = compute_script_synced_durations(
                    len(st.session_state["scenes"]),
                    target_total,
                    st.session_state["subtitles"],
                )
                for sc, dur in zip(st.session_state["scenes"], synced_durs):
                    sc["duration"] = dur
                st.success("¡Cortes de escena sincronizados con las frases del guion!")
                st.rerun()
        with col_t3:
            if st.button(
                "🎬 Alternar Efectos Ken Burns",
                width="stretch",
                help="Asigna en secuencia Zoom In (1.00→1.12), Zoom Out (1.12→1.00) y Paneos Sutiles a cada imagen.",
            ):
                for idx, sc in enumerate(st.session_state["scenes"]):
                    if sc["media_type"] == "image":
                        sc["effect"] = KEN_BURNS_MODES[idx % len(KEN_BURNS_MODES)]
                st.success("Ciclo Ken Burns (Zoom In / Zoom Out / Paneos) aplicado.")
                st.rerun()
        with col_t4:
            global_fit = st.selectbox(
                "Encuadre 9:16 global:",
                options=["cover", "blur_pad"],
                format_func=lambda x: "Cubrir 9:16 (Center Crop)"
                if x == "cover"
                else "Fondo Difuminado 9:16",
            )
            if st.button("Aplicar encuadre a todas", width="stretch"):
                for sc in st.session_state["scenes"]:
                    sc["fit_mode"] = global_fit
                st.rerun()

        st.markdown("#### 🎚️ Línea de Tiempo Interactiva con Audio y Marcadores de Escena")
        st.caption(
            "Escucha el audio directamente en la línea de tiempo y **arrastra los iconos con la miniatura de cada escena "
            "en la parte superior** hacia la izquierda o derecha para elegir exactamente en qué segundo empieza cada escena."
        )
        updated_timeline_durations = render_interactive_scene_timeline(
            scenes=st.session_state["scenes"],
            audio_path=st.session_state["audio_path"],
            audio_name=st.session_state["audio_name"],
            audio_duration=audio_dur,
            subtitles=st.session_state["subtitles"],
            key="interactive_scene_timeline_v2",
        )
        if updated_timeline_durations is not None:
            changed_any = False
            for sc, new_d in zip(st.session_state["scenes"], updated_timeline_durations):
                if abs(float(sc["duration"]) - float(new_d)) > 1e-3:
                    sc["duration"] = round(max(0.4, float(new_d)), 2)
                    changed_any = True
            if changed_any:
                st.toast("✅ Punto de inicio de escena actualizado en la línea de tiempo.")
                st.rerun()

        st.markdown("#### 📋 Tabla Detallada de Escenas y Efectos Ken Burns")
        st.markdown(
            "También puedes editar directamente el **Orden**, los **Segundos por escena** o el **Movimiento Ken Burns**. "
            "La columna **`Marca de Tiempo (inicio → fin)`** refleja al instante cualquier ajuste hecho arriba en la línea de tiempo:"
        )

        # Construir DataFrame editable con marca de tiempo acumulada (inicio -> fin)
        effect_options = [
            "zoom_in",
            "zoom_out",
            "pan_left_to_right",
            "pan_right_to_left",
            "pan_top_to_bottom",
            "pan_bottom_to_top",
            "none",
        ]
        scene_rows = []
        cum_t = 0.0
        for idx, sc in enumerate(st.session_state["scenes"]):
            dur_val = round(float(sc["duration"]), 2)
            t_start = round(cum_t, 2)
            t_end = round(cum_t + dur_val, 2)
            cum_t = t_end
            scene_rows.append(
                {
                    "orden": idx + 1,
                    "archivo": sc["name"],
                    "tipo": "🖼️ Imagen" if sc["media_type"] == "image" else "🎞️ Video MP4",
                    "marca_tiempo": f"{t_start:05.2f}s → {t_end:05.2f}s",
                    "segundos": dur_val,
                    "movimiento_ken_burns": sc["effect"] if sc["media_type"] == "image" else "none",
                    "encuadre_9_16": sc.get("fit_mode", "cover"),
                    "_idx": idx,
                }
            )
        df_scenes = pd.DataFrame(scene_rows)

        edited_df = st.data_editor(
            df_scenes,
            hide_index=True,
            width="stretch",
            column_config={
                "orden": st.column_config.NumberColumn(
                    "Orden #",
                    min_value=1,
                    max_value=500,
                    step=1,
                    help="Cambia el número para reordenar las escenas.",
                ),
                "archivo": st.column_config.TextColumn("Archivo", disabled=True),
                "tipo": st.column_config.TextColumn("Tipo", disabled=True),
                "marca_tiempo": st.column_config.TextColumn(
                    "Marca de Tiempo (inicio → fin)",
                    disabled=True,
                    help="Tiempo acumulado en segundos dentro del video final.",
                ),
                "segundos": st.column_config.NumberColumn(
                    "Duración (segundos)",
                    min_value=0.5,
                    max_value=300.0,
                    step=0.05,
                    format="%.2f s",
                    help="Edita manualmente los segundos que dura esta escena. Se respeta exactamente al exportar.",
                ),
                "movimiento_ken_burns": st.column_config.SelectboxColumn(
                    "Movimiento Ken Burns (Imágenes)",
                    options=effect_options,
                    required=True,
                    help="zoom_in (1.00->1.12), zoom_out (1.12->1.00) o paneos sutiles.",
                ),
                "encuadre_9_16": st.column_config.SelectboxColumn(
                    "Ajuste 9:16",
                    options=["cover", "blur_pad"],
                    required=True,
                ),
                "_idx": None,
            },
            key="scenes_table_editor",
        )

        if st.button("💾 Guardar Cambios de Orden, Duraciones y Efectos", type="primary"):
            sorted_rows = edited_df.sort_values(by=["orden", "_idx"]).to_dict(orient="records")
            updated_scenes = []
            for new_pos, row in enumerate(sorted_rows, start=1):
                orig = dict(st.session_state["scenes"][int(row["_idx"])])
                orig["orden"] = new_pos
                orig["duration"] = round(max(0.5, float(row["segundos"])), 2)
                orig["effect"] = (
                    str(row["movimiento_ken_burns"]) if orig["media_type"] == "image" else "none"
                )
                orig["fit_mode"] = str(row["encuadre_9_16"])
                updated_scenes.append(orig)
            st.session_state["scenes"] = updated_scenes
            st.success("Timeline actualizado correctamente.")
            st.rerun()

        # Galería visual de miniaturas en el orden actual con marca de tiempo acumulada
        st.markdown("#### Vista Rápida de Escenas en Secuencia")
        cols = st.columns(min(4, max(1, len(st.session_state["scenes"]))))
        thumb_cum_t = 0.0
        for idx, sc in enumerate(st.session_state["scenes"]):
            t0 = thumb_cum_t
            t1 = round(t0 + float(sc["duration"]), 2)
            thumb_cum_t = t1
            with cols[idx % len(cols)]:
                st.caption(
                    f"**#{idx + 1} · {sc['name']}** (`{t0:.2f}s → {t1:.2f}s` · `{sc['duration']:.2f}s`)"
                )
                if sc["media_type"] == "image" and os.path.exists(sc["path"]):
                    st.image(sc["path"], width="stretch")
                    st.caption(f"🎥 `{KEN_BURNS_LABELS.get(sc['effect'], sc['effect'])}`")
                elif sc["media_type"] == "video" and os.path.exists(sc["path"]):
                    st.video(sc["path"])


# ===========================================================================
# TAB 3: SUBTÍTULOS AUTOMÁTICOS EN ESPAÑOL CON FASTER-WHISPER
# ===========================================================================
with tab3:
    st.subheader("Generación Automática de Subtítulos en Español (`faster-whisper`)")

    col_sub_cfg, col_sub_style = st.columns([1.05, 1.25])

    with col_sub_cfg:
        st.markdown("#### 1. Motor de Transcripción IA")
        st.session_state["whisper_model"] = st.selectbox(
            "Modelo `faster-whisper`:",
            options=["tiny", "base", "small", "medium"],
            index=["tiny", "base", "small", "medium"].index(st.session_state["whisper_model"]),
            help="'base' o 'small' ofrecen excelente precisión en español con gran velocidad en CPU (INT8).",
        )
        max_words_chunk = st.slider(
            "Máximo de palabras por bloque de subtítulo:",
            min_value=2,
            max_value=7,
            value=4,
            help="Bloques cortos de 3 a 4 palabras en el tercio inferior mantienen alta retención en Reels y TikTok.",
        )
        st.session_state["custom_keywords"] = st.text_input(
            "Palabras clave adicionales a resaltar (opcional, separadas por coma):",
            value=st.session_state["custom_keywords"],
            placeholder="Ejemplo: increíble, secreto, gratis, oferta, dinero",
            help="Además del detector automático de palabras clave en español, puedes forzar el resaltado de términos específicos.",
        )

        can_transcribe = bool(
            st.session_state["audio_path"] and os.path.exists(st.session_state["audio_path"])
        )
        if st.button(
            "🎙️ Generar Subtítulos Automáticos en Español",
            type="primary",
            disabled=not can_transcribe,
            width="stretch",
        ):
            status_box = st.empty()
            with st.spinner("Ejecutando faster-whisper sobre el audio en español..."):
                chunks = transcribe_spanish_audio(
                    audio_path=st.session_state["audio_path"],
                    model_size=st.session_state["whisper_model"],
                    max_words_per_chunk=max_words_chunk,
                    custom_keywords_raw=st.session_state["custom_keywords"],
                    progress_callback=lambda msg: status_box.info(msg),
                )
                st.session_state["subtitles"] = chunks
                status_box.success(
                    f"¡Listo! Se generaron {len(chunks)} bloques de subtítulos sincronizados."
                )

        if st.button("➕ Añadir bloque de subtítulo manual", width="stretch"):
            existing_subs: List[SubtitleChunk] = list(st.session_state["subtitles"])
            last_end = existing_subs[-1].end if existing_subs else 0.0
            new_rows = [
                {
                    "inicio": ch.start,
                    "fin": ch.end,
                    "texto": ch.text,
                    "palabras_clave": ", ".join(ch.keywords_list),
                }
                for ch in existing_subs
            ]
            new_rows.append(
                {
                    "inicio": round(last_end, 2),
                    "fin": round(last_end + 2.5, 2),
                    "texto": "SUBTÍTULO DESTACADO EN ESPAÑOL",
                    "palabras_clave": "destacado, español",
                }
            )
            st.session_state["subtitles"] = parse_edited_chunks(new_rows)
            st.rerun()

    with col_sub_style:
        st.markdown("#### 2. Diseño Visual en Tercio Inferior")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.session_state["sub_font_size"] = st.slider(
                "Tamaño de fuente (px)", 42, 96, int(st.session_state["sub_font_size"]), 2
            )
            st.session_state["sub_text_color"] = st.color_picker(
                "Color Texto Base", st.session_state["sub_text_color"]
            )
        with sc2:
            st.session_state["sub_stroke_width"] = st.slider(
                "Grosor Borde Oscuro", 3, 14, int(st.session_state["sub_stroke_width"]), 1
            )
            st.session_state["sub_highlight_color"] = st.color_picker(
                "Color Palabra Clave", st.session_state["sub_highlight_color"]
            )
        with sc3:
            st.session_state["sub_y_ratio"] = st.slider(
                "Posición Vertical (Tercio Inferior)",
                0.65,
                0.88,
                float(st.session_state["sub_y_ratio"]),
                0.01,
            )
            st.session_state["sub_active_color"] = st.color_picker(
                "Color Palabra Activa", st.session_state["sub_active_color"]
            )

        st.session_state["sub_highlight_mode"] = st.selectbox(
            "Modo de Resaltado Dinámico:",
            options=["keywords_and_active", "keywords_only", "active_word"],
            format_func=lambda m: {
                "keywords_and_active": "Palabras Clave (Dorado) + Palabra Pronunciada en Tiempo Real (Cian)",
                "keywords_only": "Resaltar únicamente las Palabras Clave de cada frase",
                "active_word": "Resaltar palabra por palabra al ritmo de la voz (Karaoke)",
            }[m],
        )

        chk1, chk2 = st.columns(2)
        with chk1:
            st.session_state["sub_uppercase"] = st.checkbox(
                "Convertir a MAYÚSCULAS modernas", value=bool(st.session_state["sub_uppercase"])
            )
        with chk2:
            st.session_state["sub_bg_pill"] = st.checkbox(
                "Fondo translúcido oscuro redondeado", value=bool(st.session_state["sub_bg_pill"])
            )

    # Editor interactivo de bloques de subtítulos
    if st.session_state["subtitles"]:
        st.markdown("#### 3. Editor de Subtítulos y Palabras Clave Resaltadas")
        sub_df = pd.DataFrame(
            [
                {
                    "inicio": round(ch.start, 2),
                    "fin": round(ch.end, 2),
                    "texto": ch.text,
                    "palabras_clave": ", ".join(ch.keywords_list),
                }
                for ch in st.session_state["subtitles"]
            ]
        )
        edited_sub_df = st.data_editor(
            sub_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "inicio": st.column_config.NumberColumn("Inicio (s)", min_value=0.0, step=0.1, format="%.2f"),
                "fin": st.column_config.NumberColumn("Fin (s)", min_value=0.1, step=0.1, format="%.2f"),
                "texto": st.column_config.TextColumn("Texto del Subtítulo (ES)", width="large"),
                "palabras_clave": st.column_config.TextColumn(
                    "Palabras Clave a Resaltar (separadas por coma)", width="medium"
                ),
            },
            key="subtitles_editor",
        )
        if st.button("💾 Aplicar Cambios en Subtítulos"):
            st.session_state["subtitles"] = parse_edited_chunks(
                edited_sub_df.to_dict(orient="records")
            )
            st.success("Subtítulos actualizados.")
            st.rerun()


# ===========================================================================
# TAB 4: VISTA PREVIA DE FOTOGRAMA 9:16 Y EXPORTACIÓN MP4
# ===========================================================================
with tab4:
    st.subheader("Vista Previa 9:16 (1080×1920) y Exportación Final MP4")

    if not st.session_state["scenes"] or not st.session_state["audio_path"]:
        st.warning(
            "Para previsualizar y exportar el video necesitas haber cargado **1 archivo de audio** y **al menos 1 escena**."
        )
    else:
        col_prev, col_exp = st.columns([1.0, 1.15])

        with col_prev:
            st.markdown("#### 🔍 Inspector de Fotogramas en Vivo")
            preview_t = st.slider(
                "Instante de tiempo a previsualizar (segundos):",
                min_value=0.0,
                max_value=max(0.5, float(total_scenes_dur)),
                value=min(1.2, max(0.0, float(total_scenes_dur) * 0.25)),
                step=0.1,
            )
            if st.button("🖼️ Renderizar Fotograma de Muestra (1080×1920)", width="stretch"):
                with st.spinner("Generando fotograma 9:16 con Ken Burns y subtítulos..."):
                    scene_objs = dicts_to_scene_items(st.session_state["scenes"])
                    style_obj = get_current_subtitle_style()
                    preview_img = render_preview_frame(
                        t=preview_t,
                        scenes=scene_objs,
                        crossfade_sec=float(st.session_state["crossfade_sec"]),
                        subtitles=st.session_state["subtitles"],
                        subtitle_style=style_obj,
                    )
                    st.image(
                        preview_img,
                        caption=f"Previsualización 1080×1920 en t = {preview_t:.2f} s",
                        width=340,
                    )

        with col_exp:
            st.markdown("#### 🚀 Exportación Final para Reels / TikTok / Shorts")
            default_out_name = f"AutoReel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_filename = st.text_input("Nombre del archivo MP4 de salida:", value=default_out_name)
            if not output_filename.lower().endswith(".mp4"):
                output_filename += ".mp4"

            c_q1, c_q2 = st.columns(2)
            with c_q1:
                encode_preset = st.selectbox(
                    "Velocidad de codificación H.264:",
                    options=["ultrafast", "veryfast", "fast", "medium", "slow"],
                    index=2,
                    help="'fast' o 'medium' ofrecen excelente compresión y nitidez.",
                )
            with c_q2:
                crf_val = st.slider(
                    "Calidad Visual (CRF H.264):",
                    min_value=16,
                    max_value=26,
                    value=20,
                    help="Menor valor = mayor calidad visual (18-21 es ideal para Reels/TikTok 1080p).",
                )

            auto_transcribe_on_export = st.checkbox(
                "Generar subtítulos automáticos en español antes de exportar (si aún no se han generado)",
                value=True,
            )

            st.markdown(
                """
                * **Formato de salida:** `.mp4` vertical `1080x1920` (relación `9:16`) a `30 fps`
                * **Video:** `H.264 (libx264)` · `yuv420p` · Sin marcas de agua
                * **Audio:** `AAC` a `192 kbps`
                * **Efectos incluidos:** Movimientos Ken Burns (`1.00↔1.12` y paneos), Crossfade de `0.3s` y Subtítulos en tercio inferior
                """
            )

            if st.button(
                "🎬 Renderizar y Exportar Video Final MP4",
                type="primary",
                width="stretch",
            ):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                try:
                    # 1. Generar subtítulos automáticamente si está habilitado y la lista está vacía
                    if auto_transcribe_on_export and not st.session_state["subtitles"]:
                        status_text.info("Paso 1/2: Generando subtítulos en español con faster-whisper...")
                        st.session_state["subtitles"] = transcribe_spanish_audio(
                            audio_path=st.session_state["audio_path"],
                            model_size=st.session_state["whisper_model"],
                            custom_keywords_raw=st.session_state["custom_keywords"],
                            progress_callback=lambda msg: status_text.info(msg),
                        )

                    # 2. Ensamblar y exportar MP4 final
                    out_full_path = os.path.join(OUTPUT_DIR, output_filename)
                    scene_objs = dicts_to_scene_items(st.session_state["scenes"])
                    style_obj = get_current_subtitle_style()

                    def _on_progress(ratio: float, msg: str):
                        progress_bar.progress(min(1.0, max(0.0, float(ratio))))
                        status_text.info(msg)

                    final_path = assemble_and_export_reel(
                        audio_path=st.session_state["audio_path"],
                        scenes=scene_objs,
                        output_path=out_full_path,
                        crossfade_sec=float(st.session_state["crossfade_sec"]),
                        subtitles=st.session_state["subtitles"],
                        subtitle_style=style_obj,
                        bgm_path=st.session_state.get("bgm_path"),
                        bgm_volume=float(st.session_state.get("bgm_volume_pct", 12)) / 100.0,
                        preset=encode_preset,
                        crf=int(crf_val),
                        progress_callback=_on_progress,
                    )
                    st.session_state["last_exported_video"] = final_path
                    progress_bar.progress(1.0)
                    status_text.success(f"¡Video exportado exitosamente en: `{final_path}`!")
                except Exception as exc:
                    status_text.error(f"Error durante la exportación: {exc}")
                    raise

            # Mostrar reproductor y botón de descarga si hay un video exportado
            exported_path = st.session_state.get("last_exported_video")
            if exported_path and os.path.exists(exported_path):
                st.divider()
                st.markdown("#### ✅ Video Final Listo")
                v_col1, v_col2 = st.columns([1, 1.2])
                with v_col1:
                    st.video(exported_path)
                with v_col2:
                    file_size_mb = os.path.getsize(exported_path) / (1024 * 1024)
                    st.success(
                        f"**Archivo:** `{os.path.basename(exported_path)}`\n\n"
                        f"**Ruta local:** `{exported_path}`\n\n"
                        f"**Tamaño:** `{file_size_mb:.2f} MB`\n\n"
                        f"**Especificación:** `1080×1920 (9:16) · 30fps · H.264 · AAC 192kbps`"
                    )
                    with open(exported_path, "rb") as vf:
                        st.download_button(
                            label="⬇️ Descargar MP4 Listo para Reels / TikTok",
                            data=vf,
                            file_name=os.path.basename(exported_path),
                            mime="video/mp4",
                            width="stretch",
                            type="primary",
                        )
