"""
Interactive Audio & Scene Timeline Component for AutoReel Studio.
Uses Streamlit's bidirectional `st.components.v2.component` to render:
- Built-in audio player with play/pause, scrubbing, live time readout, and waveform canvas
- Draggable scene markers/icons above the timeline to adjust exact scene start times (and even reorder scenes by dragging past another scene!)
- Scene blocks on the timeline track showing start -> end and duration
- Optional subtitle chunks track below the waveform for easy reference while aligning cuts
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image, ImageOps


@st.cache_data(show_spinner=False, max_entries=64)
def _encode_scene_thumbnail(path: str, media_type: str, mtime: float) -> str:
    """Return a compact base64 JPEG data URI (88x120) for a scene thumbnail."""
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
    """Encode the audio file as a base64 data URI for instant HTML5 playback and WebAudio waveform decoding."""
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
.tl-play-btn:active {
    transform: translateY(0);
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
    transition: background 0.15s;
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
    letter-spacing: 0.02em;
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
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Main workspace layout: Interactive timeline on left, live preview card on right */
.tl-body {
    display: grid;
    grid-template-columns: 1fr 135px;
    gap: 16px;
    align-items: stretch;
}
@media (max-width: 768px) {
    .tl-body {
        grid-template-columns: 1fr;
    }
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

/* Timeline track stack */
.tl-stack {
    position: relative;
    background: #090d16;
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 12px;
    padding: 12px 18px 14px 18px;
    overflow: visible;
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

/* Upper zone: draggable scene icons */
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
    transition: filter 0.12s ease;
    touch-action: none;
}
.tl-pin.locked {
    cursor: pointer;
}
.tl-pin:hover {
    z-index: 25;
    filter: brightness(1.1);
}
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
    line-height: 1.2;
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
    line-height: 1.1;
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
.tl-pin.active .tl-pin-pointer {
    border-top-color: #facc15;
}

.tl-pin-stem {
    width: 2px;
    height: 14px;
    background: linear-gradient(180deg, #38bdf8 0%, rgba(56, 189, 248, 0.2) 100%);
}
.tl-pin.active .tl-pin-stem {
    background: linear-gradient(180deg, #facc15 0%, rgba(250, 204, 21, 0.3) 100%);
}

/* Scene blocks bar */
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
    transition: filter 0.12s;
}
.tl-scene-seg:hover {
    filter: brightness(1.15);
}
.tl-scene-seg.active {
    box-shadow: inset 0 0 0 2px #facc15;
}
.tl-scene-seg .seg-title {
    overflow: hidden;
    text-overflow: ellipsis;
}
.tl-scene-seg .seg-dur {
    font-size: 0.66rem;
    opacity: 0.85;
    font-weight: 600;
    margin-left: 4px;
    font-variant-numeric: tabular-nums;
}

/* Audio waveform track */
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
.tl-wave-canvas {
    width: 100%;
    height: 100%;
    display: block;
}
.tl-cut-line {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1.5px;
    background: rgba(56, 189, 248, 0.55);
    pointer-events: none;
}

/* Subtitles strip (if subtitles exist) */
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
.tl-sub-chip:hover {
    background: rgba(250, 204, 21, 0.3);
}

/* Time ruler */
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

/* Playhead spanning across lanes */
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

    // Build local mutable copy of scenes with start & duration
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

    // Build DOM structure
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

    // Setup HTML5 Audio
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
        allPins.forEach((p, idx) => {
            p.classList.toggle('active', idx === activeIdx);
        });
        const allSegs = scenesBar.querySelectorAll('.tl-scene-seg');
        allSegs.forEach((s, idx) => {
            s.classList.toggle('active', idx === activeIdx);
        });

        drawWaveform();
    }

    function seekTo(t) {
        const clamped = Math.max(0, Math.min(totalDuration, t));
        if (data.audio_data_uri) {
            try {
                audio.currentTime = clamped;
            } catch (e) {}
        }
        updatePlayheadUI(clamped);
    }

    // Render ruler ticks
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

    // Render subtitles strip if available
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

    // Render scene pins, scene segments, and cut lines on waveform
    function renderScenesAndPins() {
        pinsLane.innerHTML = '';
        scenesBar.innerHTML = '';
        waveCuts.innerHTML = '';

        scenes.forEach((sc, idx) => {
            const leftPct = timeToPct(sc.start);
            const widthPct = Math.max(0.5, timeToPct(sc.end) - leftPct);

            // 1. Upper Pin (Draggable icon for scene start)
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

            // 2. Scene Segment in horizontal bar
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
            seg.addEventListener('click', () => {
                seekTo(sc.start);
            });
            scenesBar.appendChild(seg);

            // 3. Vertical cut indicator inside waveform
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

    // Drag behavior for scene start pin `idx` (1 .. N-1)
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

            // Update scene idx-1 and scene idx
            scenes[idx - 1].end = clampedStart;
            scenes[idx - 1].duration = Number((clampedStart - prevStart).toFixed(2));
            scenes[idx].start = clampedStart;
            scenes[idx].duration = Number((nextEnd - clampedStart).toFixed(2));

            dragStatusEl.textContent = `Moviendo inicio de Escena #${idx + 1} → ${clampedStart.toFixed(2)}s (Escena #${idx}: ${scenes[idx - 1].duration.toFixed(2)}s | Escena #${idx + 1}: ${scenes[idx].duration.toFixed(2)}s)`;

            // Live DOM update without rebuilding nodes so pointer capture stays active
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

        const onPointerUp = (e) => {
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

    // Waveform scrubbing by clicking or dragging on the waveform
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

    // Decode audio waveform peaks using Web Audio API
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
            // Fallback synthetic waveform if decoding fails
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

            if (barRatio <= progressRatio) {
                ctx.fillStyle = '#38bdf8';
            } else {
                ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
            }
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(x, y, barW, barH, 2 * dpr);
            } else {
                ctx.rect(x, y, barW, barH);
            }
            ctx.fill();
        }
    }

    // Audio playback controls
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

    restartBtn.addEventListener('click', () => {
        seekTo(0);
    });

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

    const resizeObserver = new ResizeObserver(() => {
        updatePlayheadUI(currentTime);
    });
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
    Mount the interactive audio + scene timeline in Streamlit and return an updated list of
    scene durations (in seconds) whenever the user drags a scene marker to a new start time.
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
