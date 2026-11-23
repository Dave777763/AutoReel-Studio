# 🎬 AutoReel Studio

Aplicación web en Python (**Streamlit** + **MoviePy** + **Pillow** + **FFmpeg** + **faster-whisper**) para ensamblar videos verticales **9:16 (`1080x1920`, `30fps`)** sin marcas de agua, listos para **Instagram Reels**, **TikTok** y **YouTube Shorts**.

## ✨ Características Principales

1. **Carga Flexible de Medios (Navegador o Carpeta Local)**:
   - Sube o selecciona localmente un archivo de audio principal (`.mp3`, `.wav`).
   - Carga y ordena una lista de clips de video (`.mp4`) e imágenes estáticas (`.webp`, `.png`, `.jpg`).
2. **Sincronización Automática y Edición Manual de Duración**:
   - Calcula y reparte automáticamente la duración de cada escena para cubrir con exactitud la duración total del audio.
   - Permite editar manualmente los segundos por escena, el orden y el tipo de encuadre 9:16 (`Center Crop` o `Fondo Difuminado 9:16`) en una tabla interactiva.
3. **Movimientos Cinematográficos Ken Burns + Crossfade 0.3s**:
   - Alterna automáticamente entre `Zoom In (1.00 → 1.12)`, `Zoom Out (1.12 → 1.00)` y paneos sutiles (`Izquierda → Derecha`, `Derecha → Izquierda`, `Arriba → Abajo`, `Abajo → Arriba`) con interpolación sub-píxel suave.
   - Transiciones **crossfade de `0.3s`** entre escenas consecutivas sin desfasar la sincronización con el audio.
4. **Subtítulos Automáticos en Español con `faster-whisper`**:
   - Transcripción automática en español con marcas de tiempo a nivel de palabra (`word_timestamps=True`).
   - Renderizado de alta nitidez con **Pillow** en el **tercio inferior** del video, con tipografía moderna, borde oscuro (`stroke`), sombra proyectada y **palabras clave resaltadas** automáticamente (con editor interactivo).
5. **Exportación Profesional MP4 sin Marcas de Agua**:
   - Codificación final en `.mp4` vertical (`1080x1920`, `30fps`) con video **H.264 (`libx264`, `yuv420p`)** y audio **AAC a `192kbps`**.

---

## 🚀 Instalación y Ejecución

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Asegúrate de contar con `ffmpeg` disponible en el sistema o a través de `imageio-ffmpeg`).*

2. **Iniciar la aplicación web en Streamlit**:
   ```bash
   streamlit run app.py
   ```
   O en Windows haciendo doble clic en `run_autoreel.bat`.
