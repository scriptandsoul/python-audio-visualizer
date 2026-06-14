import pygame
import numpy as np
import librosa
import cv2
import os
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
AUDIO_PATH     = "track.mp3"
IMAGE_PATH     = "background.jpg"
OUTPUT_VIDEO   = "raw_render.mp4"
FINAL_VIDEO    = "final_output.mp4"

FPS            = 30
VIDEO_WIDTH    = 1920
VIDEO_HEIGHT   = 1080

NUM_BARS       = 48
MAX_BAR_HEIGHT = 660       # px — tall, dramatic bars
BAR_ALPHA      = 210       # 0-255 transparency (lower = more ghostly)
GLOW_ALPHA     = 90        # secondary bloom pass
SMOOTHING      = 0.65      # 0.0 = instant, 1.0 = frozen  (higher = slower/smoother)

# Safe zone: bars will only render BELOW this y-value
# so any title/text in the top portion is never covered
TEXT_SAFE_ZONE_Y = 820     # bars sit in bottom 260px and grow upward into the frame

# ==========================================
# 2. ADAPTIVE PALETTE FROM BACKGROUND IMAGE
# ==========================================

def extract_lofi_palette(image_path, n_colors=5):
    """
    Sample dominant colors from the background image and derive:
      - A primary bar accent (most vivid hue shifted slightly cooler)
      - A secondary/glow color (complementary / analogous)
      - A dark overlay color for the bar base
    Returns (accent_rgb, glow_rgb, shadow_rgb)
    """
    img = Image.open(image_path).convert("RGB")
    img_small = img.resize((80, 45))  # fast sampling
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

    # K-Means clustering to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.astype(np.uint8)

    # Pick the most saturated dominant color as our accent
    best_color = None
    best_sat   = -1
    for color in centers:
        hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0]
        sat = int(hsv[1])
        if sat > best_sat:
            best_sat   = sat
            best_color = color

    r, g, b = [int(c) for c in best_color]

    # Convert to HSV and tweak for anime lo-fi feel:
    # Shift hue slightly toward cool/purple, boost saturation, keep lightness dreamy
    hsv_accent = cv2.cvtColor(
        np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2HSV
    )[0][0]

    h, s, v = int(hsv_accent[0]), int(hsv_accent[1]), int(hsv_accent[2])

    # Hue shift: push 15° toward cooler territory
    h_accent = (h + 15) % 180
    s_accent = min(255, s + 60)
    v_accent = min(255, v + 40)

    accent_bgr = cv2.cvtColor(
        np.array([[[h_accent, s_accent, v_accent]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
    )[0][0]
    accent_rgb = tuple(int(c) for c in accent_bgr)

    # Glow: complementary hue, more pastel
    h_glow = (h_accent + 30) % 180
    s_glow = max(60, s_accent - 80)
    v_glow = min(255, v_accent + 30)

    glow_bgr = cv2.cvtColor(
        np.array([[[h_glow, s_glow, v_glow]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
    )[0][0]
    glow_rgb = tuple(int(c) for c in glow_bgr)

    # Shadow: very dark tint of accent for the semi-transparent bar backing
    shadow_rgb = (
        max(0, accent_rgb[0] - 170),
        max(0, accent_rgb[1] - 170),
        max(0, accent_rgb[2] - 170),
    )

    print(f"  Palette extracted → accent={accent_rgb}  glow={glow_rgb}  shadow={shadow_rgb}")
    return accent_rgb, glow_rgb, shadow_rgb


# ==========================================
# 3. AUDIO ANALYSIS
# ==========================================
print("Analyzing audio with Librosa...")
y, sr = librosa.load(AUDIO_PATH, sr=None)
duration     = librosa.get_duration(y=y, sr=sr)
total_frames = int(duration * FPS)
hop_length   = int(sr / FPS)

# STFT for frequency bars
stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))

# Use mel-scale splitting so bass bars react more — perceptually correct
mel_filter = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=NUM_BARS)
mel_spec   = np.dot(mel_filter, stft)

if mel_spec.max() > 0:
    mel_spec = (mel_spec / mel_spec.max()) * MAX_BAR_HEIGHT

# Beat detection for subtle flash effect
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
tempo    = float(np.atleast_1d(tempo)[0])   # unwrap numpy scalar/array either way
beat_set = set(beat_frames.tolist())
print(f"  Duration: {duration:.1f}s  |  BPM: {tempo:.1f}  |  Frames: {total_frames}")

# ==========================================
# 4. EXTRACT ADAPTIVE PALETTE
# ==========================================
print("Sampling background palette...")
ACCENT_RGB, GLOW_RGB, SHADOW_RGB = extract_lofi_palette(IMAGE_PATH)

# ==========================================
# 5. PYGAME & VIDEO WRITER SETUP
# ==========================================
pygame.init()
screen = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT))

bg_image = pygame.image.load(IMAGE_PATH)
bg_image = pygame.transform.scale(bg_image, (VIDEO_WIDTH, VIDEO_HEIGHT))

# Ken Burns: slight zoom buffer
bg_large = pygame.transform.scale(bg_image, (int(VIDEO_WIDTH * 1.06), int(VIDEO_HEIGHT * 1.06)))

fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ==========================================
# 6. MAIN RENDERING LOOP
# ==========================================
print(f"Rendering {total_frames} frames...")

smoothed_heights = np.zeros(NUM_BARS)

# Bar layout: spread across full bottom width, anchored at TEXT_SAFE_ZONE_Y
margin_x    = 180
usable_w    = VIDEO_WIDTH - 2 * margin_x
gap         = 6
bar_width   = (usable_w - gap * (NUM_BARS - 1)) // NUM_BARS

for frame_idx in range(total_frames):
    if frame_idx >= mel_spec.shape[1]:
        break

    # ── Ken Burns (gentle rightward drift + zoom) ──────────────────────────
    progress  = frame_idx / total_frames
    zoom_w    = bg_large.get_width()
    zoom_h    = bg_large.get_height()
    pan_x     = int(progress * (zoom_w - VIDEO_WIDTH) * 0.4)
    pan_y     = int((zoom_h - VIDEO_HEIGHT) * 0.5)
    screen.blit(bg_large, (-pan_x, -pan_y))

    # ── Smooth bar heights ──────────────────────────────────────────────────
    target          = mel_spec[:, frame_idx]
    smoothed_heights = smoothed_heights * SMOOTHING + target * (1.0 - SMOOTHING)

    # ── Soft dark vignette at the bottom to ground the bars ────────────────
    vignette = pygame.Surface((VIDEO_WIDTH, 300), pygame.SRCALPHA)
    for row in range(300):
        alpha = int((row / 300) ** 2 * 80)   # quadratic fade
        pygame.draw.line(
            vignette, (10, 8, 18, alpha),
            (0, row), (VIDEO_WIDTH, row)
        )
    screen.blit(vignette, (0, VIDEO_HEIGHT - 300))

    # ── Beat flash ──────────────────────────────────────────────────────────
    if frame_idx in beat_set:
        flash = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT), pygame.SRCALPHA)
        flash.fill((*ACCENT_RGB, 18))
        screen.blit(flash, (0, 0))

    # ── Draw bars ───────────────────────────────────────────────────────────
    for i in range(NUM_BARS):
        raw_h  = max(4, int(smoothed_heights[i]))
        x_pos  = margin_x + i * (bar_width + gap)

        # Color gradient: low freqs → accent, high freqs → glow
        t_color = i / (NUM_BARS - 1)
        bar_color = lerp_color(ACCENT_RGB, GLOW_RGB, t_color)

        # Height-based brightness: louder = slightly more opaque
        height_t  = min(1.0, raw_h / MAX_BAR_HEIGHT)
        alpha_bar = int(BAR_ALPHA * (0.55 + 0.45 * height_t))

        # ── Glow pass (wider, softer, drawn behind) ─────────────────────
        glow_w = bar_width + 10
        glow_h = raw_h + 20
        glow_x = x_pos - 5
        glow_y = TEXT_SAFE_ZONE_Y - glow_h

        glow_surf = pygame.Surface((glow_w, glow_h), pygame.SRCALPHA)
        glow_surf.fill((*GLOW_RGB, GLOW_ALPHA))
        screen.blit(glow_surf, (glow_x, glow_y))

        # ── Main bar (transparent fill) ──────────────────────────────────
        bar_surf = pygame.Surface((bar_width, raw_h), pygame.SRCALPHA)

        # Body fill
        bar_surf.fill((*bar_color, alpha_bar))

        # Top-edge highlight (thin bright line = glass/crystal feel)
        pygame.draw.line(
            bar_surf,
            (*lerp_color(bar_color, (255, 255, 255), 0.6), min(255, alpha_bar + 80)),
            (0, 0), (bar_width - 1, 0), 2
        )

        screen.blit(bar_surf, (x_pos, TEXT_SAFE_ZONE_Y - raw_h))

        # ── Reflection: mirrored ghost below the baseline ────────────────
        reflect_h  = min(raw_h // 3, 80)
        if reflect_h > 2:
            ref_surf = pygame.Surface((bar_width, reflect_h), pygame.SRCALPHA)
            for row in range(reflect_h):
                fade_a = int(alpha_bar * 0.35 * (1 - row / reflect_h) ** 2)
                pygame.draw.line(
                    ref_surf, (*bar_color, fade_a),
                    (0, row), (bar_width, row)
                )
            screen.blit(ref_surf, (x_pos, TEXT_SAFE_ZONE_Y))

    # ── Convert to video frame ──────────────────────────────────────────────
    frame_np  = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

    if frame_idx % 300 == 0 or frame_idx == total_frames - 1:
        print(f"  {frame_idx}/{total_frames} frames rendered...")

video_writer.release()
pygame.quit()
print("Frames done. Merging audio...")

# ==========================================
# 7. AUDIO STITCHING
# ==========================================
try:
    video_clip = VideoFileClip(OUTPUT_VIDEO)
    audio_clip = AudioFileClip(AUDIO_PATH)

    video_with_audio = video_clip.with_audio(audio_clip)
    video_with_audio.write_videofile(
        FINAL_VIDEO,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )

    video_clip.close()
    audio_clip.close()

    if os.path.exists(OUTPUT_VIDEO):
        os.remove(OUTPUT_VIDEO)

    print(f"\n✅  Done! Saved to: {FINAL_VIDEO}")

except Exception as e:
    print(f"Audio merge error: {e}")
