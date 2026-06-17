import os
import numpy as np
import librosa
import pygame
import imageio

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
AUDIO_FILE   = "suki.mp3"
IMAGE_FILE   = "suki2.png"          # Your 1920×1080 source image
OUTPUT_VIDEO = "hardware_viz_raw.mp4"
FINAL_VIDEO  = "cozy_lofi_final.mp4"
FPS          = 30

# ─── VISUALIZER CONFIGURATION (Underneath Title Text) ─────────────────────────
# Positioned cleanly under the "너의 볼살 (Neoui bolsal)" title text.
# Note: Bars grow UPWARD from the SEP_Y line.
SEP_X       = 270    # Horizontal starting position under the text
SEP_Y       = 997     # Vertical baseline position 
SEP_BARS    = 80      # Total number of visualizer bars
SEP_BAR_W   = 6       # Width of each individual bar
SEP_BAR_GAP = 4       # Space between each bar
SEP_MAX_H   = 69      # Maximum bounce height in pixels
SEP_COLOR   = (255, 130, 190)   # Hot pink LEDs to match the theme
SEP_GLOW    = (200,  40, 120)   # Darker pink glow layer underneath


# ─── STEP 1: LOAD AUDIO AND EXTRACT SPECTRUM ──────────────────────────────────
print("Step 1 — Loading audio and extracting spectrum...")
y, sr = librosa.load(AUDIO_FILE, sr=None)
duration    = librosa.get_duration(y=y, sr=sr)
total_frames = int(duration * FPS)
hop_length  = int(sr / FPS)

stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
freq_db = librosa.amplitude_to_db(stft, ref=np.max)
freq_norm = np.clip((freq_db + 80) / 80, 0, 1)   # Shape: (1025, n_frames)

print(f"   Duration: {duration:.1f}s  |  Frames: {total_frames}  |  Sample rate: {sr}Hz")

# ─── STEP 2: INIT HEADLESS PYGAME ─────────────────────────────────────────────
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
bg = pygame.image.load(IMAGE_FILE)
W, H = bg.get_size()
assert (W, H) == (1920, 1080), f"Expected 1920×1080 but got {W}×{H}. Check IMAGE_FILE."
screen = pygame.display.set_mode((W, H))

# ─── STEP 3: RENDER FRAMES ────────────────────────────────────────────────────
print(f"Step 2 — Rendering {total_frames} frames...")
writer = imageio.get_writer(OUTPUT_VIDEO, fps=FPS, codec='libx264', quality=9)

def draw_eq(surface, freq_data, frame_idx,
            origin_x, origin_y, n_bars, bar_w, bar_gap, max_h,
            color, glow_color, freq_bins):
    """Draws one equalizer strip. origin_y is the BOTTOM baseline of the bars."""
    bin_step = freq_bins // n_bars
    for i in range(n_bars):
        val   = freq_data[i * bin_step, frame_idx]
        bh    = max(3, int(val * max_h))           # Minimum 3px height when silent
        bx    = origin_x + i * (bar_w + bar_gap)
        by    = origin_y - bh                      # Subtracting height makes bars grow up

        # Glow layer (1px padding around the core bar)
        pygame.draw.rect(surface, glow_color,
                         (bx - 1, by - 1, bar_w + 2, bh + 2), border_radius=2)
        # Main foreground LED bar
        pygame.draw.rect(surface, color,
                         (bx, by, bar_w, bh), border_radius=2)

n_bins = freq_norm.shape[0]

for fi in range(min(total_frames, freq_norm.shape[1])):
    screen.blit(bg, (0, 0))

    # ── Title Text Visualizer (Bass/Mids) ──────────────────────────────
    # Grabbing lower frequency bins for a punchier rhythm response
    draw_eq(screen, freq_norm[:256, :], fi,
            SEP_X, SEP_Y, SEP_BARS, SEP_BAR_W, SEP_BAR_GAP, SEP_MAX_H,
            SEP_COLOR, SEP_GLOW, 256)

    # ── Heart Visualizer (Disabled) ────────────────────────────────────
    # Left commented out to prevent runtime errors (missing variables)
    # draw_eq(screen, freq_norm[256:512, :], fi,
    #         HEART_X, HEART_Y, HEART_BARS, HEART_BAR_W, HEART_BAR_GAP, HEART_MAX_H,
    #         HEART_COLOR, HEART_GLOW, 256)

    frame = np.frombuffer(pygame.image.tostring(screen, "RGB"), dtype=np.uint8)
    writer.append_data(frame.reshape((H, W, 3)))

    if fi % 150 == 0:
        print(f"   Frame {fi}/{total_frames} ({fi/total_frames*100:.0f}%)")

writer.close()
pygame.quit()
print("   Video render complete.")

# ─── STEP 4: MUX AUDIO + VIDEO ────────────────────────────────────────────────
print("Step 3 — Merging audio track using ffmpeg...")
os.system(
    f'ffmpeg -y '
    f'-i "{OUTPUT_VIDEO}" '
    f'-i "{AUDIO_FILE}" '
    f'-c:v copy -c:a aac '
    f'-map 0:v:0 -map 1:a:0 '
    f'-shortest '
    f'"{FINAL_VIDEO}"'
)
print(f"\n✨ Done! Output saved as '{FINAL_VIDEO}'")
