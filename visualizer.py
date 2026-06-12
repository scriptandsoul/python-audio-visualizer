import pygame
import numpy as np
import librosa
import cv2
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
AUDIO_PATH = "track.mp3"          # Your K-R&B / J-R&B track
IMAGE_PATH = "background.jpg"     # Your 1920x1080 anime art
OUTPUT_VIDEO = "raw_render.mp4"   # Temporary video output
FINAL_VIDEO = "final_output.mp4"  # Final video with audio synced

FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
BAR_COLOR = (255, 0, 128)         # Neon magenta/pink accent color

num_bars = 40  

# ==========================================
# 2. AUDIO ANALYSIS (LIBROSA)
# ==========================================
print("Analyzing audio frequencies with Librosa...")
y, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
total_frames = int(duration * FPS)

hop_length = int(sr / FPS)
stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))

bands = np.array_split(stft, num_bars, axis=0)
bar_data = np.array([np.mean(band, axis=0) for band in bands])

if bar_data.max() > 0:
    bar_data = (bar_data / bar_data.max()) * 300  # Max bar height in pixels

# ==========================================
# 3. PYGAME & VIDEO WRITER SETUP
# ==========================================
pygame.init()
screen = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT)) 

bg_image = pygame.image.load(IMAGE_PATH)
bg_image = pygame.transform.scale(bg_image, (VIDEO_WIDTH, VIDEO_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

# ==========================================
# 4. MAIN RENDERING LOOP
# ==========================================
print(f"Beginning render sequence ({total_frames} total frames)...")

for frame_idx in range(total_frames):
    if frame_idx >= bar_data.shape[1]:
        break

    # Draw background image
    screen.blit(bg_image, (0, 0))

    # Layout math: Centered horizontally in the empty middle space
    start_x = 420          
    available_width = 750  
    bar_gap = 6            
    bar_width = (available_width - (bar_gap * (num_bars - 1))) // num_bars

# Draw the frequency bars
    for i in range(num_bars):
        raw_height = bar_data[i, frame_idx]
        bar_height = max(5, int(raw_height)) 
        
        x_pos = start_x + i * (bar_width + bar_gap)
        
        #  NEW CENTER-ALIGNED LINE:
        # Starts at the exact middle of the screen and grows upward
        y_pos = (VIDEO_HEIGHT // 2) - bar_height  

        pygame.draw.rect(screen, BAR_COLOR, (x_pos, y_pos, bar_width, bar_height))

    # Convert frames to video matrix
    frame_string = pygame.image.tostring(screen, 'RGB')
    frame_np = np.frombuffer(frame_string, dtype=np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3))
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    video_writer.write(frame_bgr)

    if frame_idx % 500 == 0 or frame_idx == total_frames - 1:
        print(f"Rendered {frame_idx}/{total_frames} frames...")

video_writer.release()
pygame.quit()
print("Visual elements fully built. Multiplexing audio...")

# ==========================================
# 5. AUDIO STITCHING (MOVIEPY v2.0+)
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
        
    print(f"SUCCESS! Render saved to: {FINAL_VIDEO}")

except Exception as e:
    print(f"An error occurred while merging audio: {e}")