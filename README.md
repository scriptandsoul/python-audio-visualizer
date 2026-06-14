# Script & Soul Audio Visualizer

To run this script locally:
1. Clone the repository.
2. Set up your virtual environment and run `pip install -r requirements.txt`.
3. Add your own audio file to the root directory and name it `track.mp3`.
4. Add your own 1920x1080 image to the root directory and name it `background.jpg`.
5. Run `python visualizer.py`!

## Visualizer Updates

### 🎨 Adaptive Color Palette
- Uses K-Means color extraction from the background image.
- Automatically selects a dominant accent color and generates complementary bar/glow colors.
- Keeps the visualizer naturally matched to the artwork.

### 📊 Glass-Style Audio Bars
- Increased max bar height to **420px**.
- Semi-transparent bars (`BAR_ALPHA = 140`) with per-pixel alpha blending.
- Added a subtle 2px highlight edge for a glass/crystal appearance.

### 📝 Text-Safe Layout
- Added `TEXT_SAFE_ZONE_Y = 820`.
- Bars grow upward from a fixed baseline, preventing overlap with titles, subtitles, or artwork text.
- Easily adjustable for different backgrounds.

### ✨ Reflection Effect
- Added soft mirrored reflections below the baseline.
- Rendered at ~1/3 height with fading alpha for extra depth.

### 🎵 Improved Audio Response
- Switched from linear frequency bands to **Mel-scale frequency splitting**.
- Bass frequencies now drive larger, more reactive bars for more natural movement.

### 🌌 Neon Glow Pass
- Added a secondary glow layer behind each bar.
- Soft bloom effect creates anime-style night scene lighting.

### 🎥 Subtle Ken Burns Motion
- Background slowly pans during playback.
- Adds motion and atmosphere without distracting from the visualizer.

### ⚙️ Main Customization Option
- `TEXT_SAFE_ZONE_Y` — Lower it if your background text sits lower in the image; raise it if you need more protected space above the bars.

---

## 🎧 Listen Along
If you want to see this script in action or just need some music to code to, check out the channel:
👉 **Script & Soul on YouTube [https://www.youtube.com/channel/UCUQFtZLnG72V4eqaxTpEYlw](https://www.youtube.com/watch?v=MyHwTAWVl7A]
