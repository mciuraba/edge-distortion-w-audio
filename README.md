# Bass Edge Distortion

Glitch distortion triggered by bass onsets — only affects edges of objects in frame.

## Setup

```bash
pip install -r requirements.txt
```

## Basic usage

```bash
python bass_edge_distort.py input.mp4 output.mp4
```

## Tuning the effect

### Effect feel
```bash
# Snappier, shorter hit
python bass_edge_distort.py input.mp4 out.mp4 --decay 0.10 --decay-shape 0.2

# Longer, heavier tail
python bass_edge_distort.py input.mp4 out.mp4 --decay 0.25 --decay-shape 0.5
```

### Effect intensity
```bash
# More aggressive distortion
python bass_edge_distort.py input.mp4 out.mp4 --displacement 35 --chromatic 20 --edge-max 20

# Subtle
python bass_edge_distort.py input.mp4 out.mp4 --displacement 10 --chromatic 6 --edge-max 8
```

### Onset detection (most important to tune per-track)
```bash
# Fewer onsets detected (only hard hits)
python bass_edge_distort.py input.mp4 out.mp4 --onset-delta 0.4

# More onsets detected (including softer bass hits)
python bass_edge_distort.py input.mp4 out.mp4 --onset-delta 0.15

# Focus only on sub-bass (kick drum body)
python bass_edge_distort.py input.mp4 out.mp4 --bass-fmin 30 --bass-fmax 100
```

## All parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--decay` | 0.15s | How long effect lasts after onset |
| `--decay-shape` | 0.3 | Tail shape: lower = snappier |
| `--displacement` | 22px | Max edge warp at peak |
| `--chromatic` | 12px | Max R/B channel split at peak |
| `--edge-base` | 2px | Edge zone always active |
| `--edge-max` | 14px | Extra edge zone at peak |
| `--noise-scale` | 0.025 | Displacement field smoothness |
| `--onset-delta` | 0.25 | Onset sensitivity (↑ = fewer) |
| `--min-interval` | 0.08s | Min gap between onsets |
| `--bass-fmin` | 30 Hz | Low cut of bass detection band |
| `--bass-fmax` | 220 Hz | High cut of bass detection band |
| `--canny-lo` | 40 | Canny edge threshold low |
| `--canny-hi` | 130 | Canny edge threshold high |

## How it works

1. **Bass onset detection** — librosa extracts onset strength restricted to
   the bass frequency band, then peak-picks transient moments only.
   Sustain is ignored.

2. **Exponential envelope** — each onset triggers a spike that decays in ~150ms,
   like a short reverb tail. Multiple hits don't accumulate.

3. **Edge mask** — Canny edge detection on each frame creates a binary mask.
   The mask is dilated (proportional to current energy) and feathered.

4. **Displacement** — smooth random displacement fields (pre-computed, one per onset)
   warp pixels inside the edge zone. Different hits get different warp shapes
   from a pool of 16 pre-generated fields.

5. **Chromatic aberration** — R and B channels shifted ±N pixels horizontally
   inside the edge zone. Classic glitch look.

6. **Blend** — final frame = original × (1−mask) + distorted × mask.
   Only edges are affected; interior of objects stays clean.