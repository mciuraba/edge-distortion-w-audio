#!/usr/bin/env python3
"""
Bass Onset Edge Distortion
--------------------------
Triggers glitch distortion on video edges only at bass transients.
Effect decays quickly after each hit (like a snare reverb tail).

Usage:
    python bass_edge_distort.py input.mp4 output.mp4
    python bass_edge_distort.py input.mp4 output.mp4 --decay 0.2 --displacement 25
"""

import cv2
import numpy as np
import librosa
import argparse
import tempfile
import os
import sys
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────
#  AUDIO ANALYSIS
# ─────────────────────────────────────────────────────────────

def extract_audio_to_wav(video_path: str) -> str:
    import subprocess
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
         "-ar", "44100", "-f", "wav", tmp],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"  Audio extracted to: {tmp}")  # dodaj to
    return tmp


def build_onset_envelope(audio_path: str, fps: float, config: dict):
    from scipy.ndimage import uniform_filter1d

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512
    n_frames_video = int(np.ceil(duration * fps)) + 2

    print(f"  Audio: {duration:.2f}s, sr={sr}, y_max={np.abs(y).max():.4f}")

    # Kick
    _, y_perc = librosa.effects.hpss(y, margin=3.0)

    # STFT bass
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = (freqs >= config["bass_fmin"]) & (freqs <= config["bass_fmax"])
    S_bass = S[bass_mask, :]
    energy = S_bass.sum(axis=0).astype(np.float32)

    # Pitch tracking
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=config["bass_fmin"],
        fmax=config["bass_fmax"],
        sr=sr,
        hop_length=hop_length,
    )

    n_stft = energy.shape[0]
    f0 = f0[:n_stft] if len(f0) >= n_stft else np.pad(f0, (0, n_stft - len(f0)))
    voiced_flag = voiced_flag[:n_stft] if len(voiced_flag) >= n_stft else np.pad(voiced_flag, (0, n_stft - len(voiced_flag)))

    print(f"  Pitch tracking: voiced frames={voiced_flag.sum()}/{n_stft}")

    # Onset change in note 
    onset_frames_list = []

    prev_pitch = None
    prev_voiced = False
    silence_frames = 0
    min_wait = int(sr / hop_length * config["min_interval_s"])
    last_onset = -min_wait

    for i in range(n_stft):
        is_voiced = bool(voiced_flag[i])
        pitch = f0[i] if (is_voiced and not np.isnan(f0[i])) else None

        if not is_voiced:
            silence_frames += 1
            prev_voiced = False
            continue

        # Trigger conditionals
        just_appeared = not prev_voiced

        # Pitch change
        pitch_changed = False
        if prev_pitch is not None and pitch is not None:
            semitone_diff = abs(12 * np.log2(pitch / (prev_pitch + 1e-8)))
            pitch_changed = semitone_diff >= config["pitch_change_semitones"]

        if (just_appeared or pitch_changed) and (i - last_onset) >= min_wait:
            onset_frames_list.append(i)
            last_onset = i
            silence_frames = 0

        prev_pitch = pitch
        prev_voiced = is_voiced

    onset_frames = np.array(onset_frames_list, dtype=int)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    if len(onset_times) > 0:
        print(f"  Detected {len(onset_times)} bass note changes "
              f"(range: {onset_times.min():.2f}s – {onset_times.max():.2f}s)")
        print(f"  First 10: {onset_times[:10]}")
    else:
        print("  No bass notes detected – sprawdź bass_fmin/fmax")

    # Envelope from decay
    envelope = np.zeros(n_frames_video, dtype=np.float32)
    decay_frames = config["decay_time"] * fps
    half_life = decay_frames * config["decay_shape"]

    for t in onset_times:
        start = int(t * fps)
        for offset in range(int(decay_frames * 6)):
            fi = start + offset
            if fi >= n_frames_video:
                break
            val = np.exp(-offset / max(half_life, 1e-6))
            if val < 0.005:
                break
            if val > envelope[fi]:
                envelope[fi] = val

    print(f"  Envelope: max={envelope.max():.4f}, nonzero frames={np.count_nonzero(envelope)}")
    return envelope, onset_times

# ─────────────────────────────────────────────────────────────
#  DISPLACEMENT FIELD
# ─────────────────────────────────────────────────────────────

# Pre-generate a small pool of displacement fields so each onset
# gets its own "look", but the field doesn't flicker frame-to-frame.
FIELD_POOL_SIZE = 16

def precompute_displacement_pool(h: int, w: int, config: dict) -> list:
    """
    Creates FIELD_POOL_SIZE smooth random displacement fields.
    Each field is normalized to [-1, 1] – actual pixel shift is applied at render time.
    """
    pool = []
    sigma = max(h, w) * config["noise_scale"]  # spatial smoothness

    for seed in range(FIELD_POOL_SIZE):
        rng = np.random.default_rng(seed=seed * 137 + 42)
        dx = rng.standard_normal((h, w)).astype(np.float32)
        dy = rng.standard_normal((h, w)).astype(np.float32)
        dx = gaussian_filter(dx, sigma=sigma)
        dy = gaussian_filter(dy, sigma=sigma)
        # Normalize to [-1, 1]
        dx /= (np.abs(dx).max() + 1e-8)
        dy /= (np.abs(dy).max() + 1e-8)
        pool.append((dx, dy))

    return pool


def get_field_index(onset_times: np.ndarray, frame_time: float) -> int:
    """Pick which pool field to use based on the most recent onset."""
    past = onset_times[onset_times <= frame_time]
    if len(past) == 0:
        return 0
    onset_idx = int(np.searchsorted(onset_times, past[-1]))
    return onset_idx % FIELD_POOL_SIZE


# ─────────────────────────────────────────────────────────────
#  PER-FRAME DISTORTION
# ─────────────────────────────────────────────────────────────

def apply_edge_distortion(
    frame: np.ndarray,
    energy: float,
    field_dx: np.ndarray,
    field_dy: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Distort only the edge regions of the frame, scaled by energy ∈ [0,1].
    Returns the processed frame (same dtype as input).
    """
    if energy < 0.01:
        return frame

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge mask
    edges = cv2.Canny(gray, config["canny_lo"], config["canny_hi"])

    dilate_px = int(config["edge_base_px"] + energy * config["edge_max_px"])
    dilate_px = max(dilate_px, 1)
    k_size = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.dilate(edges, kernel)

    # Feather mask edges for smooth blending
    feather_sigma = max(dilate_px * 0.4, 1.0)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0),
                               sigmaX=feather_sigma)
    mask_3 = mask_f[:, :, np.newaxis]   # (H,W,1) broadcast-ready

    # Displacement
    strength = energy * config["max_displacement"]
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    remap_x = np.clip(map_x + field_dx * strength, 0, w - 1)
    remap_y = np.clip(map_y + field_dy * strength, 0, h - 1)

    distorted = cv2.remap(frame, remap_x, remap_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # Chromatic abberation :3
    ca = int(energy * config["chromatic_shift"])
    if ca > 0:
        b, g, r = cv2.split(distorted)
        M_r = np.float32([[1, 0,  ca], [0, 1, 0]])
        M_b = np.float32([[1, 0, -ca], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
        distorted = cv2.merge([b, g, r])

    # Blend: inside only the edge mask for cooler effect
    out_f = (frame.astype(np.float32) * (1.0 - mask_3)
             + distorted.astype(np.float32) * mask_3)
    return out_f.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def process_video(input_path: str, output_path: str, config: dict):
    import subprocess
    print(f"\n{'='*55}")
    print(f"  Bass Edge Distortion")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}\n")

    # ── Extract audio ─────────────────────────────────────────
    print("[1/4] Extracting audio...")
    audio_path = extract_audio_to_wav(input_path)

    # ── Video properties ──────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {frame_w}×{frame_h} @ {fps:.2f} fps  |  {total_frames} frames")

    # ── Onset envelope ────────────────────────────────────────
    print("[2/4] Analyzing bass onsets...")
    envelope, onset_times = build_onset_envelope(audio_path, fps, config)

    # ── Precompute displacement pool ──────────────────────────
    print("[3/4] Precomputing displacement fields...")
    disp_pool = precompute_displacement_pool(frame_h, frame_w, config)

    # ── Process frames ────────────────────────────────────────
    print("[4/4] Processing frames...")
    tmp_video = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (frame_w, frame_h))

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        energy     = float(envelope[fi]) if fi < len(envelope) else 0.0
        
        frame_time = fi / fps
        pool_idx   = get_field_index(onset_times, frame_time)
        dx, dy     = disp_pool[pool_idx]

        processed  = apply_edge_distortion(frame, energy, dx, dy, config)
        writer.write(processed)

        if fi % max(1, total_frames // 20) == 0:
            pct = fi / max(total_frames, 1) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  frame {fi}/{total_frames}"
                  f"  energy={energy:.3f}", end="\r")
        fi += 1

    print(f"\n  Processed {fi} frames.")
    cap.release()
    writer.release()

    # ── Merge audio back via ffmpeg ───────────────────────────
    print("  Merging audio into output...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-shortest", output_path
    ], check=True, capture_output=True)

    # ── Cleanup ───────────────────────────────────────────────
    try:
        os.remove(tmp_video)
        os.remove(audio_path)
    except OSError:
        pass

    print(f"\n✓ Done!  →  {output_path}\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Bass-reactive edge distortion (onset-triggered)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  help="Input video file (.mp4, .mov, .mkv, …)")
    p.add_argument("output", help="Output video file")

    # Effect parameters
    p.add_argument("--decay",         type=float, default=0.15,
                   help="Effect decay time in seconds after onset")
    p.add_argument("--decay-shape",   type=float, default=0.3,
                   help="Decay curve shape: smaller = snappier tail (0.1–0.8)")
    p.add_argument("--displacement",  type=float, default=22.0,
                   help="Max pixel displacement at peak")
    p.add_argument("--chromatic",     type=float, default=12.0,
                   help="Max chromatic aberration shift in pixels")
    p.add_argument("--edge-base",     type=int,   default=2,
                   help="Edge zone thickness at rest (pixels)")
    p.add_argument("--edge-max",      type=int,   default=14,
                   help="Extra edge zone thickness at peak (pixels)")
    p.add_argument("--noise-scale",   type=float, default=0.025,
                   help="Displacement field smoothness (fraction of frame size)")

    # Onset detection
    p.add_argument("--min-interval",  type=float, default=0.08,
                   help="Minimum seconds between consecutive detected onsets")
    p.add_argument("--bass-fmin",     type=float, default=30.0,
                   help="Low cut of bass band (Hz)")
    p.add_argument("--bass-fmax",     type=float, default=220.0,
                   help="High cut of bass band (Hz)")

    # Edge detection
    p.add_argument("--canny-lo",      type=int,   default=40,
                   help="Canny lower threshold")
    p.add_argument("--canny-hi",      type=int,   default=130,
                   help="Canny upper threshold")
    
    p.add_argument("--onset-delta", type=float, default=0.35,
               help="Próg detekcji: 0.35 = wzrost o 35%% ponad lokalną bazę")
    
    p.add_argument("--no-hpss", action="store_true",
                help="Wyłącz separację perkusyjną (reaguj też na nuty basu)")
    
    p.add_argument("--pitch-change", type=float, default=1.5,
               help="Ile półtonów zmiany pitchu = nowa nuta (default: 1.5)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        # envelope
        "decay_time":        args.decay,
        "decay_shape":       args.decay_shape,
        # displacement
        "max_displacement":  args.displacement,
        "noise_scale":       args.noise_scale,
        # chromatic aberration
        "chromatic_shift":   args.chromatic,
        # edge mask
        "edge_base_px":      args.edge_base,
        "edge_max_px":       args.edge_max,
        # onset detection
        "onset_delta":       args.onset_delta,
        "min_interval_s":    args.min_interval,
        "bass_fmin":         args.bass_fmin,
        "bass_fmax":         args.bass_fmax,
        # edge detection
        "canny_lo":          args.canny_lo,
        "canny_hi":          args.canny_hi,

        "onset_delta":  args.onset_delta,
        "use_hpss":     not args.no_hpss,   # ← dodaj
        "pitch_change_semitones": args.pitch_change
    }

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    process_video(args.input, args.output, config)