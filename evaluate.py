"""
Evaluation utilities for generated game audio.

We score each output on:
  - prompt_alignment : CLAP-based text-audio similarity (higher = closer to prompt)
  - realism          : audio spectral quality heuristics (peak-to-RMS, spectral flatness)
  - latency          : wall-clock seconds per generation

Uses Hugging Face `transformers` ClapModel — no extra installs required beyond
what's already in requirements.txt.
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np

OUTPUT_DIR = Path("outputs")
CLAP_MODEL_ID = "laion/clap-htsat-unfused"

# -----------------------------------------------------------------------------
# Heuristic (always available) metrics
# -----------------------------------------------------------------------------

def realism_score(wav_path: str) -> Dict[str, float]:
    """Rough audio quality heuristics. Higher is generally better."""
    import scipy.io.wavfile as wavfile
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio /= (np.max(np.abs(audio)) + 1e-9)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    crest = peak / (rms + 1e-9)
    spec = np.abs(np.fft.rfft(audio[: 2 ** 15]))
    spec = spec[spec > 0]
    flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-9))

    return {"rms": round(rms, 4), "crest": round(crest, 2), "flatness": round(flatness, 3)}


# -----------------------------------------------------------------------------
# CLAP-based prompt alignment (via Hugging Face transformers)
# -----------------------------------------------------------------------------

_clap = {"model": None, "processor": None, "device": None}


def _get_clap():
    if _clap["model"] is None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[load] CLAP ({CLAP_MODEL_ID}) -> {device}")
            _clap["processor"] = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
            _clap["model"] = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
            _clap["device"] = device
        except Exception as e:
            print(f"[warn] CLAP unavailable ({e}). Prompt-alignment will be skipped.")
            _clap["model"] = False
    return _clap["model"]


def prompt_alignment(wav_path: str, prompt: str) -> float:
    """Cosine similarity between CLAP audio and text embeddings."""
    import torch
    import scipy.io.wavfile as wavfile

    model = _get_clap()
    if not model:
        return float("nan")

    processor = _clap["processor"]
    device = _clap["device"]

    # Load audio and resample to 48 kHz (CLAP's expected sample rate)
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio /= (np.max(np.abs(audio)) + 1e-9)

    target_sr = 48000
    if sr != target_sr:
        # simple linear resample (good enough for eval)
        n_new = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_new),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    with torch.no_grad():
        inputs = processor(
            text=[prompt],
            audio=audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model(**inputs)
        audio_emb = outputs.audio_embeds[0]
        text_emb = outputs.text_embeds[0]

    a = audio_emb / (audio_emb.norm() + 1e-9)
    t = text_emb / (text_emb.norm() + 1e-9)
    return float(torch.dot(a, t).cpu())


# -----------------------------------------------------------------------------
# Main scoring loop
# -----------------------------------------------------------------------------

def score_results(results_json: str = "outputs/comparison_results.json"):
    with open(results_json) as f:
        rows = json.load(f)

    scored = []
    for i, row in enumerate(rows):
        print(f"[score] {i+1}/{len(rows)}  {row['style']:<11} {row['scene'][:50]}")
        out = {"scene": row["scene"], "style": row["style"]}
        for kind in ["music", "sfx"]:
            entry = row[kind]
            h = realism_score(entry["output_path"])
            align = prompt_alignment(entry["output_path"], entry["prompt_used"])
            out[kind] = {
                **h,
                "prompt_alignment": round(align, 3) if not np.isnan(align) else None,
                "latency_sec": entry["latency_sec"],
            }
        scored.append(out)

    # Aggregate
    def agg(style, kind, metric):
        vals = [r[kind][metric] for r in scored if r["style"] == style and r[kind][metric] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "music": {
            "baseline_alignment":   agg("baseline",   "music", "prompt_alignment"),
            "engineered_alignment": agg("engineered", "music", "prompt_alignment"),
            "baseline_latency":     agg("baseline",   "music", "latency_sec"),
            "engineered_latency":   agg("engineered", "music", "latency_sec"),
        },
        "sfx": {
            "baseline_alignment":   agg("baseline",   "sfx", "prompt_alignment"),
            "engineered_alignment": agg("engineered", "sfx", "prompt_alignment"),
            "baseline_latency":     agg("baseline",   "sfx", "latency_sec"),
            "engineered_latency":   agg("engineered", "sfx", "latency_sec"),
        },
    }

    # Print a nice before/after table
    print("\n" + "=" * 60)
    print(f"{'':<20} {'baseline':>12} {'engineered':>12} {'delta':>10}")
    print("-" * 60)
    for kind in ["music", "sfx"]:
        b = summary[kind]["baseline_alignment"]
        e = summary[kind]["engineered_alignment"]
        if b is not None and e is not None:
            pct = (e - b) / b * 100 if b != 0 else 0
            print(f"{kind+' alignment':<20} {b:>12.3f} {e:>12.3f} {pct:>+9.1f}%")
    print("=" * 60)

    with open(OUTPUT_DIR / "eval_scores.json", "w") as f:
        json.dump({"per_sample": scored, "summary": summary}, f, indent=2)
    print(f"\n[ok] wrote {OUTPUT_DIR/'eval_scores.json'}")


if __name__ == "__main__":
    score_results()