

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    GPT2LMHeadModel,
)
from diffusers import AudioLDM2Pipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MUSICGEN_MODEL_ID = "facebook/musicgen-small"
AUDIOLDM2_MODEL_ID = "cvssp/audioldm2"

# How many tokens of MusicGen output ~= 1 second of audio
# (MusicGen uses 50 Hz tokens, so max_new_tokens=256 ~= 5.1s)
MUSIC_SECONDS_TO_TOKENS = lambda sec: int(sec * 50)


@dataclass
class GenerationResult:
    """Container for one generation run, used for eval comparisons."""
    scene: str
    prompt_used: str
    prompt_style: str          # "baseline" or "engineered"
    model: str                 # "musicgen" or "audioldm2"
    output_path: str
    duration_sec: float
    latency_sec: float
    sample_rate: int


# -----------------------------------------------------------------------------
# Prompt engineering: baseline vs engineered prompts
# -----------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "music": {
        "baseline": "{scene}",
        "engineered": (
            "{scene}, orchestral video game soundtrack, cinematic, "
            "atmospheric, high quality, 120 bpm, loopable background music"
        ),
    },
    "sfx": {
        "baseline": "{scene}",
        "engineered": (
            "{scene}, high-fidelity game sound effect, clear, "
            "no music, no speech, foley recording, studio quality"
        ),
    },
}


def build_prompt(scene: str, kind: str, style: str) -> str:
    """Fill a prompt template for a given scene, audio kind, and style."""
    return PROMPT_TEMPLATES[kind][style].format(scene=scene)


# -----------------------------------------------------------------------------
# Model loaders (lazy so we don't pay the cost unless we need them)
# -----------------------------------------------------------------------------

_musicgen = {"model": None, "processor": None}
_audioldm = {"pipe": None}


def load_musicgen():
    if _musicgen["model"] is None:
        print(f"[load] MusicGen ({MUSICGEN_MODEL_ID}) -> {DEVICE}")
        _musicgen["processor"] = AutoProcessor.from_pretrained(MUSICGEN_MODEL_ID)
        _musicgen["model"] = MusicgenForConditionalGeneration.from_pretrained(
            MUSICGEN_MODEL_ID
        ).to(DEVICE)
    return _musicgen["model"], _musicgen["processor"]


def load_audioldm2():
    if _audioldm["pipe"] is None:
        print(f"[load] AudioLDM2 ({AUDIOLDM2_MODEL_ID}) -> {DEVICE}")
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = AudioLDM2Pipeline.from_pretrained(
            AUDIOLDM2_MODEL_ID, torch_dtype=dtype
        )
        # Workaround: newer transformers loads `language_model` as `GPT2Model`
        # (no generation head), but the AudioLDM2 pipeline needs
        # `GPT2LMHeadModel`. Reload it explicitly and swap it in.
        # See: https://github.com/huggingface/diffusers/pull/11244
        try:
            pipe.language_model = GPT2LMHeadModel.from_pretrained(
                AUDIOLDM2_MODEL_ID,
                subfolder="language_model",
                torch_dtype=dtype,
            )
        except Exception as e:
            print(f"[warn] could not patch language_model: {e}")
        pipe = pipe.to(DEVICE)
        _audioldm["pipe"] = pipe
    return _audioldm["pipe"]


# -----------------------------------------------------------------------------
# Generation functions
# -----------------------------------------------------------------------------

def generate_music(
    prompt: str,
    duration_sec: float = 8.0,
    out_name: str = "music.wav",
) -> GenerationResult:
    """Generate background music with MusicGen."""
    model, processor = load_musicgen()
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=MUSIC_SECONDS_TO_TOKENS(duration_sec),
        )
    latency = time.time() - t0

    sr = model.config.audio_encoder.sampling_rate
    audio_np = audio[0, 0].cpu().numpy()
    # Normalize to int16 WAV
    audio_int16 = np.int16(audio_np / np.max(np.abs(audio_np) + 1e-9) * 32767)
    out_path = OUTPUT_DIR / out_name
    wavfile.write(out_path, sr, audio_int16)

    return GenerationResult(
        scene="",
        prompt_used=prompt,
        prompt_style="",
        model="musicgen",
        output_path=str(out_path),
        duration_sec=len(audio_np) / sr,
        latency_sec=round(latency, 2),
        sample_rate=sr,
    )


def generate_sfx(
    prompt: str,
    duration_sec: float = 4.0,
    num_inference_steps: int = 100,
    out_name: str = "sfx.wav",
) -> GenerationResult:
    """Generate a sound effect with AudioLDM2."""
    pipe = load_audioldm2()

    t0 = time.time()
    audio = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=duration_sec,
        negative_prompt="low quality, music, speech, noise",
    ).audios[0]
    latency = time.time() - t0

    sr = 16000  # AudioLDM2 outputs 16 kHz
    audio_int16 = np.int16(audio / np.max(np.abs(audio) + 1e-9) * 32767)
    out_path = OUTPUT_DIR / out_name
    wavfile.write(out_path, sr, audio_int16)

    return GenerationResult(
        scene="",
        prompt_used=prompt,
        prompt_style="",
        model="audioldm2",
        output_path=str(out_path),
        duration_sec=duration_sec,
        latency_sec=round(latency, 2),
        sample_rate=sr,
    )


# -----------------------------------------------------------------------------
# End-to-end pipeline: scene -> {music, sfx}
# -----------------------------------------------------------------------------

def generate_scene_pack(
    scene: str,
    style: str = "engineered",
    music_seconds: float = 8.0,
    sfx_seconds: float = 4.0,
    tag: str = "",
) -> Dict[str, GenerationResult]:
    """
    Given a scene description, generate a music + sfx pair.
    Returns a dict with both GenerationResult objects.
    """
    print(f"\n[scene] {scene}  (style={style})")
    safe_tag = tag or scene[:20].replace(" ", "_").replace(",", "").lower()

    music_prompt = build_prompt(scene, "music", style)
    sfx_prompt = build_prompt(scene, "sfx", style)

    music_res = generate_music(
        music_prompt,
        duration_sec=music_seconds,
        out_name=f"{safe_tag}_{style}_music.wav",
    )
    music_res.scene = scene
    music_res.prompt_style = style

    sfx_res = generate_sfx(
        sfx_prompt,
        duration_sec=sfx_seconds,
        out_name=f"{safe_tag}_{style}_sfx.wav",
    )
    sfx_res.scene = scene
    sfx_res.prompt_style = style

    print(f"  music: {music_res.output_path}  ({music_res.latency_sec}s)")
    print(f"  sfx:   {sfx_res.output_path}  ({sfx_res.latency_sec}s)")
    return {"music": music_res, "sfx": sfx_res}


# -----------------------------------------------------------------------------
# Evaluation: baseline vs engineered
# -----------------------------------------------------------------------------

EVAL_SCENES = [
    "dark forest at night with distant wolves",
    "bustling medieval marketplace",
    "spaceship engine room, humming machinery",
    "epic boss battle in a volcanic arena",
    "peaceful village by a flowing river",
]


def run_comparison(scenes: List[str] = None) -> List[Dict]:
    """
    Generate baseline + engineered audio for each scene.
    Returns a list of dicts suitable for JSON dump / later scoring.
    """
    scenes = scenes or EVAL_SCENES
    rows = []
    for i, scene in enumerate(scenes):
        for style in ["baseline", "engineered"]:
            pack = generate_scene_pack(scene, style=style, tag=f"s{i:02d}")
            rows.append({
                "scene": scene,
                "style": style,
                "music": asdict(pack["music"]),
                "sfx": asdict(pack["sfx"]),
            })
    # Save machine-readable results for the eval slide
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[ok] wrote {OUTPUT_DIR/'comparison_results.json'}")
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Game audio generator (MusicGen + AudioLDM2)")
    p.add_argument("--scene", type=str, help="Single scene description")
    p.add_argument("--batch", type=str, help="Text file, one scene per line")
    p.add_argument("--compare", action="store_true",
                   help="Run baseline vs engineered comparison on the eval set")
    p.add_argument("--style", type=str, default="engineered",
                   choices=["baseline", "engineered"])
    p.add_argument("--music-sec", type=float, default=8.0)
    p.add_argument("--sfx-sec", type=float, default=4.0)
    args = p.parse_args()

    if args.compare:
        run_comparison()
        return

    if args.batch:
        with open(args.batch) as f:
            scenes = [line.strip() for line in f if line.strip()]
    elif args.scene:
        scenes = [args.scene]
    else:
        scenes = ["cozy fireplace in a wooden cabin, crackling fire"]
        print("[info] no scene given, using default demo scene")

    for i, scene in enumerate(scenes):
        generate_scene_pack(
            scene,
            style=args.style,
            music_seconds=args.music_sec,
            sfx_seconds=args.sfx_sec,
            tag=f"scene{i:02d}",
        )


if __name__ == "__main__":
    main()