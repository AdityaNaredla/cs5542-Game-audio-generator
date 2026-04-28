# 🎮 Game Audio Generator — MusicGen + AudioLDM2

> **CS 5542 · Quiz Challenge 2 · Foundation Models for Speech, Music, and Sound AI**

A multimodal AI pipeline that turns a single scene description into a complete **game audio pack**: atmospheric background music *and* matching sound effects — using two pretrained foundation models working together.

```text
  "dark forest at night, distant wolves"
              │
      ┌───────┴────────┐
      ▼                ▼
  MusicGen-small   AudioLDM2
  (background      (sound
   music)           effects)
      │                │
      └───────┬────────┘
              ▼
      🎵  Scene Audio Pack  🔊
```

## Why this is interesting

Indie game devs and hobbyist creators spend enormous time and money sourcing royalty-free music *and* sound effects that actually match each other. This pipeline generates **both from the same scene prompt**, ensuring they share a vibe — without licensing, without searching freesound.org for hours.

## Models used

| Purpose | Model | Source |
|---|---|---|
| Background music | **MusicGen-small** | `facebook/musicgen-small` |
| Sound effects | **AudioLDM2** | `cvssp/audioldm2` |
| (Eval) Prompt alignment | **LAION-CLAP** | `laion/clap-htsat-unfused` |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate one scene
python game_audio_generator.py --scene "haunted mansion library, ticking clock"

# 3. Batch generate from a file
python game_audio_generator.py --batch scenes.txt

# 4. Run the baseline vs engineered-prompt comparison
python game_audio_generator.py --compare
python evaluate.py
```

GPU is strongly recommended (T4 or better). CPU works but each scene takes 2–5 minutes.

## Prompt engineering

The core experiment compares **baseline** prompts (just the scene) against **engineered** prompts with domain-specific modifiers:

| Kind | Baseline | Engineered |
|---|---|---|
| Music | `{scene}` | `{scene}, orchestral video game soundtrack, cinematic, atmospheric, high quality, 120 bpm, loopable background music` |
| SFX | `{scene}` | `{scene}, high-fidelity game sound effect, clear, no music, no speech, foley recording, studio quality` |

AudioLDM2 also benefits from a **negative prompt** (`"low quality, music, speech, noise"`) to suppress spoken content and musical artifacts in SFX outputs.

## Evaluation

`evaluate.py` scores each generated clip on three axes:

1. **Prompt alignment** — cosine similarity between the CLAP text and audio embeddings. Higher = output matches the prompt better.
2. **Realism heuristics** — crest factor and spectral flatness, which flag obviously distorted or overly flat outputs.
3. **Latency** — wall-clock seconds per generation.

Results are written to `outputs/eval_scores.json`. A summary table appears on the "Results" slide of the accompanying deck.

## Repository layout

```
.
├── game_audio_generator.py   # main pipeline (CLI)
├── evaluate.py               # CLAP + heuristic scoring
├── scenes.txt                # eval scenes
├── requirements.txt
├── outputs/                  # generated .wav files + JSON results
└── README.md
```

## AI tools disclosure

| Tool | How it was used |
|---|---|
| **Anthropic Claude** | Project scaffolding, prompt-template design, README drafting, slide outline |
| **Hugging Face** | Model hosting (MusicGen, AudioLDM2, CLAP) |
| **Google Colab** | GPU runtime for generation and eval |

## Limitations

- MusicGen-small tops out around 30 seconds of coherent music; longer outputs lose structure.
- AudioLDM2 occasionally leaks humming/melodic content into SFX even with a negative prompt.
- CLAP-based alignment is a proxy, not a human judgment — high CLAP similarity ≠ subjectively "better."
- No on-the-fly looping: MusicGen doesn't natively produce seamless loops; post-processing (crossfade) would be needed for actual game use.

## License

Code: MIT. Model weights follow their respective upstream licenses (see Hugging Face model cards).
