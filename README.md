# 🎮 Game Scene Generator — MusicGen + AudioLDM2 + Stable Diffusion

> **CS 5542 · Quiz Challenge 2 · Foundation Models for Speech, Music, and Sound AI**

A multimodal AI pipeline that turns a single scene description into a complete **game scene pack**: atmospheric background music, matching sound effects, *and* concept art — all driven by the same prompt, using three pretrained foundation models working together.

```text
        "haunted mansion library, ticking clock"
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  MusicGen-small   AudioLDM2    Stable Diffusion
   (background      (sound         (concept
    music)           effects)        art)
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              🎵  +  🔊  +  🎨
                Scene Pack
```

## Why this is interesting

Indie game devs spend hours sourcing music, sound effects, and reference art that all match each other. This pipeline generates **all three from the same scene prompt**, so they share a vibe by construction — no licensing, no manual matching.

The image generator was added partway through the project after I realized I'd already used Stable Diffusion in Quiz 1 for an interior design image series. The same model and prompting techniques transferred directly — turning the audio generator into a true scene-to-multimedia system.

## Models used

| Purpose | Model | HuggingFace ID |
|---|---|---|
| Background music | **MusicGen-small** | `facebook/musicgen-small` |
| Sound effects | **AudioLDM2** | `cvssp/audioldm2` |
| Concept art | **Stable Diffusion 1.5** | `runwayml/stable-diffusion-v1-5` |
| (Eval) Prompt alignment | **LAION-CLAP** | `laion/clap-htsat-unfused` |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate one scene (audio only)
python game_audio_generator.py --scene "haunted mansion library, ticking clock"

# 3. Generate full multimodal pack (audio + image)
python game_audio_generator.py --scene "haunted mansion library, ticking clock" --with-image

# 4. Batch from a file
python game_audio_generator.py --batch scenes.txt --with-image

# 5. Baseline vs engineered prompt comparison
python game_audio_generator.py --compare
python evaluate.py
```

GPU strongly recommended. On a typical RTX-class card, expect ~5 sec per audio clip and ~3–5 sec per image. CPU works but each scene takes 2–5 minutes.

## Prompt engineering

The core experiment compares **baseline** prompts (raw scene) against **engineered** prompts with domain-specific modifiers and negative prompts:

| Modality | Baseline | Engineered |
|---|---|---|
| Music | `{scene}` | `{scene}, orchestral video game soundtrack, cinematic, atmospheric, 120 bpm, loopable` |
| SFX | `{scene}` | `{scene}, high-fidelity game sound effect, foley recording, studio quality` |
| Image | `{scene}` | `{scene}, video game concept art, cinematic lighting, ArtStation trending, 4k` |

AudioLDM2 also receives a **negative prompt** (`"low quality, music, speech, noise"`) to suppress vocal/musical bleed in SFX. Stable Diffusion gets one too (`"low quality, blurry, watermark, text, ..."`) to suppress common artifacts.

## Results

Real CLAP scores from `outputs/eval_scores.json`, averaged across 5 evaluation scenes:

| Metric | Baseline | Engineered | Delta |
|---|---|---|---|
| Music alignment (CLAP × 100) | 19.9 | 26.6 | **+34%** |
| SFX alignment (CLAP × 100) | 7.3 | 10.3 | **+41%** |
| Music latency | 5.46 s | 5.33 s | ~0 |
| SFX latency | 5.69 s | 5.65 s | ~0 |

Engineered prompts won on every measurable axis with no latency cost. The biggest single contributor was the **negative prompt on AudioLDM2**.

Honest failure case: the "dark forest, distant wolves" scene actually scored *lower* on CLAP after prompt engineering on SFX (−0.15 vs −0.05). CLAP is treated as a proxy here, not ground truth — it correlates with human preference but isn't a substitute for it.

## Evaluation

`evaluate.py` scores each generated audio clip on:

1. **Prompt alignment** — text↔audio cosine similarity from `transformers.ClapModel` (no `laion-clap` install needed).
2. **Realism heuristics** — crest factor and spectral flatness, which flag obviously distorted or overly flat outputs.
3. **Latency** — wall-clock seconds per generation.

Results are written to `outputs/eval_scores.json`.

## Repository layout

```
.
├── game_audio_generator.py   # main pipeline: MusicGen + AudioLDM2 + SD
├── evaluate.py               # CLAP-based scoring (via transformers.ClapModel)
├── scenes.txt                # 8 evaluation scenes
├── requirements.txt
├── outputs/                  # generated .wav, .png, and .json results
└── README.md
```

## Implementation notes

A few non-obvious things worth flagging if you're reading or extending the code:

**AudioLDM2 + new transformers compatibility patch.** Recent `transformers` versions load AudioLDM2's `language_model` as `GPT2Model` (no generation head), but the `diffusers` pipeline calls `.generate()` on it — which throws `AttributeError: 'GPT2Model' has no attribute '_update_model_kwargs_for_generation'`. The fix is to load `GPT2LMHeadModel` explicitly and swap it in after `AudioLDM2Pipeline.from_pretrained()`. See `load_audioldm2()` in `game_audio_generator.py`. (Reference: [diffusers PR #11244](https://github.com/huggingface/diffusers/pull/11244).)

**Lazy model loading.** Each model is loaded on first use. If you only generate audio, SD never loads — saves ~4 GB VRAM.

**Memory savings.** SD uses fp16 + attention slicing on GPU. MusicGen and AudioLDM2 stay in their default precision because they're smaller.

**Reproducibility.** SD uses seed `42` by default for reproducibility. Same scene + style → same image. Override via the `seed` parameter on `generate_image()`.

## AI tools disclosure

| Tool | How it was used |
|---|---|
| **Anthropic Claude** | Project scaffolding, prompt-template design, debugging the AudioLDM2/transformers patch, README drafting, slide outline |
| **Hugging Face** | Hosts MusicGen, AudioLDM2, Stable Diffusion 1.5, and CLAP weights — used as-is |
| **Google Colab** | GPU runtime for generation and evaluation passes |
| **GitHub Copilot** | Inline autocomplete on Python boilerplate |

## Limitations

- **Short horizons.** MusicGen-small loses coherence past ~30 seconds; no built-in seamless looping.
- **SFX bleed.** AudioLDM2 occasionally leaks melodic content into sound effects even with a negative prompt.
- **CLAP is a proxy.** Automatic alignment correlates with — but doesn't equal — human preference. A clip can score high on CLAP and still sound off.
- **English-only.** Tested only on English prompts; multilingual behavior unverified.
- **No tempo/key control.** Can't yet constrain musical attributes from user input — only via prompt hints.
- **Image style is generic.** SD 1.5 produces decent concept art but lacks the consistency of dedicated game-art models. Swapping in SDXL or a fine-tuned checkpoint would help.

## License

Code: MIT. Model weights follow their respective upstream licenses (see Hugging Face model cards).
