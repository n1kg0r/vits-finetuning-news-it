


This project documents the full pipeline of building a **high-quality TTS model** for Italian news audio, starting from **data acquisition** to **model evaluation** and a **deep dive into VITS internals**.  
We experiment with state-of-the-art models, prepare custom datasets, fine-tune [VITS](https://arxiv.org/abs/2106.06103), and evaluate both **subjective** (listening tests) and **objective** metrics.


---

## Project Overview

We built a **custom Italian TTS** model by fine-tuning VITS on a dataset of Italian news readings.  
The goal was to achieve **natural prosody**, **high intelligibility**, and **minimal glitches**, outperforming baseline open-source models.

---

## Workflow Overview

```mermaid
graph TD
    A[Download YouTube Data] --> B[Audio Segmentation & Transcription]
    B --> C[Punctuation Restoration & Normalization]
    C --> D[Train/Test Split]
    D --> E[Fine-Tune VITS]
    E --> F[Listening Evaluation]
    E --> G[Objective Metrics: WER, PESQ, STOI, MelDiff]
    G --> H[Analysis & Visualization]
    H --> I[Architecture Deep Dive]


# Data Acquisition and Preprocessing

## YouTube Downloading and Segmentation

- Used `yt-dlp` to download Italian news content.  
- Segmented into smaller chunks for easier training.

```bash
yt-dlp -x --audio-format wav <video_url>
```

# Automatic Transcription

## Used Whisper to transcribe speech into text.

```python
import whisper

model = whisper.load_model("medium")
result = model.transcribe("audio.wav", language="it")
print(result["text"])
```


# VITS Architecture Overview

**VITS** stands for **Variational Inference Text-to-Speech**.
It is a modern end-to-end TTS architecture that directly generates audio from text **without a separate vocoder**, achieving natural and expressive speech.

---

## Key Ideas / Main Tricks

1. **Latent audio representation $z$**

   * Instead of predicting mel spectrograms deterministically, VITS introduces a latent variable $z$ per time step.
   * Each $z_t$ is a **probabilistic vector**, encoding a **distribution over possible acoustic states** at that moment.
   * This captures natural variation in speech: pitch, timbre, and subtle prosody changes.

2. **Variational Inference (VI)**

   * During training, VITS learns an **approximate posterior** $q(z|x, y)$ where $x$ is text and $y$ is audio.
   * The goal is to **align the posterior with a prior** $p(z|x)$ sampled from text, so we can generate realistic audio from text alone.
   * VI is the main reason it is called **VITS**.

3. **Normalizing Flows**

   * Used to transform a simple latent prior (e.g., Gaussian) into a complex latent distribution matching the posterior.
   * This allows highly flexible, expressive audio sampling.

4. **HiFi-GAN Decoder**

   * Converts sampled latent $z$ into waveform.
   * Acts like a **renderer**, picking one plausible audio curve from the distribution encoded in $z$.
   * Works **per time step**, producing smooth audio with natural micro-variations.

5. **Monotonic Alignment Search**

   * Ensures alignment between text and latent audio representations.
   * Avoids explicit duration predictors; the model learns **which latent segments correspond to which text tokens**.

---

## Why it works better than conventional TTS

| Conventional TTS                                      | VITS                                              |
| ----------------------------------------------------- | ------------------------------------------------- |
| Predicts mel spectrogram → needs separate vocoder     | End-to-end latent-to-waveform                     |
| Deterministic: one fixed output per input             | Probabilistic latent: natural variation in speech |
| Cascaded errors possible: mel errors → vocoder errors | Single model: fewer error cascades                |
| Often slower to train                                 | Efficient and high-quality synthesis              |

---

## Data / Information Flow (ASCII Schema)

```
Text input
   │
   ▼
Text Encoder ────────────────┐
   │                        │
   ▼                        │
Latent prior p(z|text)      │
                             │
Audio Encoder ──> Posterior q(z|text, audio)
   │                            │
   ▼                            │
Normalizing Flow <──────────────┘
   │
   ▼
Sample z ~ posterior/prior
   │
   ▼
HiFi-GAN Decoder
   │
   ▼
Waveform output
```

* **Text Encoder:** transforms text into embeddings
* **Latent z:** probabilistic vector per time step (distribution over acoustic states)
* **Normalizing Flow:** transforms simple Gaussian prior → posterior
* **HiFi-GAN Decoder:** renders a plausible waveform from sampled $z$

---

### GAN / z Clarification

* The HiFi-GAN-like decoder takes **each $z_t$** sampled from the latent distribution and converts it into **audio waveform for that segment**.
* It doesn’t operate on the whole curve at once; rather, it **processes each latent time step conditioned on previous waveform context**, ensuring continuity and naturalness.

---

### Analogy

Think of $z$ like **probabilistic keyframes in animation**:

* Each keyframe isn’t a single pose, but a **cloud of possible poses**.
* The decoder samples one plausible pose per frame, producing a smooth, natural motion (or audio in this case).

This captures the **expressiveness and variability of real human speech** in a principled, learnable way.

