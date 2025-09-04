# Expressive Italian TTS for News Narration
## Fine-tuning VITS, collecting data, inference tweaks & more


This project documents the full pipeline of building a **high-quality TTS model** for Italian news audio, starting from **data acquisition** to **model finetuning** and eventual **evaluation**.  
We experiment with state-of-the-art models, prepare custom datasets, fine-tune [VITS](https://arxiv.org/abs/2106.06103), and evaluate both **subjective** (listening tests) and **objective** metrics.


---

## Project Overview

We built a **custom Italian TTS** model by fine-tuning VITS on a dataset of Italian news readings.  
The goal was to achieve **natural prosody**, **high intelligibility**, and **minimal glitches**, outperforming baseline open-source models.

1. **Explore existing models**  
   - Check what other models address similar TTS problems.  
   - Spoiler: the problem is reasonably unsolved, so we proceed to data collection.

2. **Data collection**  
   - We parse YouTube videos from *Il Resto del Carlino* telegiornale:  
     - Good audio quality  
     - Well-structured playlists  
     - Bolognese  
   - After parsing, split the audio into short clips (~10s), preferably by pauses.  
   - Transcribe audio, since VITS expects data in the format:  
     ```
     wav|text|normalized_text
     ```

3. **Data enhancement**  
   - Restore punctuation  
   - Normalize numbers  
   - Add small silence padding at the start of clips  

4. **VITS fine-tuning**  
   - Train on our dataset (~1.5h of audio)  
   - Collect outputs every ~10k steps to monitor progress

5. **Inference and engineering tweaks**  
   - After ~80k steps, the model sounds reasonably good  
   - Beyond this, likely overfitting  
   - Apply small inference tweaks (e.g., stress marks) to boost performance

6. **Evaluation**  
   - Compare outputs with the original voice  
   - Compare with *X-TTS*, one of the SOTA models (not fully open-source)

---

## Notebooks

All the workflows can be found in the `notebooks` folder.  
> Note: Some notebooks may not open directly on GitHub; download or clone the repo to view them locally.

| Notebook | 
|----------|
| 1_tts_exploration.ipynb | 
| 2_collecting_data.ipynb |
| 3_data_enhanement.ipynb |
| 4_finetuning.ipynb | 
| 5_tts_inference.ipynb |
| 6_tts_tests.ipynb | 

---

## Outputs

Some outputs are stored in the following folders:

| Folder | Description | 
|--------|-------------|
| sota_outputs | SOTA Italian TTS outputs |
| vits_outputs | Final VITS model outputs with inference tweaks | 
| vits_outputs_while_training | VITS outputs during training (no tweaks) | 

---

## Data Processing & Pipeline

- Download and segment YouTube audio  
- Automatic transcription using Whisper  
- Punctuation restoration and text normalization  
- Fine-tuning VITS on short clips  
- Evaluating by WER comparing to XTTS2 - one of the modern SOTAs for inferencing and voice cloning, not opensource though






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

