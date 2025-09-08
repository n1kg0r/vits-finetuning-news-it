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

# Introduction
## Motivation

Text-to-Speech (TTS) is rapidly becoming a core technology for human-computer interaction, enabling more natural and efficient ways to access information. While English TTS systems are mature, high-quality expressive TTS in Italian is still scarce. Producing natural, intelligible, and expressive Italian speech remains a challenging task, particularly for nuanced applications like news narration.

## Use Case
 
News narration is an ideal testbed for expressive TTS: it combines structured content with moderate prosodic variation and benefits from high intelligibility. Many people consume audio content during commutes or multitasking, making high-quality Italian TTS a valuable tool for language learning, accessibility, and keeping up with current events. By focusing on news, we target a manageable yet impactful subset of the TTS problem.

## Research Gap

Existing open-source models such as Parler-TTS, Bark, and XTTS2 can produce expressive audio, but they often lack fine-grained control over voice, style, and prosody. Moreover, these models either have limited support for Italian, slow inference times, or licensing restrictions, making them suboptimal for our specific use case.

## Objective

The objective of this project is to build a moderately expressive Italian TTS system for news narration by fine-tuning the VITS architecture on a curated dataset of Italian news readings. This approach allows for improved prosody, intelligibility, and control over output style, while remaining feasible to develop and evaluate within a small-scale research setting.



In the following sections, we detail the pipeline for dataset preparation, model fine-tuning, and evaluation, highlighting key design choices and observed outcomes.

For a detailed exploration of state-of-the-art TTS models, see [`1_tts_exploration.ipynb`](notebooks/1_tts_exploration.ipynb).

---

## Data Collection

We prepared a high-quality dataset of Italian news readings using the following workflow:

### 1. Download YouTube audio
- Selected videos from *Il Resto del Carlino* telegiornale playlists as our narrator source
- Downloaded WAV audio using [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)

### 2. Crop and clean audio
- Removed intro/outro segments.  
- Separated vocals from background noise using [Spleeter](https://github.com/deezer/spleeter)
- Exported clean vocal tracks for further processing.

### 3. Segment audio into chunks
- Split audio into short clips (~10s) based on silence detection
- Ensured minimum chunk length (1s) and avoided overly long segments

### 4. Transcription
- Automatically generated transcripts using [OpenAI Whisper](https://github.com/openai/whisper) in Italian
- Stored metadata in `wav|text` format, ready for VITS fine-tuning

> This pipeline ensures a clean, well-segmented, and accurately transcribed dataset, forming the foundation for expressive TTS training.


---


# Data Enhancement

To improve TTS fine-tuning quality, we applied two important preprocessing steps:

## 1. Punctuation Restoration
- Many transcripts lacked proper punctuation, which led to unnatural pauses and robotic prosody.
- We used a **multilingual punctuation restoration model** ([deepmultilingualpunctuation](https://github.com/deeppavlov/DeepPavlov/tree/master/deeppavlov/models/punctuation)) to restore punctuation in all text.
- This improved the model's ability to learn **natural prosody** and produce more expressive speech.

## 2. Audio Padding
- During training (~60k steps), we noticed glitches at the start of audio clips.
- Added a small **100ms silent padding** at the beginning of each audio file.
- This mitigated initial artifacts and helped the model better learn **intonation and timing**.

> These enhancements ensure the dataset is both **textually clean** and **acoustically well-prepared**, forming a stronger foundation for expressive TTS training.


---

# Finetuning of VITS Italian pretrained checkpoint

--- 

# VITS Architecture Overview 

**VITS** (**Variational Inference Text-to-Speech**) is an end-to-end TTS model that generates audio **directly from text**, without a separate vocoder, producing natural and expressive speech.

---

## Key Ideas

### 1. Latent Audio Representation (z)
- Probabilistic vector per time step  
- Captures pitch, timbre, and subtle prosody variations

### 2. Variational Inference (VI)
- Learns posterior `q(z|x, y)` and aligns it with prior `p(z|x)`  
- Enables realistic audio generation from text

### 3. Normalizing Flows
- Transforms simple Gaussian prior → complex latent distribution  
- Allows flexible and expressive audio sampling

### 4. HiFi-GAN Decoder
- Renders waveform from sampled `z`, per time step  
- Produces smooth, natural audio

### 5. Monotonic Alignment Search
- Aligns text tokens with latent audio segments  
- Avoids explicit duration prediction

---

## Advantages over Conventional TTS

| Conventional TTS                                      | VITS                                              |
| ----------------------------------------------------- | ------------------------------------------------- |
| Predicts mel spectrogram → needs separate vocoder     | End-to-end latent-to-waveform                     |
| Deterministic output                                   | Probabilistic latent: natural variation          |
| Cascaded errors: mel → vocoder                         | Single model: fewer error cascades               |
| Often slower to train                                  | Efficient, high-quality synthesis                |

---

## Data Flow (ASCII Schema)


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

## Analogy

Think of `z` as **probabilistic keyframes in animation**:

- Each keyframe = a **cloud of possible poses**  
- Decoder samples one per frame → smooth, natural motion (or speech)  

This captures the **expressiveness and variability of real human speech** in a principled, learnable way.



--- 

# VITS Finetuning Overview

## Load Pretrained Model
- Italian VITS (`tts_models/it/mai_male/vits`)
- Pretrained checkpoint and config loaded
- Using **Coqui-TTS / IDIAP** implementations

## Dataset Preparation
- WAV files + metadata (`wav|text|text`)
- Audio processed, segmented, padded, and punctuated

## Observations Before Fine-Tuning
- Works “out of the box” for Italian
- **Strengths:** understandable pronunciation, works on Italian text
- **Weaknesses:** robotic, flat prosody, incorrect pauses, no accented letters support

--- 

# VITS Inference: Results & Tweaks


## Training Timeline & Observations

- **Before ~20k steps**  
  - First signs of understandable Italian speech  
  - No proper pausing due to missing punctuation → restored punctuation

- **Around ~60k steps**  
  - Detected padding problem at the beginning of audio  
  - Occasional uncontrollable stress errors → used apostrophe as a stress mark

- **Around ~80k steps**  
  - Improvements plateaued  
  - Stopped training to avoid overfitting on our 1.5h long dataset

---

## Text Prompt Tweaks

- Apply accents, apostrophes, double letters  
- Adjust Italian spelling and lowercase letters  

**Effects:**  
- More natural prosody  
- Better pronunciation  
- Minor exaggerated stresses remain  

---

## Audio Padding Tweaks

- Add small pause at **beginning of audio** (BOS)  
- Helps model handle initial artifacts and improves rhythm  

**Effect:** smoother, more natural speech  

---

## Reference Audio Prompting

- Provide a **reference voice sample** to mimic intonation or style  
- Combines with text tweaks for improved expressiveness  

**Observations:**  
- More control over styling
- Less accurate prosody and stress
- Can be combined with leading punctuation and apostrophes for final polish



# VITS vs xTTS2 Comparison (WER Evaluation)

## Setup

- **VITS model:** Finetuned Italian VITS (~1.5h data, 80k steps)  
- **Comparison model:** xTTS2 multilingual TTS  
- **Evaluation metric:** WER (Word Error Rate) on short Italian audio chunks  
- **Transcription:** OpenAI Whisper (`small`) with number normalization

$$
\text{WER} = \frac{S + D + I}{N}
$$

---

## WER Results

| File | VITS WER | xTTS2 WER |
|------|----------|-----------|
| chunk_00900.wav | 0.394 | 0.121 |
| chunk_00700.wav | 0.722 | 0.111 |
| chunk_00600.wav | 0.667 | 0.111 |
| chunk_00500.wav | 0.455 | 0.182 |
| chunk_00300.wav | 0.273 | 0.364 |
| chunk_00200.wav | 1.094 | 0.250 |
| chunk_00895.wav | 0.895 | 0.158 |
| chunk_00400.wav | 0.696 | 0.087 |
| chunk_00100.wav | 0.438 | 0.062 |
| chunk_00799.wav | 0.640 | 0.360 |

---

## Observations

- xTTS2 generally achieves lower WER than our finetuned VITS  
- VITS is still understandable, close enough for open-source / small dataset use  
- WER is not the most precise measurement for speech naturalness or prosody  
- Minor stress, pause, and pronunciation differences remain

---




# Final Thoughts and Future Work

## Results Overview

- Subjectively, speech quality is **good**, understandable, and reasonably natural / which is unlike most other current opoensource Italian models 
- Our finetuned Italian VITS is **open-source**, unlike most of other **good**-sounding sota models  
- Inference speed is **much faster** than other open-source models.  
- Performance is **much better than baseline VITS**, though still behind xTTS2 in WER.  
  - xTTS2 is subject to licensing restrictions and may have slightly worse intonation control, albeit better WER and sound

## Future Work

- **Quantitative evaluation:** Explore metrics for stylistic naturalness and other aspects of speech quality.  
- **More data:** Increase dataset size to improve model robustness, being careful to avoid overfitting.  
- **Stylistically versatile data:** VITS allows prompting with reference audio; diverse data can help model mimic styles better.  
- **Genre/style tagging:** Investigate tagging narration style or adding separate tokens/characters to encode stylistic variation.  
- **Architecture improvements:** Consider potential tweaks to the VITS architecture for better prosody or style control.  

> Overall, we are on the right track: the current model is usable, open-source, and a strong baseline for further experimentation.
