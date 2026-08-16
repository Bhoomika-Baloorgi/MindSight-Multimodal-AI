# 🎙️ Speech Emotion Recognition (7-Class) — HuBERT Fine-Tune

Fine-tuned speech emotion recognition model that classifies audio into **7 emotions**:
`angry, disgust, fear, happy, neutral, sad, surprise`

Built for a chatbot pipeline: audio input → predicted emotion → response tone adjustment.

---

## Model & Approach

| Component | Choice | Why |
|---|---|---|
| Backbone | `superb/hubert-large-superb-er` | Already emotion-pretrained (IEMOCAP), needs less data per class than a generic speech model |
| Classification head | Fresh 7-class head (`ignore_mismatched_sizes=True`) | Original checkpoint's head was 4-class; encoder weights kept, head reinitialized |
| Loss | Custom Focal Loss (γ=2.0) | Focuses gradient on hard/minority-class examples instead of relying on class weighting alone |
| Training | HuggingFace `Trainer`, fp16, gradient checkpointing, 8 epochs | — |

## Datasets (all via Kaggle)

Combined 4 acted-speech corpora, plus a targeted top-up for the weakest class:

| Dataset | Kaggle handle | Role |
|---|---|---|
| RAVDESS | `uwrfkaggler/ravdess-emotional-speech-audio` | Base training data |
| CREMA-D | `ejlok1/cremad` | Base training data (largest source; has no native "surprise" class) |
| TESS | `ejlok1/toronto-emotional-speech-set-tess` | Base training data |
| SAVEE | `ejlok1/surrey-audiovisual-expressed-emotion-savee` | Base training data |
| MELD | `zaber666/meld-dataset` | **Surprise-only top-up** — audio extracted from video clips via `ffmpeg`, filtered to surprise-labeled utterances, to offset CREMA-D having zero surprise samples |

**Class imbalance handling:**
- MELD surprise top-up (targeted at the specific gap, not blanket resampling)
- `audiomentations`-based augmentation (pitch shift, time stretch, gaussian noise, gain) applied **only** to under-represented classes in the training split, capped at 3x their original size
- Focal loss during training

Final dataset: **17,387 clips** → 13,909 train / 1,739 val / 1,739 test (stratified split).

---

## Results

### Held-out test set (same distribution as training — acted/scripted speech)

- **Accuracy: 89.94%**
- **Macro F1: 0.90** (balanced across all 7 classes, not skewed by the majority classes)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| angry | 0.86 | 0.98 | 0.91 |
| disgust | 0.92 | 0.89 | 0.90 |
| fear | 0.80 | 0.87 | 0.84 |
| happy | 0.96 | 0.87 | 0.91 |
| neutral | 0.86 | 0.97 | 0.91 |
| sad | 0.93 | 0.73 | 0.82 |
| surprise | 1.00 | 0.98 | 0.99 |

Weakest class: **sad**, most often confused with neutral and fear — an expected confusion, since low-arousal negative and ambiguous emotions share acoustic ground (pitch, energy, pacing).

### ⚠️ Real-world / customized audio — important known limitation

Manual testing on 7 self-recorded `.ogg` clips (own voice, phone mic, natural speech — **not** from the training distribution) scored only **~29% accuracy (2/7 correct)**, with a systematic bias toward predicting "happy" on out-of-distribution audio.

**This is expected and is a domain-shift problem, not a bug in the code.** The model learned emotion signatures from clean, scripted, acted-corpus speech. Real conversational audio — different mic, compression, background noise, unscripted delivery — is a meaningfully different acoustic domain. The 89.94% test accuracy is real and reproducible, but it does **not** reflect performance on arbitrary real-world audio out of the box.

**Do not treat the 89.94% figure as expected chatbot-deployment accuracy without further domain adaptation (see below).**

---

## How to Use

1. Open `Audio_Model.ipynb` in Google Colab (GPU runtime — T4 or better recommended).
2. Upload a Kaggle API token (`kaggle.json`) when prompted.
3. Run all cells top to bottom. Training takes ~2.5 hours on a T4.
4. The fine-tuned model + feature extractor are saved via `model.save_pretrained(SAVE_DIR)` / `feature_extractor.save_pretrained(SAVE_DIR)`.

### Inference

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa, torch, torch.nn.functional as F

SAVE_DIR = "<path to saved model>"
feature_extractor = AutoFeatureExtractor.from_pretrained(SAVE_DIR)
model = AutoModelForAudioClassification.from_pretrained(SAVE_DIR)

def predict_emotion(audio_path, max_duration=4.0):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=16000,
        max_length=int(16000 * max_duration), truncation=True,
        padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_values"]).logits
        probs = F.softmax(logits, dim=-1).squeeze().numpy()
    return probs  # index into model.config.id2label for class names
```

---

## Known Issues & Recommended Next Steps

1. **Domain generalization gap** (see above) — the model needs further fine-tuning on real-world, in-domain audio (matching the actual deployment mic/app/environment) before production use. This is a small, cheap follow-up fine-tune (few hundred clips, low learning rate ~5e-6, ~5 epochs), not a full retrain — plan to do this once a small labeled set of real customized recordings is collected.
2. **Confidence calibration** — consider adding a confidence threshold in the chatbot's inference wrapper; when top-1 confidence is low, surface the top-2 candidates instead of committing to a single label.
3. **Sad/neutral/fear confusion** — could improve with more real (non-acted) examples of these three classes specifically, since they're the hardest boundary in the confusion matrix.

## Environment Notes

- Requires GPU runtime (evaluation and training were memory-tight on a T4 free-tier Colab instance — `dataloader_num_workers=0`, `eval_accumulation_steps`, and `gradient_checkpointing` were needed to avoid OOM).
- `datasets.config.TORCHVISION_AVAILABLE = False` is set to work around an unrelated `datasets`/`torchvision` import bug unrelated to this project's audio pipeline.
