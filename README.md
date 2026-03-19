
# MindSight — Multimodal AI Mental Health Detection

**Authors:** Bhoomika Baloorgi, Bhagyajyoti G  
**Date:** March 2026

## Pipeline Status: ✅ WORKING END-TO-END

## What We Built
A multimodal AI system that detects mental/emotional state from 4 inputs:
- Facial expressions (webcam)
- Voice tone (microphone)
- Text sentiment (journal/chat input)
- Smartphone behavioral patterns

## Model Accuracies
| Modality | Architecture | Dataset | Accuracy |
|---|---|---|---|
| Facial | MobileNetV2 + Dense | Balanced FER (45,304 images) | 49.12% (Phase 1) |
| Voice | YAMNet + Dense | TESS (2,800 audio files) | 96.96% |
| Text | TF-IDF + Logistic Regression | GoEmotions (54,263 samples) | 63.68% |
| Behavioral | Random Forest | ScreenTime vs Wellness (400 rows) | 91.25% |
| **Fusion** | **Weighted Average** | **All 4 combined** | **✅ Working** |

## Fusion Weights (from MindLift paper)
- Facial: 0.40
- Voice: 0.30
- Text: 0.20
- Behavioral: 0.10

## Output
7 emotions: `angry, disgust, fear, happy, neutral, sad, surprise`

## Files in MindSight_Models (Google Drive)
- `facial_model.keras` — 13.3 MB
- `voice_model.keras` — 7.9 MB
- `text_model.pkl` — 0.5 MB
- `text_vectorizer.pkl` — 0.3 MB
- `behavioral_model.pkl` — 0.9 MB
- `pipeline_summary.json` — full summary

## Next Steps
1. Upgrade text model: TF-IDF → BERT (target 88%+)
2. Facial model Phase 2+3 fine-tuning (target 85%+)
3. GPT-4 API for personalized mental health reports
4. Flask backend + React frontend
