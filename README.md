# MindSight — Multimodal AI Mental Health Detection

**Authors:** Bhoomika Baloorgi, Bhagyajyoti G  
**Last Updated:** March 2026

## Pipeline Status: FULLY WORKING WITH AI REPORTS

## What MindSight Does
Detects a student's emotional state from 4 inputs simultaneously,
then generates a personalized mental health report using Groq AI.

## Model Accuracies
| Modality | Architecture | Dataset | Accuracy |
|---|---|---|---|
| Facial | MobileNetV2 Phase 1+2+3 | Balanced FER (45,304 images) | 79.46% |
| Voice | YAMNet + Dense | TESS (2,800 audio files) | 96.96% |
| Text | BERT (bert-base-uncased) | GoEmotions (43,410 samples) | 74.30% |
| Behavioral | Random Forest | ScreenTime vs Wellness (400 rows) | 91.25% |
| Fusion | Weighted Average | All 4 combined | Working |
| AI Report | Groq LLaMA 3.3 70B | — | Working |

## Sample Output
Input: "I feel so anxious about my exams, I cannot sleep"  
Detected: ANXIOUS at 93% confidence → Distressed stage  
Report: Personalized 4-section mental health report generated

## Models Saved In Google Drive: MindSight_Models folder
- facial_model.keras — 79.46%
- voice_model.keras — 96.96%
- text_model/ — 74.30% (BERT fine-tuned on GoEmotions)
- behavioral_model.pkl — 91.25%

## How To Run
1. Open MindSight_Fusion_Pipeline notebook in Colab
2. Mount Drive and load models
3. Install: pip install tensorflow tensorflow_hub librosa scikit-learn groq transformers
4. Run all cells in order — full pipeline works end to end

## Next Steps
1. Flask backend API — wrap fusion + report into REST endpoints
2. React frontend — chat UI with webcam and mic input
3. Real webcam and mic integration for live multimodal input
