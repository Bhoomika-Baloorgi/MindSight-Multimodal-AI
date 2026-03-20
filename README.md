# MindSight — Multimodal AI Mental Health Detection

**Authors:** Bhoomika Baloorgi, Bhagyajyoti G
**Last Updated:** March 2026

## Pipeline Status: FULLY WORKING WITH AI REPORTS

## What MindSight Does
Detects a student's emotional state from 4 inputs simultaneously,
then generates a personalized mental health report using AI.

## Model Accuracies
| Modality | Architecture | Accuracy |
|---|---|---|
| Facial | MobileNetV2 Phase 1+2+3 | 79.46% |
| Voice | YAMNet + Dense | 96.96% |
| Text | TF-IDF + Logistic Regression | 63.68% |
| Behavioral | Random Forest | 91.25% |
| Fusion | Weighted Average | Working |
| AI Report | Groq LLaMA 3.3 70B | Working |

## Sample Output
Input: I am so stressed about my exams, I cant sleep
Detected: SAD at 42.84% confidence
Report: Personalized 4-section mental health report generated

## What Teammates Need To Do Next
1. BERT upgrade for text model (use Kaggle GPU)
2. Flask backend API (wrap fusion + report into REST API)
3. React frontend (chat UI + webcam + mic input)
4. Real webcam and mic integration

## Models Saved In Google Drive: MindSight_Models folder
- facial_model.keras   79.46%
- voice_model.keras    96.96%
- text_model.pkl       63.68%
- behavioral_model.pkl 91.25%

## How To Run
1. Open MindSight_Fusion_Pipeline notebook in Colab
2. Mount Drive and clone repo
3. Install libraries: pip install tensorflow tensorflow_hub librosa scikit-learn groq
4. Run all cells in order
5. Full pipeline works end to end
