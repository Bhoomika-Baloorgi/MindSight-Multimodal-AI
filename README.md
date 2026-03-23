# MindSight — Generative AI-Powered Student Mental Health Detection and Insights
**Authors:** Bhagyajyoti G, Bhavana P K, Bhoomika P B, Chandana S
**Last Updated:** March 2026

## Pipeline Status: FULLY WORKING WITH BERT + AI REPORTS

## Model Accuracies
| Modality | Architecture | Accuracy |
|---|---|---|
| Facial | MobileNetV2 Phase 1+2+3 | 84.60% |
| Voice | YAMNet + Dense | 96.96% |
| Text | BERT (bert-base-uncased) | 82.35% |
| Behavioral | Random Forest | 91.25% |
| Fusion | Weighted Average | Working |
| AI Report | Groq LLaMA 3.3 70B | Working |

## Sample Output
Input: "I am so stressed about my exams I cant sleep"
Modality breakdown: Facial=SAD, Voice=FEAR, Text=FEAR, Behavioral=SAD
Final: FEAR at 44.82% confidence
Report: Personalized 4-section mental health report generated

## What Teammates Need To Do Next
1. Flask backend API — wrap fusion + report into REST endpoints
2. React frontend — chat UI with webcam and mic input
3. Real webcam and mic integration for live multimodal input

## Models in Google Drive MindSight_Models folder
- facial_model.keras        84.60%
- voice_model.keras         96.96%
- bert_text_model/          82.35% (BERT)
- behavioral_model.pkl      91.25%

## How To Run
1. Open MindSight_Final_Pipeline notebook in Colab
2. Mount Drive and clone repo
3. pip install tensorflow tensorflow_hub librosa scikit-learn groq transformers torch
4. Run all cells in order
5. Full pipeline works end to end
