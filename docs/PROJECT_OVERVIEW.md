# MindSight — Multimodal AI Mental Health Analysis
Last updated: 2026-08-25 08:44:46 by Bhagyajyoti-123

## Models Used
| Model     | Architecture         | Input           | Output           |
|-----------|----------------------|-----------------|------------------|
| Face      | MobileNetV2 dual-out | 96x96 RGB image | 7 emotions+3 states |
| Text      | BERT fine-tuned      | Text string     | 7 emotions       |
| Behaviour | Random Forest        | 12 features     | Stress level     |
| Audio     | WavLM large (WIP)    | Audio waveform  | 7 emotions       |

## Datasets
- Face     : FER2013 + RAF-DB + personal images
- Text     : Custom 75k text dataset
- Behaviour: Screen time + mental wellness dataset
- Audio    : TESS + RAVDESS + CREMA-D

## Fusion Weights
| Modality  | Current | With Audio |
|-----------|---------|------------|
| Face      | 0.45    | 0.40       |
| Text      | 0.30    | 0.20       |
| Behaviour | 0.25    | 0.10       |
| Audio     | TBA     | 0.30       |

## Emotion Classes
angry, disgust, fear, happy, neutral, sad, surprise

## Emotional States
- Positive : happy, surprise
- Neutral  : neutral
- Negative : angry, disgust, fear, sad

## Inputs
1. Face image (JPG/PNG)
2. Text — how you feel
3. Behaviour — 12 lifestyle questions
4. Audio — WAV/MP3 (coming soon)

## Outputs
- Predicted emotion + confidence
- Emotional state
- Probability chart per modality
- AI mental health report via Groq LLaMA3
- Results saved as JSON to Drive

## How to Run
1. Open notebooks/MindSight_Analysis.ipynb in Colab
2. Runtime → T4 GPU
3. Run cells 0-7 then Cell 8
4. Follow prompts for each input
5. Get mental health insight report

## Future Plans
- WavLM audio integration
- Phone sensor data instead of questionnaire
- Flutter mobile app
- Real-time webcam analysis
- Longitudinal mood tracking
