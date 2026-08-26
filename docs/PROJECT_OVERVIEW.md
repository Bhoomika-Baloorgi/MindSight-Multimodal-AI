# MindSight — Multimodal AI Mental Health Analysis
Last updated: 2026-08-26 01:17:47 by Bhagyajyoti-123

## Models
| Model     | Architecture         | Input           | Output                |
|-----------|----------------------|-----------------|-----------------------|
| Face      | MobileNetV2 dual-out | 96x96 RGB image | 7 emotions + 3 states |
| Text      | BERT fine-tuned      | Text string     | 7 emotions            |
| Behaviour | Random Forest        | 12 features     | Stress level          |
| Audio     | WavLM large (WIP)    | Audio waveform  | 7 emotions            |

## Datasets
- Face     : FER2013 + RAF-DB + personal images
- Text     : Custom 75k dataset
- Behaviour: Screen time + mental wellness data
- Audio    : TESS + RAVDESS + CREMA-D

## Fusion Weights
| Modality  | Current | With Audio |
|-----------|---------|------------|
| Face      | 0.45    | 0.40       |
| Text      | 0.30    | 0.20       |
| Behaviour | 0.25    | 0.10       |
| Audio     | TBA     | 0.30       |

## Emotions
angry, disgust, fear, happy, neutral, sad, surprise

## States
- Positive : happy, surprise
- Neutral  : neutral
- Negative : angry, disgust, fear, sad

## How to Run
1. Open notebooks/MindSight_Analysis.ipynb in Colab
2. Runtime → T4 GPU
3. Run cells 0-7 then Cell 8
4. Upload face image, type feelings, answer questionnaire
5. Get AI mental health report via Groq LLaMA3

## Future Plans
- WavLM audio integration
- Phone sensor data instead of questionnaire
- Flutter mobile app
- Real-time webcam analysis
- Longitudinal mood tracking
