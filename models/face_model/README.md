# Face Emotion Recognition — MobileNetV2

## Architecture
- Base       : MobileNetV2 (pretrained ImageNet)
- Dual output: 7-class emotion + 3-state sentiment
- Input      : 96x96 RGB face image
- Embedding  : 1280-d GlobalAvgPool (for multimodal fusion)

## Emotion Classes (7)
angry, disgust, fear, happy, neutral, sad, surprise

## State Classes (3)
- Positive : happy, surprise
- Negative : angry, disgust, fear, sad
- Neutral  : neutral

## Datasets
- FER2013  (~35k images)
- RAF-DB   (~15k images)
- Personal face images (augmented 8x)

## Training Phases
1. Head only, backbone frozen
2. Top 30 MobileNetV2 layers unfrozen
3. Balanced resampling (3000 per class)
4. Hard-negative mining (neutral reduced to 1500)

## Note
Model .keras files are stored on Google Drive (too large for Git).
All other artifacts are in this folder.

## Fusion (coming soon)
Face embedding (1280-d)  ──┐
Audio embedding (2048-d) ──┤ → Fusion head → Emotion
Text embedding  (768-d)  ──┘
