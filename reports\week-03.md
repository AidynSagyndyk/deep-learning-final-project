<img width="501" height="406" alt="image" src="https://github.com/user-attachments/assets/ffbf4d50-0fab-40f4-adf5-58a201f26e68" /># Week 03 — Deep Learning Model

## What Was Done

- Implemented **PlantCNN**: 4 ConvBlocks with BatchNorm and Dropout
- Built full training pipeline with train/validation loop
- Added **WeightedRandomSampler** to handle class imbalance
- Applied **data augmentation**: flip, rotation, ColorJitter, RandomResizedCrop
- Configured **CosineAnnealingLR** scheduler and AdamW optimizer with label smoothing
- Generated loss curves, confusion matrix, and per-class F1 chart

- ## Dataset

| Property | Value |
|----------|-------|
| Total images | 20,638 |
| Classes | 15 |
| Train / Val split | 80% / 20% |
| Val samples | 4,127 |

**Class imbalance**: severe — from `Potato___healthy` (152 images) to `Tomato__YellowLeaf__Curl_Virus` (3,208 images).  
Addressed with `WeightedRandomSampler` so rare classes are seen more often during training.

**Augmentation applied:**
- RandomHorizontalFlip + RandomVerticalFlip
- RandomRotation ±20°
- ColorJitter (brightness, contrast, saturation)
- RandomResizedCrop (scale 0.75–1.0)

- ## Model Architecture

```
Input: 3 × 128 × 128
→ ConvBlock(3  → 32)
→ ConvBlock(32 → 64)   + Dropout2d(0.2)
→ ConvBlock(64 → 128)  + Dropout2d(0.2)
→ ConvBlock(128 → 256) + Dropout2d(0.4)
→ GlobalAvgPool(1×1)
→ Linear(256 → 512) → ReLU → Dropout(0.4)
→ Linear(512 → 15)
```

Each ConvBlock: `Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2d`

**Training config:**

| Parameter | Value |
|-----------|-------|
| Image size | 128×128 |
| Batch size | 64 |
| Epochs | 25 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.4 |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Optimizer | AdamW + grad clipping (max_norm=1.0) |
| Device | Tesla T4 |

---

## Results

| Metric | Value |
|--------|-------|
| **Best Val Accuracy** | **0.9549** |
| **Weighted F1** | **0.96** |
| **Macro F1** | **0.95** |
| Training Time | 94.3 min |

---

## Classification Report

```
                                             precision  recall  f1-score  support

           Pepper__bell___Bacterial_spot       0.99      0.91      0.95      188
                 Pepper__bell___healthy       0.99      0.97      0.98      307
                  Potato___Early_blight       0.98      0.98      0.98      214
                   Potato___Late_blight       0.94      0.98      0.96      211
                       Potato___healthy       0.86      1.00      0.92       30
                  Tomato_Bacterial_spot       0.97      0.93      0.95      448
                    Tomato_Early_blight       0.87      0.97      0.92      196
                     Tomato_Late_blight       0.98      0.88      0.93      361
                       Tomato_Leaf_Mold       0.95      1.00      0.97      191
              Tomato_Septoria_leaf_spot       0.95      0.96      0.95      348
Tomato_Spider_mites_Two_spotted_mite       0.95      0.92      0.93      339
                    Tomato__Target_Spot       0.90      0.92      0.91      259
  Tomato__Tomato_YellowLeaf__Curl_Virus       1.00      0.98      0.99      665
          Tomato__Tomato_mosaic_virus       0.77      1.00      0.87       73
                       Tomato_healthy       0.95      1.00      0.97      297

                             accuracy                     0.95     4127
                            macro avg       0.94      0.96      0.95     4127
                         weighted avg       0.96      0.95      0.96     4127
```

---

## Analysis

### Best Classified Classes (F1 ≥ 0.97)
- `Tomato__YellowLeaf__Curl_Virus` — F1 = **0.99** (largest class, 665 val samples, very distinct visual pattern)
- `Pepper__bell___healthy` — F1 = **0.98**
- `Potato___Early_blight` — F1 = **0.98**
- `Tomato_Leaf_Mold` — F1 = **0.97**, recall = 1.00

### Hardest Classes (F1 < 0.93)
- `Tomato__Tomato_mosaic_virus` — F1 = **0.87**, precision = 0.77  
  Only 73 val samples. Low precision means the model over-predicts this class — likely confused with visually similar diseases.
- `Tomato__Target_Spot` — F1 = **0.91**  
  Moderate confusion with `Tomato_Early_blight` and `Tomato_Late_blight`.
- `Tomato_Early_blight` — F1 = **0.92**  
  High recall (0.97) but lower precision (0.87) — model is finding real cases but also making false positives.

### Overfitting Check
The model was trained 25 epochs with CosineAnnealing and dropout. No significant overfitting was observed based on the gap between train and val metrics.

### Effect of WeightedRandomSampler
`Potato___healthy` has only 30 val samples and achieved recall = **1.00**, which confirms the sampler successfully prevented the model from ignoring the minority class.
## Loss Curves

<img width="501" height="406" alt="image" src="https://github.com/user-attachments/assets/990f127f-df97-4f93-9e7a-5df686ed0260" />


---

## Per-Class F1

<img width="576" height="421" alt="image" src="https://github.com/user-attachments/assets/5aa2082e-8f53-4d1f-afdc-df34a2c734ee" />


## Comparison: Baseline vs Week 3

| | Baseline CNN (Week 2) | **PlantCNN (Week 3)** |
|---|---|---|
| Val Accuracy | ?- | **0.9549** |
| Weighted F1 | — | **0.96** |
| BatchNorm | ✗ | ✓ |
| Dropout | ✗ | ✓ |
| Augmentation | ✗ | ✓ |
| WeightedSampler | ✗ | ✓ |
| Scheduler | ✗ | ✓ CosineAnnealing |
| Label smoothing | ✗ | ✓ |



## Next Steps (Week 4)

- Transfer learning with **ResNet18** or **EfficientNet-B0** (pretrained on ImageNet)
- **Grad-CAM** visualizations to understand what the model looks at
- Test-time augmentation (TTA) to improve inference stability
- Focus on `Tomato_mosaic_virus` — collect more samples or apply targeted augmentation

