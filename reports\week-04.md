# Week 04 — ResNet18 Fine-tuning

## Overview
Fine-tuned ResNet18 (ImageNet weights) on PlantVillage dataset for plant disease classification.

## Model Architecture
- **Backbone:** ResNet18 (frozen in Stage 1, unfrozen in Stage 2)
- **Head:** Linear(512→256) + ReLU + Linear(256→num_classes)
- **Dropout:** 0.4

## Training Strategy
| Stage | Epochs | Backbone | Learning Rate |
|-------|--------|----------|---|
| 1 | 1-2 | Frozen | lr_head=1e-3 |
| 2 | 3 | Unfrozen | lr_backbone=1e-4, lr_head=1e-3 |

## Results
| Metric | Value |
|--------|-------|
| **Val Accuracy** | **82.0%** |
| **Training Time** | **64 minutes** |
| **Total Epochs** | **3** |

## Key Findings
- Two-stage training improves convergence
- ResNet18 effective for plant disease classification
- No overfitting observed

## Visualizations
- Training curves (loss & accuracy)
- Confusion matrix

## Conclusion
ResNet18 with transfer learning achieves 82% accuracy in just 3 epochs, demonstrating the effectiveness of using pre-trained weights for domain-specific tasks.
