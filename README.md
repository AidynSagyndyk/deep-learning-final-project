# Plant Disease Classification using Deep Learning

Automated detection and classification of plant diseases from leaf images using convolutional neural networks.

## What We're Solving

**Problem:** Farmers manually inspect thousands of plants to detect diseases, which is:
-  Time-consuming (requires expertise)
-  Inefficient (high false negatives)
-  Expensive (requires agronomists)
-  Too late (diseases spread quickly)

**Solution:** AI-powered disease classification that:
-  Identifies diseases in **seconds**
-  Works from **smartphone photos**
-  Provides **95%+ accuracy**
-  Enables **early intervention**

##  Why It Matters

Plant diseases cause **30-40% crop losses globally**, costing farmers billions. Early detection can:
- Prevent disease spread to entire fields
- Reduce pesticide use (targeted treatment only)
- Increase crop yield by 20-30%
- Save time and money for farmers

##  What We Built

Two deep learning models trained on **20,638 leaf images**:

### Model 1: PlantCNN (Custom Architecture)
- **Accuracy:** 95.49% 
- **Design:** 4-layer CNN with BatchNorm + Dropout
- **Training:** 25 epochs (94 minutes)
- **Best for:** High-accuracy agricultural decisions

### Model 2: ResNet18 (Transfer Learning)
- **Accuracy:** 82% 
- **Design:** Pre-trained ImageNet weights + custom head
- **Training:** 3 epochs (64 minutes, limited by GPU)
- **Best for:** Rapid deployment and fast inference

##  Dataset

**PlantVillage Dataset:**
- 20,638 high-quality leaf images
- 15 disease classes (tomato, potato, pepper)
- Healthy and diseased plant categories
- Controlled lab environment (consistent lighting)

### Disease Classes
```
Tomato diseases (7):        Potato diseases (4):      Pepper diseases (2):
- Early blight              - Early blight            - Bacterial spot
- Late blight               - Late blight             - Healthy
- Leaf mold                 - Healthy
- Septoria leaf spot        
- Spider mites              + More...
- Target spot
- Mosaic virus
- Yellow leaf curl virus
```

##  How It Works

1. **Data Preprocessing**
   - Resize images to 128×128 or 224×224
   - Apply normalization (ImageNet stats)
   - Augmentation: flips, rotations, color jitter
   - Handle class imbalance with weighted sampling

2. **Model Training**
   - Custom CNN: From scratch with careful architecture design
   - ResNet18: Fine-tune pre-trained weights
   - Loss function: CrossEntropyLoss with label smoothing
   - Optimizer: AdamW with CosineAnnealingLR scheduler

3. **Evaluation**
   - Validation accuracy: 95.49% (PlantCNN) / 82% (ResNet18)
   - Confusion matrix: Shows which diseases are confused
   - F1-score: 96% weighted average
   - Per-class analysis for all 15 diseases

## Key Results

| Metric | PlantCNN | ResNet18 |
|--------|----------|----------|
| **Accuracy** | **95.49%** | 82.0% |
| **F1-Score** | **96%** | — |
| **Training Time** | 94 min | 64 min |
| **Epochs** | 25 | 3* |

*ResNet18 limited by GPU memory (would need 15+ epochs for full convergence)

##  Model Strengths

### Best Classified Classes
-  Tomato Yellow Leaf Curl Virus: **99% F1** (distinct visual pattern)
-  Pepper Healthy: **98% F1** (clear distinction)
-  Potato Early Blight: **98% F1** (consistent symptoms)

### Challenging Classes
-  Tomato Mosaic Virus: **87% F1** (rare, similar symptoms)
-  Tomato Target Spot: **91% F1** (confused with other blights)

##  Technical Stack

- **Framework:** PyTorch 2.0+
- **Vision:** torchvision, scikit-learn
- **GPU:** NVIDIA Tesla T4 (CUDA)
- **Libraries:** NumPy, Pandas, Matplotlib


## Real-World Applications

1. **Farm Management Apps** — Farmers upload leaf photo → get instant disease diagnosis
2. **Precision Agriculture** — Drones monitor fields + AI detects diseases early
3. **Supply Chain Quality** — Harvest inspection at packhouses
4. **Research** — Plant pathology studies and crop genetics
5. **Education** — Teaching students about plant diseases

##  Real-World Impact Example

**Scenario:** Farmer finds suspicious spots on tomato plants
-  Without AI: Wait for agronomist (1-2 days), possibly too late
-  With AI: Take photo with phone → get result in seconds → treat immediately

**Outcome:** Prevents disease spread, saves 30-40% of crop

##  What We Learned

1. **Custom CNNs work well** when carefully designed with BatchNorm + Dropout
2. **Transfer learning is efficient** — ResNet18 converges in 3 epochs vs 25 for custom CNN
3. **Data augmentation prevents overfitting** — especially important for small rare classes
4. **Class imbalance matters** — WeightedRandomSampler ensures all diseases are learned

##  Limitations & Future Work

### Current Limitations
- Dataset from controlled lab (may not match real farm conditions)
- Requires GPU for training
- Limited samples for rare diseases
- ResNet18 incomplete training (needs more GPU memory)

### Future Improvements
- [ ] Deploy as mobile app (TensorFlow Lite)
- [ ] Test on wild/field images
- [ ] Add more crop types (wheat, rice, corn)
- [ ] Grad-CAM visualization (explain what model sees)
- [ ] Real-time inference on edge devices
- [ ] Multi-disease detection (plant can have 2+ diseases)


##  Performance Summary

**PlantCNN achieves 95.49% accuracy**, making it suitable for:
-  Production deployment in agriculture
-  Critical crop protection decisions
-  Multi-crop disease detection

**ResNet18 achieves 82% accuracy**, demonstrating:
-  Efficiency of transfer learning
-  Potential for rapid fine-tuning
-  GPU memory as a practical constraint


- **Dataset:** PlantVillage
- **Architecture:** ResNet (He et al., 2015)
- **Framework:** PyTorch & torchvision

---
