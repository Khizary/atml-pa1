# PA1: Inductive Biases, Domain Adaptation, Generalization, and Open Set Recognition

**Advanced Topics in Machine Learning - Fall 2026**
Muhammad Khizar (28100118@lums.edu.pk)

## Overview

This assignment covers four experiments on inductive bias, domain adaptation, domain generalization, and open set recognition.
Specifically, we address:

1. **Task 1 - Inductive Biases:** What visual features do ResNet50, ViT-B/16, and CLIP actually rely on? We use controlled interventions (grayscale, hue rotation, cue conflicts, translation, patch shuffling) on STL-10 to probe shape, texture, color, and spatial biases. We also check whether prediction level changes correlate with representation level changes using cosine stability and UMAP.

2. **Task 2 - Unsupervised Domain Adaptation:** We compare ERM, DAN, DANN, and CDAN on PACS (Photo, Art, Cartoon → Sketch) with access to unlabeled target data. DAN and CDAN both improve target accuracy by about 19 points over ERM, while DANN suffers catastrophic optimization failure. A λ_MMD sweep shows the alignment-accuracy tradeoff.

3. **Task 3 - Domain Generalization:** Without any target data, we compare source-domain alignment (DAN-DG) and sharpness-aware minimization (SAM). SAM achieves the best Sketch accuracy (+22.7 F1 over ERM), outperforming DAN-DG and even DAN from Task 2 which had access to target data. The ρ sweep reveals a sharp cliff where perturbation becomes too aggressive.

4. **Task 4 - Open Set Recognition:** We evaluate post-hoc novelty scores (MSP, MLS, Energy, Mahalanobis) on CIFAR-10/100, then compare Vanilla, GCSC (RandAugment), and PROSER. GCSC provides the best unknown rejection, supporting the finding that a strong closed set classifier is all you need. PROSER's placeholder classifiers underperformed expectations.

## Key Findings

- All models showed >97% shape bias under AdaIN cue conflicts, but this likely reflects AdaIN's failure to destroy shape cues rather than genuine shape preference. The cosine stability scores are more informative than the prediction level results here.
- There is a constant tension between invariance and discriminability. Stronger alignment helps up to a point, but beyond that it destroys the signal the classifier needs (DAN λ=10 collapses to 5.6% accuracy, SAM ρ=0.1 collapses to 19% F1).
- Flat minima (SAM) generalised better than domain-invariant features (DAN-DG) for Photo→Sketch, even without any target information.
- Strong features from wide augmentation (GCSC) outperformed specialized training objectives (PROSER) for open set recognition.

## Repository Structure

```
├── Submission/
│   ├── pa1t1.ipynb              # Task 1: Inductive biases (STL-10)
│   ├── pa1t2.ipynb              # Task 2: Domain adaptation (PACS)
│   ├── pa1t3.ipynb              # Task 3: Domain generalization (PACS)
│   ├── pa1t4.ipynb              # Task 4: Open set recognition (CIFAR-10/100)
│   ├── pa1t1-alt.ipynb          # Alternate Task 1 (Gatys-style attempt)
│   └── experiment_pa1t1_LBFG.ipynb  # Task 1 experiment with L-BFGS optimizer
├── outputs/
│   ├── task1/                   # Figures, results, logs for Task 1
│   ├── task2/                   # Figures, results, checkpoints for Task 2
│   ├── task3/                   # Figures, results, checkpoints for Task 3
│   └── task4/                   # Figures, results, checkpoints for Task 4
├── report/
│   └── report.tex               # LaTeX source for the report
├── ATML-PA1.pdf                 # Compiled report
└── README.md
```

## Setup and Reproduction

All notebooks were run on Kaggle with GPU acceleration. The main dependencies are PyTorch, torchvision, open_clip_torch, scikit-learn, umap-learn, and standard scientific Python packages. Each notebook installs its own dependencies via `pip install` in the first cell.

Datasets:
- **STL-10** (Task 1) - downloaded automatically via torchvision
- **PACS** (Tasks 2, 3) - loaded from Kaggle datasets
- **CIFAR-10/100** (Task 4) - downloaded automatically via torchvision

All experiments use seed `6304` for reproducibility. Pretrained checkpoints are saved in `outputs/taskN/checkpoints/`.

## Notes

DANN's performance collapse in Task 2 is a known issue. The reversed domain gradient overwhelmed the classification gradient, collapsing the feature extractor to a single class. This is an optimization instability that could likely be resolved with gradient clipping or learning rate scheduling, but I did not have time to run those experiments. Similarly, I attempted to address the AdaIN limitation in Task 1 with an alternate Gatys-style implementation, but the results were not meaningfully different with the compute budget available on Kaggle.

Analysis and discussion of results is in the report (`ATML-PA1.pdf`), not in the notebooks.