# Semantic-Aware Manipulation Localization for Trustworthy Multimedia Content Understanding

This project provides training and testing scripts for semantic manipulation localization.

## Abstract

Recent advances in generative editing have made multimedia images increasingly easy to modify at the semantic level. Unlike conventional manipulations that leave detectable low-level artifacts, semantic edits may only alter a small attribute, state, or relationship of an object while preserving strong visual consistency with the surrounding content. Such edits can substantially change image interpretation, posing a new challenge to trustworthy multimedia content understanding. In this paper, we study Semantic Manipulation Localization (SML), which aims to localize fine-grained meaning-altering edits in multimedia images. To support this task, we construct a dedicated benchmark through a semantics-driven manipulation pipeline, where semantically decisive regions are identified, edited, and annotated with pixel-level masks. We further propose TRACE, a semantic-aware localization framework that progressively models semantic sensitivity from three aspects. First, a semantic anchoring module grounds localization in meaning-carrying image regions. Second, a semantic perturbation sensing module injects frequency-domain cues to capture subtle visual changes under strong content consistency. Third, a semantic-constrained reasoning module verifies candidate regions through joint reasoning over manipulated content and valid semantic scope. Extensive experiments demonstrate that TRACE consistently outperforms representative image manipulation localization methods on the proposed benchmark, producing more complete, compact, and semantically coherent localization results. These results suggest that semantic-aware localization is essential for multimedia content integrity analysis in the era of generative editing.

## Datasets and Results

- **SML Dataset：** [IEEE Dataport (DOI: 10.21227/7d02-j376)](https://dx.doi.org/10.21227/7d02-j376)
- **Predicted results by our Trace Model：** [Google Drive — final res.zip](https://drive.google.com/file/d/1vhvhJ2TUzmcnVaooMG8pjkkC1kHlujpX/view?usp=sharing)

---

## 1) Download checkpoint

Create a checkpoint directory and place the SAM3 checkpoint file as `sam3.pt`:

```bash
mkdir -p checkpoint
# Put the downloaded file here:
# checkpoint/sam3.pt
```

> Make sure the config/script points to this exact file path.

---

## 2) Set up environment

Recommended with Conda:

```bash
conda create -n SML python=3.12 -y
conda activate SML
pip install -r requirements.txt
```

---

## 3) Prepare `./data` folder

Use the following structure (or update config paths accordingly):

```text
data/
├── train/
│   ├── image/    # training images
│   ├── mask/     # training masks
│   └── edge/   # optional edge labels
├── val/
│   ├── img/    # validation images
│   └── mask/     # validation masks
└── test/
    ├── img/    # test images
    └── mask/     # test masks
```

If your folder names differ, edit the config files in `./configs`.

---

## 4) Edit files in `./configs`

Before running, update paths and key parameters in your selected config file:

- Dataset paths (`train/val/test`)
- Checkpoint path (`./checkpoint/sam3.pt`)
- Save/output directory
- Batch size / epochs / learning rate / image size
- GPU-related settings (if needed)

Also confirm that `train.sh` / `test.sh` reference the correct config file.

---

## 5) Train

Run from project root:

```bash
bash train.sh
```

---

## 6) Test

Run from project root:

```bash
bash test.sh
```

---

## Quick checks

- If you see `FileNotFoundError` for config: fix config path in `train.sh` / `test.sh`.
- If checkpoint shape mismatch warnings appear: model head settings and checkpoint may differ; training can still run if intended.
