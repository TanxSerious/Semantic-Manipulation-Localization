# Semantic Manipulation Localization

This project provides training and testing scripts for semantic manipulation localization.

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