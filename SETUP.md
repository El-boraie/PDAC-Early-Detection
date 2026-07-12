# Environment Setup — Run This Before Any Other Day 1 Work

This exists because `requirements.txt` alone is not enough to start: it can't install
PyTorch correctly (needs your specific CUDA version), can't create the environment
itself, and can't handle the one interactive login step (Weights & Biases).
Follow these steps in order.

## Step 1 — Check your CUDA version

Open PowerShell and run:
```powershell
nvidia-smi
```
Look at the top-right of the output for something like `CUDA Version: 12.x`. Write down
that number — you need it for Step 3.

If this command fails or shows no GPU, your NVIDIA driver isn't installed/updated —
install it from nvidia.com before continuing. Training will silently fall back to CPU
otherwise, which will make Day 2's U-Net training unusably slow.

## Step 2 — Create the environment

Using conda (matches your existing `fyp_env` kernel):
```powershell
conda create -n fyp_env python=3.13 -y
conda activate fyp_env
```

## Step 3 — Install PyTorch with the matching CUDA build

Go to https://pytorch.org/get-started/locally/ and use the selector: Stable, Windows,
Pip, Python, and your CUDA version from Step 1. It will generate a command like:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
**Use the exact command the selector gives you** — the CUDA suffix (`cu121`, `cu124`,
etc.) must match your driver, guessing wrong silently gives you a CPU-only build with
no error message.

Verify it worked before moving on:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`torch.cuda.is_available()` must print `True`. If it prints `False`, stop and fix this
before continuing — don't proceed to Step 4 with a broken GPU setup.

## Step 4 — Install everything else

```powershell
pip install -r requirements.txt
```

## Step 5 — Freeze the working versions

```powershell
pip freeze > requirements-lock.txt
```
Keep this file — it's your reproducible, verified record of exactly what worked,
useful for your appendix and for recovering if something breaks later.

## Step 6 — Weights & Biases login (one-time, interactive)

```powershell
wandb login
```
This opens a browser prompt for your API key (free account at wandb.ai if you don't
have one). This step cannot be scripted/automated — it needs you to paste the key once.

## Step 7 — Smoke test

Run this to confirm the full stack imports cleanly before Claude Code starts building
anything on top of it:
```powershell
python -c "
import torch, torchvision, xgboost, sklearn, imblearn
import SimpleITK, nibabel, pydicom, cv2, skimage
import pytorch_grad_cam, shap, captum, lime
import numpy, pandas, scipy, matplotlib, seaborn
import wandb, streamlit
print('All imports OK')
print('CUDA available:', torch.cuda.is_available())
"
```
If anything errors here, fix it now — this is a five-minute check versus a confusing
failure three files deep into Day 1's data loading pipeline.

---

**Once all seven steps pass, the environment is actually ready** — this is the point
where Day 1 of the compressed schedule (repo scaffolding + CT data loading) should
start, not before.
