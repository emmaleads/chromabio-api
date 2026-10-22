---
title: ChromaBio Lambda-Max API
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ChromaBio lambda-max API

`POST /predict` with `{"smiles": "..."}` returns the predicted lambda-max (nm) in water, ethanol,
DMSO and hexane, the maximum Tanimoto similarity to the training set, the confidence tier
(with its error band), and the three nearest training molecules.

- Model: `HistGradientBoostingRegressor` trained by `train_lambda_max.py` (scikit-learn 1.6.1).
- Similarity: Tanimoto, 2048-bit Morgan bit vector, radius 2, against `data/train_table.csv`.
- Tier bands come from scaffold-grouped out-of-fold errors (see the web app's evaluation tab).

## Check it first (Colab or local)

    pip install -r requirements.txt
    python verify.py

## Deploy on a Hugging Face Space

1. Create a new Space, SDK = **Docker**, visibility Public.
2. Upload every file in this folder (including `model/` and `data/train_table.csv`), or push with git.
3. Wait for the build, then open `https://<user>-<space>.hf.space/` — it should return `{"status":"ok", ...}`.
4. Put that URL in `API_URL` at the top of the script in `chromabio.html`.

Free Spaces sleep when idle: open the Space once a few minutes before demoing.
Set `ALLOWED_ORIGINS` (comma-separated) in the Space settings to restrict CORS to your page's origin.

## Quick test

    curl -X POST https://<user>-<space>.hf.space/predict -H 'Content-Type: application/json' \
         -d '{"smiles":"CN(C)c1ccc(N=Nc2ccc(S(=O)(=O)[O-])cc2)cc1"}'
