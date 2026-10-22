"""ChromaBio lambda-max API.

POST /predict  {"smiles": "..."}  ->  per-solvent lambda-max, confidence tier, neighbours.

The featurisation below is copied from train_lambda_max.py so that inputs match
training exactly (2048-bit Morgan count fingerprint, radius 2, + 9 descriptors).
Similarity (which sets the tier) is Tanimoto on a 2048-bit Morgan bit vector,
radius 2, against the unique training molecules in data/train_table.csv.
Run verify.py to confirm this reproduces your numbers.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).parent
MODEL_PATH = HERE / "model" / "lambda_max_model_3.joblib"
TRAIN_PATH = HERE / "data" / "train_table.csv"
MAX_SMILES_LEN = 300
MAX_HEAVY_ATOMS = 150

# ---- copied from train_lambda_max.py -------------------------------------------------
EPS = {"O": 80.1, "CO": 32.7, "CCO": 24.5, "CC#N": 37.5, "ClCCl": 8.9, "ClC(Cl)Cl": 4.8,
       "ClC(Cl)(Cl)Cl": 2.2, "Cc1ccccc1": 2.4, "c1ccccc1": 2.3, "C1CCOC1": 7.5, "CS(C)=O": 46.7,
       "CN(C)C=O": 36.7, "C1CCCCC1": 2.0, "CCCCCC": 1.9, "CCCCCCC": 1.9, "CCOC(C)=O": 6.0,
       "C1COCCO1": 2.2, "CC(C)=O": 20.7, "CC(C)O": 17.9, "CCOCC": 4.3, "CCCCO": 17.5, "CCCO": 20.3,
       "CCCCOCCCC": 3.1, "CC1CCCCC1": 2.0, "OCCO": 37.7, "CCCCCCCCO": 10.3, "OCC(F)(F)F": 26.7,
       "c1ccncc1": 12.4, "CCCCCCO": 13.3, "CC(C)(C)O": 12.5, "OCC(O)CO": 42.5, "CC1CCCO1": 6.2,
       "CCCCCO": 13.9, "NC=O": 111.0, "CCN(CC)CC": 2.4, "CC(=O)O": 6.2, "gas": 1.0}
NAMES = {"water": "O", "methanol": "CO", "ethanol": "CCO", "acetonitrile": "CC#N", "dcm": "ClCCl",
         "chloroform": "ClC(Cl)Cl", "toluene": "Cc1ccccc1", "benzene": "c1ccccc1", "thf": "C1CCOC1",
         "dmso": "CS(C)=O", "dmf": "CN(C)C=O", "cyclohexane": "C1CCCCC1", "hexane": "CCCCCC",
         "ethyl acetate": "CCOC(C)=O", "dioxane": "C1COCCO1", "acetone": "CC(C)=O"}
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def solvent_eps(solvent):
    s = str(solvent).strip()
    s = NAMES.get(s.lower(), s)
    return EPS.get(s)


def featurize(smiles, solvent="CCO"):
    mol = Chem.MolFromSmiles(smiles)
    eps = solvent_eps(solvent)
    if mol is None or eps is None:
        return None
    f_eps = (eps - 1) / (2 * eps + 1)
    desc = [sum(b.GetIsConjugated() for b in mol.GetBonds()),
            rdMolDescriptors.CalcNumAromaticRings(mol), Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol), Descriptors.MolWt(mol), Chem.GetFormalCharge(mol),
            mol.GetNumHeavyAtoms(), eps, f_eps]
    return np.concatenate([_gen.GetCountFingerprintAsNumPy(mol), desc]).astype(np.float32)
# ---------------------------------------------------------------------------------------

TIERS = [  # same thresholds and error bands as the web page (from scaffold-grouped OOF errors)
    {"key": "high", "min": 0.6, "label": "High confidence", "band_nm": 37, "band90_nm": 53, "median_nm": 18},
    {"key": "medium", "min": 0.3, "label": "Medium confidence", "band_nm": 54, "band90_nm": 75, "median_nm": 27},
    {"key": "low", "min": 0.0, "label": "Low confidence", "band_nm": 89, "band90_nm": 123, "median_nm": 46}]


def tier_of(sim):
    return next(t for t in TIERS if sim >= t["min"])


DEFAULT_SOLVENTS = {"Water": "O", "Ethanol": "CCO", "DMSO": "CS(C)=O", "Hexane": "CCCCCC"}

# ---- load model and training set once at start-up ------------------------------------
_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"] if isinstance(_bundle, dict) else _bundle
_train = pd.read_csv(TRAIN_PATH)
TRAIN_MOLS = sorted(_train["can"].unique())
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TRAIN_FPS = [_fp_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in TRAIN_MOLS]
TRAIN_INDEX = {s: i for i, s in enumerate(TRAIN_MOLS)}
_ROWS = {s: g for s, g in _train.groupby("can")}


def similarity_and_neighbours(mol, solvent_smiles, k=3):
    fp = _fp_gen.GetFingerprint(mol)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, TRAIN_FPS))
    top = np.argsort(-sims)[:k]
    nb = []
    for i in top:
        smi = TRAIN_MOLS[i]
        g = _ROWS[smi]
        same = g[g["solvent"] == solvent_smiles]
        lam = float(same["y"].median()) if len(same) else float(g["y"].median())
        nb.append({"smiles": smi, "similarity": round(float(sims[i]), 2),
                   "lambda_nm": round(lam, 1), "same_solvent": bool(len(same))})
    w = np.array([n["similarity"] for n in nb])
    lookup = float(np.dot(w, [n["lambda_nm"] for n in nb]) / w.sum()) if w.sum() > 0 else None
    return float(sims.max()), nb, lookup


def predict_smiles(smiles, solvents=None):
    smiles = (smiles or "").strip()
    if not smiles or len(smiles) > MAX_SMILES_LEN:
        raise ValueError(f"SMILES must be 1-{MAX_SMILES_LEN} characters.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise ValueError(f"Molecule too large (max {MAX_HEAVY_ATOMS} heavy atoms).")
    solvents = solvents or DEFAULT_SOLVENTS
    X, names = [], []
    for name, sv in solvents.items():
        f = featurize(smiles, sv)
        if f is None:
            raise ValueError(f"Unknown solvent: {sv}. Allowed: {sorted(NAMES) + sorted(EPS)}")
        X.append(f); names.append(name)
    preds = MODEL.predict(np.vstack(X))
    pred_by_solvent = {n: round(float(p), 1) for n, p in zip(names, preds)}
    can = Chem.MolToSmiles(mol)
    sim, nb, lookup = similarity_and_neighbours(mol, "CCO")
    tier = tier_of(sim)
    in_train = can in TRAIN_INDEX
    out = {"smiles": can, "pred_by_solvent": pred_by_solvent, "max_similarity": round(sim, 2),
           "tier": tier, "neighbors": nb,
           "lookup_nm": round(lookup, 1) if lookup else None,
           "lookup_method": "Similarity-weighted nearest-neighbour estimate (k=3)",
           "in_training_set": in_train, "literature_nm": None}
    if in_train:
        g = _ROWS[can]
        out["measured_in_training"] = {r.solvent: float(r.y) for r in g.itertuples()}
    return out


app = FastAPI(title="ChromaBio lambda-max API", version="1.0")
_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


class Req(BaseModel):
    smiles: str


@app.get("/")
def root():
    return {"service": "chromabio-lambda-max", "status": "ok", "train_molecules": len(TRAIN_MOLS)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Req):
    try:
        return predict_smiles(req.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
