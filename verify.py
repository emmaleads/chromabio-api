"""Check that the service reproduces your model and similarity numbers.

Run once, from this folder, in an environment with requirements.txt installed
(Colab works):   python verify.py
"""
import numpy as np
import pandas as pd

import app as svc

# 1. Reproduce your held-out predictions from test_predictions.csv
t = pd.read_csv("data/test_predictions.csv").sample(300, random_state=0)
X, keep = [], []
for r in t.itertuples():
    f = svc.featurize(r.smiles, r.solvent)
    if f is not None:
        X.append(f); keep.append(r.Index)
p = svc.MODEL.predict(np.vstack(X))
ref = t.loc[keep, "pred"].values
# the saved model is refit on all data, the CSV came from the 80% split model, so expect
# close but not identical values; MAE against the true values should be small.
print(f"1. Rows checked: {len(keep)}")
print(f"   mean |saved model - CSV pred|: {np.abs(p - ref).mean():.2f} nm (small but non-zero is expected: refit on all data)")
print(f"   MAE vs true lambda: {np.abs(p - t.loc[keep, 'true'].values).mean():.1f} nm (in-sample, should be below ~30)")

# 2. Similarities of the demo dyes. Expected from your evaluation output:
expected = {"Fluorescein": 0.31, "Methyl Orange": 0.42, "Methylene Blue": 0.48,
            "Crystal Violet": 0.43, "Sudan IV": 0.40, "Rhodamine B": 0.84}
dyes = {"Fluorescein": "O=C1OC2(c3ccccc13)c1ccc(O)cc1Oc1cc(O)ccc12",
        "Methyl Orange": "CN(C)c1ccc(N=Nc2ccc(S(=O)(=O)[O-])cc2)cc1",
        "Methylene Blue": "CN(C)c1ccc2nc3ccc(=[N+](C)C)cc3sc2c1",
        "Crystal Violet": "CN(C)c1ccc(cc1)C(=C1C=CC(C=C1)=[N+](C)C)c1ccc(cc1)N(C)C",
        "Sudan IV": "Cc1ccccc1N=Nc1ccc(N=Nc2c(O)ccc3ccccc23)c(C)c1",
        "Rhodamine B": "CCN(CC)c1ccc2c(c1)oc1cc(=[N+](CC)CC)ccc1=c2-c1ccccc1C(=O)O"}
print("2. Demo-dye max similarity (service vs your evaluation output):")
ok = True
for n, s in dyes.items():
    out = svc.predict_smiles(s)
    d = out["max_similarity"] - expected[n]
    flag = "ok" if abs(d) <= 0.02 else "MISMATCH"
    ok &= flag == "ok"
    print(f"   {n:15s} service {out['max_similarity']:.2f}  expected {expected[n]:.2f}  {flag}"
          f"   ethanol pred {out['pred_by_solvent']['Ethanol']}  in_training_set={out['in_training_set']}")
print("   -> similarity definition matches your evaluation" if ok else
      "   -> similarity differs: tell me which fingerprint your evaluation used and I will change it")
