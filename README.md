# GSC 2026 — Challenge 01: Secure Federated Learning

**Team 256** · IEEE Computer Society Global Student Challenge  


| Link                 | URL                                                                              |
| -------------------- | -------------------------------------------------------------------------------- |
| Portal (submissions) | [challenge-01](https://ai-hackathon-two-phi.vercel.app/challenge-01)             |
| Kaggle starter       | [FL Security Challenge](https://www.kaggle.com/) *(see portal for current link)* |
| GitHub               | [GSC26-Challenge1-256](https://github.com/NadaAOUITI/GSC26-Challenge1-256)       |
| Research log         | `[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)`                                   |


---

## What we built

In federated learning, honest clients train a shared model. An attacker submits **malicious client updates** that hide a **backdoor**: a visual trigger (sunglasses or mask) on face images forces the global model to predict **black hair**, while normal images still look accurate.

We implement **both sides** of this challenge:


| Track       | Deliverable                                              | Our approach                                                                |
| ----------- | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Attack**  | `attack_submission.csv` — 12 poisoned `SmallCNN` weights | CelebA fine-tune + hi-fi synthetic triggers + model-replacement scaling (γ) |
| **Defense** | `defense_submission.py` — `robust_aggregation()`         | Coordinate median, Multi-Krum, and **trimmed-mean hybrid** variants         |


The portal aggregates benign + malicious models **once** per hidden case (FedAvg, trimmed mean, Krum, Multi-Krum, median, or Bulyan). We train and evaluate with **post-aggregation** proxies, not raw malicious-model ASR alone.

---



## Architecture

Challenge 01 system overview

*Three FL cases → attack pipeline (train, scale, pack CSV) and defense pipeline (*`robust_aggregation`*) → portal evaluation (ASR + clean accuracy).*

Text-only overview (if image does not load)

```
  [Benign clients] ──┐
                     ├──► Server aggregation (hidden rule) ──► Global model ──► Portal metrics
  [Malicious .pt]  ──┘         ▲
                               │
                    defense_submission.py (optional robust aggregator)
                               │
  Attack path: CelebA + triggers → fine-tune SmallCNN → γ scaling → attack_submission.csv
```



---



## Results (portal)

Scoring formulas (optimize these, not ASR-only leaderboard sort):

```
Attack  = 0.4 × CleanAcc + 0.6 × ASR
Defense = 0.6 × CleanAcc + 0.4 × (1 − ASR)
Combined = 0.5 × best_attack + 0.5 × best_defense
```



### Attack — current keep (2026-07-25)


| Metric            | Value                                                            |
| ----------------- | ---------------------------------------------------------------- |
| ASR               | **0.716**                                                        |
| Clean accuracy    | **0.804**                                                        |
| **Formula score** | **~0.751**                                                       |
| Recipe            | Hi-fi triggers, γ=1.5, 1400 images, 6 epochs, λ_bd=2.5, λ_reg=10 |


Higher γ and per-case scaling grids **increased ASR but crashed clean accuracy** on the portal (formula dropped to ~0.69). See `[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)` for the full experiment table.

### Defense — status


| Variant                                     | Portal note                           |
| ------------------------------------------- | ------------------------------------- |
| `defense_submission.py` (coordinate median) | ASR ~0.895, score ~0.556              |
| `defense_submission_trimmed.py`             | Ready to upload — trimmed mean hybrid |


---



## Quick start

```powershell
git clone https://github.com/NadaAOUITI/GSC26-Challenge1-256.git
cd GSC26-Challenge1-256
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python attack/download_celeba.py --max-images 1400
python attack/generate_sample_submission.py   # if sample CSV not present locally
```

**Reproduce our attack keep:**

```powershell
python attack/attack_baseline.py --mode backdoor --scale-factor 1.5 `
  --poison-mode all --train-scope full --data-root data `
  --max-images 1400 --epochs 6 --batch-size 32 `
  --lambda-bd 2.5 --lambda-reg 10 --lr 1e-4 --cases 1,2,3 `
  --output-root participant_models

python attack/create_attack_submission.py --models-root participant_models --output attack_submission.csv
python attack/validate_attack_submission.py --submission attack_submission.csv
```

**Validate defense:**

```powershell
python defense/test_defense_submission.py --submission defense_submission.py
```

---



## Repository layout

```text
├── README.md                      ← you are here
├── ENGINEERING_NOTES.md           ← experiment log (portal results, playbook)
├── model.py                       ← SmallCNN (fixed architecture)
├── defense_submission.py          ← portal defense file
├── defense_submission_*.py        ← median, multikrum, trimmed variants
├── attack/
│   ├── attack_baseline.py         ← backdoor training + scaling
│   ├── triggers.py                ← sunglasses / mask geometry
│   ├── create_attack_submission.py / validate_attack_submission.py
│   ├── eval_post_agg.py           ← post-aggregation local proxy
│   ├── export_scaled_models.py    ← γ sweep without retraining
│   └── case_{1,2,3}/              ← benign starter weights
├── defense/
│   ├── test_defense_submission.py
│   └── eval_local.py
└── utilities/
    ├── aggregators.py             ← local-only FedAvg / Krum / trimmed mean
    ├── model_io.py
    └── checks.py
```

Heavy artifacts are gitignored: `data/`, `participant_models*/`, `attack_submission*.csv`, `.venv/`.

---



## Cases & triggers


| Case | Benign | Malicious | Trigger          |
| ---- | ------ | --------- | ---------------- |
| 1    | 8      | 2         | Black sunglasses |
| 2    | 20     | 5         | Surgical mask    |
| 3    | 15     | 5         | Black sunglasses |


---



## License & contact

GSC 2026 Team 256. 