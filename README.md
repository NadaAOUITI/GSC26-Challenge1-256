# Challenge 01 — FL Backdoor Attack & Defense (Team 256)

**Nada (attack lead) · Tassnim (defense lead)**  
Portal: [challenge-01](https://ai-hackathon-two-phi.vercel.app/challenge-01)  
Living research log: [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md) (read this — it has every portal result)

---

## 1. What this challenge is

Two tracks, **equal weight** on the combined score:

| Track | You submit | Goal |
|-------|------------|------|
| **Attack** | `attack_submission.csv` (12 malicious `SmallCNN`s) | Plant trigger → **black hair**, keep clean accuracy high |
| **Defense** | `defense_submission.py` | `robust_aggregation(num_models, models)` that resists backdoors |

```
Attack score   = 0.4 × CleanAcc + 0.6 × ASR
Defense score  = 0.6 × CleanAcc + 0.4 × (1 − ASR)
Combined       = 0.5 × max(attack) + 0.5 × max(defense)
```

**Important:** Portal **Overall Track Score** often mirrors **ASR only**.  
Always compute the **formula** yourself. Optimize formula score, not ASR-rank.

| Case | Benign (given) | Malicious (you) | Trigger |
|------|----------------|-----------------|---------|
| 1 | 8 | 2 | Black sunglasses |
| 2 | 20 | 5 | Surgical mask |
| 3 | 15 | 5 | Black sunglasses |

Portal aggregates **once** (hidden method per case: FedAvg / trimmed mean / Krum / Multi-Krum / median / Bulyan).

Limits: **10 attack / day**, **2 defense / day**. Best score per track is kept.

---

## 2. Frozen keeps (do not overwrite without beating portal formula)

### Attack keep ≈ **0.752**
| Knob | Value |
|------|--------|
| Triggers | Hi-fi drawn sunglasses/mask (`attack/triggers.py` default geometry) |
| Scale γ | **1.5** (after fine-tune) |
| Data | 1400 CelebA hair images |
| Epochs | 6 |
| λ_bd / λ_reg | **2.5** / **10** |
| Scope / poison | full / all |
| Portal | ASR **0.722**, CleanAcc **0.796** |

Recipe command:

```bash
python attack/attack_baseline.py --mode backdoor --scale-factor 1.5 \
  --poison-mode all --train-scope full --data-root data \
  --max-images 1400 --epochs 6 --batch-size 32 \
  --lambda-bd 2.5 --lambda-reg 10 --lr 1e-4 --cases 1,2,3 \
  --output-root participant_models
```

### Defense keep ≈ **0.558**
Still the starter **FedAvg**-like baseline on the portal (ASR ~0.885).  
Coordinate median / Multi-Krum were tried and did **not** beat it.  
**Next defense upload:** `defense_submission_trimmed.py` → copy to `defense_submission.py`.

---

## 3. Repo layout (what matters)

```text
challenge-01-fl-security/
├── README.md                 ← you are here
├── ENGINEERING_NOTES.md      ← experiment log + playbook (source of truth)
├── model.py                  ← SmallCNN (do not change architecture)
├── requirements.txt
├── quickstart.ipynb
├── defense_submission.py     ← current defense file for portal
├── defense_submission_*.py   ← variants (median, multikrum, trimmed)
├── attack/
│   ├── case_{1,2,3}/         ← benign client_*.pt (committed)
│   ├── attack_baseline.py    ← train malicious models
│   ├── triggers.py           ← hi-fi + geometry / jitter / multi-trigger
│   ├── celeba_data.py / download_celeba.py
│   ├── create_attack_submission.py / validate_attack_submission.py
│   ├── eval_post_agg.py      ← local post-aggregation proxy
│   └── sample_submission.csv
├── defense/
│   ├── visible_case/         ← practice models
│   ├── eval_local.py
│   └── test_defense_submission.py
└── utilities/
    ├── aggregators.py        ← FedAvg / median / Multi-Krum / trimmed (local only)
    ├── model_io.py
    └── checks.py
```

**Not in git** (see root `.gitignore`): `data/`, all `participant_models*/`, `attack_submission*.csv`, `attack/sample_submission.csv` (~53MB template — get it from the Kaggle starter), `.venv/`.

If you need `attack/sample_submission.csv` for packing, copy it from the official `challenge_starter/attack/` zip (or keep a local copy; it is gitignored on purpose).

---

## 4. Setup (teammate first clone)

```bash
# from repo root GSC-2026/
cd challenges/challenge-01-fl-security
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python attack/download_celeba.py --max-images 1400
```

CelebA lands under `data/`. Re-run if missing.

---

## 5. Attack workflow

1. Train malicious `.pt` files (keep recipe above, or experiment flags).
2. Pack + validate:

```bash
python attack/create_attack_submission.py --models-root participant_models --output attack_submission.csv
python attack/validate_attack_submission.py --submission attack_submission.csv
```

3. Confirm validator prints `valid`.
4. Upload **`attack_submission.csv`** on the portal.
5. Log ASR / CleanAcc / **formula** in `ENGINEERING_NOTES.md`.
6. Gate: keep only if formula **> 0.752**.

### Useful experiment flags (`attack_baseline.py`)

| Flag | Meaning | Portal outcome (summary) |
|------|---------|---------------------------|
| `--scale-factor 1.5` | Model-replacement γ | **Keep** |
| `--dba` | Distributed trigger parts | Discard ~0.746 |
| `--neurotoxin --neurotoxin-keep-ratio 0.25` | Quiet-param BD grads | Hard fail ~0.579 |
| `--constrain` | Constrain-during-train | Discard ~0.713 |
| `--diversify` | Per-mal train + benign blend | Discard ~0.749 |
| `--trigger-jitter` | Continuous geometry noise | Closest miss ~0.751 |
| `--multi-trigger` | Discrete preset mix | Discard ~0.750 |
| `--lambda-bd 3.5` | Stronger BD loss | Discard ~0.750 |
| `--scale-factor 1.75` | Milder γ↑ | Discard ~0.747 (Clean↓) |

**Playbook:** one axis per portal submit. If it loses, discard and stay on keep.

Local post-agg check (relative only — CelebA CleanAcc proxy is weak):

```bash
python attack/eval_post_agg.py --models-root participant_models \
  --aggregators fedavg,median,multikrum,trimmed_mean
```

---

## 6. Defense workflow

1. Implement `robust_aggregation` in a `defense_submission*.py` file.
2. Allowed imports: **`torch`**, **`numpy`**, Python stdlib only. No file I/O, no network.
3. Validate:

```bash
python defense/test_defense_submission.py --submission defense_submission.py
```

4. Local proxy (optional):

```bash
python defense/eval_local.py --submission defense_submission_trimmed.py \
  --malicious-root participant_models
```

5. Upload **`defense_submission.py`** (copy the variant you want into that name).

| File | Status |
|------|--------|
| `defense_submission.py` | Currently coordinate median (portal ≈ FedAvg) |
| `defense_submission_multikrum.py` | Portal worse than FedAvg |
| `defense_submission_trimmed.py` | **Next to try** — drop farthest `⌊N/3⌋` from median, then mean |

---

## 7. What we already learned (short)

- Local malicious ASR ~1.0 **≠** portal ASR (~0.72). Always think post-aggregation.
- Malicious Δθ is already **~15–20×** median benign — soft “stay near benign” kills ASR.
- γ↑ raises ASR a bit but CleanAcc drops enough to lose formula score; stay at **γ=1.5**.
- Attack recipe family is largely saturated near **0.752**. Biggest combined-score lever left: **defense** (trimmed / Bulyan-lite).
- Full table: [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md) §4.

---

## 8. Portal upload checklist

**Attack**
- [ ] `valid` from `validate_attack_submission.py`
- [ ] File named exactly `attack_submission.csv`
- [ ] After results: compute `0.4×Clean + 0.6×ASR` (ignore Overall if it equals ASR)

**Defense**
- [ ] `valid` from `test_defense_submission.py`
- [ ] File named exactly `defense_submission.py`
- [ ] No forbidden imports

---

## 9. Roles & next actions

| Who | Focus now |
|-----|-----------|
| **Tassnim** | Upload trimmed defense; if it fails, Bulyan-lite |
| **Nada** | Hold attack keep 0.752; only reopen attack with a *new* structural idea |

Questions → update `ENGINEERING_NOTES.md` or ping on the team channel.
