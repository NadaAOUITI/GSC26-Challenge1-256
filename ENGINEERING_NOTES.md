# GSC 2026 FL Security — Engineering & Research Log

**Team 256** · Living notebook. Append after every portal result or local experiment.

> How to use: add a dated entry under **Experiment log** or **Situation playbook**
> when you learn something real. One idea per experiment. Beat frozen keeps on
> **formula score**, not ASR-rank.

---

## 1. Challenge in one page

### Task
- **Attack:** submit malicious `SmallCNN` models that plant a backdoor (trigger → **black hair**) while keeping clean accuracy high.
- **Defense:** implement `robust_aggregation(num_models, models)` — weights only, no data, no file I/O.

### Triggers
| Case | Benign | Malicious | Trigger |
|------|--------|-----------|---------|
| 1 | 8 | 2 | Black sunglasses |
| 3 | 15 | 5 | Black sunglasses |
| 2 | 20 | 5 | Surgical mask |

### Scoring (participant guide §13, §24)

```
Attack score  = 0.4 × CleanAcc + 0.6 × ASR
Defense score = 0.6 × CleanAcc + 0.4 × (1 − ASR)
Combined      = 0.5 × max(attack) + 0.5 × max(defense)
```

Portal **Score** column uses these formulas (0–1 scale). **ASR-sorted leaderboard ≠ best strategy** — optimize formula score.

### One-shot aggregation
Portal aggregates benign + your malicious models **once** (FedAvg, Trimmed Mean, Krum, Multi-Krum, Median, or Bulyan — fixed per hidden case). Design for **robust aggregation**, not FedAvg-only.

---

## 2. Frozen keeps (update only when beaten on portal)

### Attack keep
| Field | Value |
|-------|-------|
| Recipe | hi-fi drawn triggers, γ=1.5, 1400 img, 6 ep, λ_bd=2.5, λ_reg=10, full scope, poison=all |
| Portal ASR | 0.722 |
| Portal CleanAcc | 0.796 |
| **Score** | **≈ 0.752** |
| Files | `attack/triggers.py`, `participant_models/` (hi-fi γ=1.5 run) |

### Defense keep (max score retained)
| Field | Value |
|-------|-------|
| Method | FedAvg baseline (starter) |
| Portal ASR | 0.885 |
| Portal CleanAcc | 0.853 |
| **Score** | **≈ 0.558** |
| Note | Multi-Krum (2026-07-20) scored **0.547** — do not use as keep |

---

## 3. Situation playbook

### “Local malicious ASR is ~1.0 but portal ASR is ~0.72”
- **Diagnosis:** aggregation dilution + trigger/eval mismatch — not “train more epochs.”
- **Do:** simulate **post-FedAvg** global model locally; tune γ and malicious diversity (DBA).
- **Don’t:** trust `eval_heldout_asr.py` alone (scores raw malicious model, not aggregate).

### “I want to climb ASR-rank on the leaderboard”
- **Diagnosis:** board is often sorted by ASR; high ASR + low CleanAcc loses on **score**.
- **Do:** maximize `0.4×Clean + 0.6×ASR`.
- **Example:** 0.731 ASR / 0.735 Clean (rank ~8) → score 0.733 **worse** than keep 0.752.

### “Should I increase γ toward n/m (4–5)?”
- **Diagnosis:** CelebA proxy loves high γ; portal CleanAcc crashes.
- **Do:** stay near **γ=1.5** unless post-agg local eval proves otherwise.
- **Don’t:** `--scale-factor auto` or γ=2.5+ without new evidence.

### “Should I try prettier / photoreal triggers?”
- **Result:** photoreal soft overlays → score **0.748** (failed vs 0.752).
- **Do:** geometry search (eye-band, lens size) scored on **post-aggregation** ASR.
- **Don’t:** swap overlay style without beating frozen keep.

### “Defense: Multi-Krum / median didn’t kill the backdoor”
- **Results:** Multi-Krum ASR **0.905** (score 0.547); median ASR **0.895** (score 0.556) ≈ FedAvg (**0.885 / 0.558**).
- **Why:** attackers stay close to the benign cloud; global L2 drop and coord-median still average in poison.
- **Do:** **harder filters** — trimmed-mean (drop `⌊N/3⌋` farthest from median, then FedAvg) or Bulyan-lite; build `defense/eval_local.py` before tomorrow’s slots.
- **Don’t:** re-upload Multi-Krum or plain median expecting a different result.

### “One axis per experiment”
- Change **one** knob; freeze winner recipe.
- Log hypothesis → local proxy → portal → keep or discard.

---

## 4. Experiment log

| Date | Track | Hypothesis | ASR | Clean | Score | Verdict |
|------|-------|------------|-----|-------|-------|---------|
| 2026-07-17 | attack | First real backdoor (CelebA) | 0.655 | 0.823 | 0.722 | Baseline real attack |
| 2026-07-18 | attack | More data/epochs/λ_bd | 0.673 | 0.826 | 0.734 | Good direction |
| 2026-07-18 | attack | Heavy γ (auto n/m) | 0.731 | 0.669 | 0.706 | CleanAcc crash |
| 2026-07-19 | attack | Hi-fi triggers + γ=1.5 | 0.722 | 0.796 | **0.752** | **KEEP** |
| 2026-07-19 | attack | Photoreal overlays (A) | 0.716 | 0.795 | 0.748 | Discard |
| 2026-07-19 | attack | nonblack_boost poison (B) | 0.717 | 0.798 | 0.750 | Discard |
| 2026-07-19 | attack | last_block freeze (C) | 0.708 | 0.805 | 0.747 | Discard |
| 2026-07-19 | attack | Real CelebA eyeglasses mix | 0.699 | 0.811 | 0.744 | Discard |
| 2026-07-19 | defense | Soft cosine reweight | 0.902 | 0.855 | ~0.54 | Worse than FedAvg |
| 2026-07-20 | defense | Multi-Krum | 0.905 | 0.848 | 0.547 | Worse than FedAvg |
| 2026-07-20 | defense | Coordinate median | 0.895 | 0.857 | 0.556 | ≈FedAvg; ASR still ~0.9 |
| 2026-07-20 | lab | Built eval_post_agg + eval_local + trimmed defense | — | — | — | Local: trimmed kills our ASR→0 but CelebA CleanAcc proxy weak |
| 2026-07-20 | attack | DBA partial triggers + γ=1.5 | 0.719 | 0.787 | 0.746 | Discard (< keep 0.752) |
| 2026-07-20 | attack | Geometry search → eyes_lower+mask_high, γ=1.5 | 0.723 | 0.790 | 0.750 | Discard vs keep 0.752; better than DBA |
| 2026-07-20 | attack | Neurotoxin quiet-mask 25% + γ=1.5 default geo | 0.442 | 0.784 | 0.579 | Discard hard — ASR collapsed; do NOT stack |
| 2026-07-20 | attack | Constrain-during-train λ=0.1 + γ=1.5 keep recipe | 0.635 | 0.831 | 0.713 | Discard — Clean↑ ASR↓; do NOT stack |
| 2026-07-20 | attack | λ_bd=3.5 on keep (γ=1.5, λ_reg=10) | 0.721 | 0.793 | 0.750 | Discard — ASR flat, Clean slightly down vs keep 0.752 |
| 2026-07-20 | attack | Diversify: per-mal train + benign mix 0.25 | 0.719 | 0.794 | 0.749 | Discard — no gain vs keep 0.752 |
| 2026-07-20 | attack | Mild γ=1.75 rescale of frozen keep | 0.729 | 0.775 | 0.747 | Discard — ASR↑ but Clean↓ outweighed; keep γ=1.5 |
| 2026-07-20 | attack | Train-time trigger jitter + γ=1.5 keep | 0.725 | 0.789 | 0.751 | Discard — closest miss; ASR↑ tiny, Clean↓; stay on keep |
| 2026-07-20 | attack | Bagdasaryan: λ_reg=20 then γ=1.75 | 0.727 | 0.775 | 0.747 | Discard — same Clean crash as bare γ=1.75; λ_reg armor failed |
| 2026-07-20 | attack | Multi-trigger discrete presets + γ=1.5 | 0.723 | 0.791 | 0.750 | Discard — flat vs keep 0.752 |

*(Add your next row here.)*

---

## 5. What to try next (ranked)

### Defense
1. **Trimmed-mean hybrid** — `defense_submission_trimmed.py` (ready for tomorrow; local-validated)
2. **Bulyan-lite** — Krum keep-set → coord median → mean (if trimmed still fails)
3. Plain median / Multi-Krum — **retired** (portal ASR ~0.895 / 0.905)

### Attack — 3 remaining slots (paper-derived, 2026-07-20)

**Context:** Keep δ is already **~15–20×** median benign δ. Soft constrain / train-and-scale-to-median / Neurotoxin@0.25 all fought that the wrong way. γ↑ raises ASR but drops Clean.

| Slot | Paper | Idea (one axis) | Diff vs failed tries |
|------|-------|-----------------|----------------------|
| ~~1~~ | Bagdasaryan | ~~λ_reg=20 then γ=1.75~~ | **Discarded 0.747** |
| ~~2~~ | Nguyen survey | ~~Discrete multi-trigger~~ | **Discarded 0.750** |
| 3 | Neurotoxin | Mild `keep_ratio=0.5` | Optional last attack shot; prefer trimmed defense first |

Gate each on formula **> 0.752**. If slot 1 fails, do not raise γ further.

### Local lab (built 2026-07-20)
| Script | Role |
|--------|------|
| `attack/eval_post_agg.py` | Score global model after FedAvg / median / Multi-Krum / trimmed_mean |
| `defense/eval_local.py` | Score any `defense_submission*.py` vs benign[+malicious] pool |
| `utilities/aggregators.py` | Shared local aggregators (not for portal file) |

### Retired (do not repeat without new evidence)
- Photoreal overlays, poison mix, partial freeze, real glasses mix
- DBA partial triggers (portal score **0.746** < keep **0.752**)
- Geometry eyes_lower+mask_high (**0.750**)
- Neurotoxin quiet-mask (**0.579** — ASR crashed to 0.442; never stack on this)
- Constrain-during-train (**0.713** — CleanAcc up, ASR down; never stack)
- `train-and-scale` (shrinks poison; local post-FedAvg ASR ~0)
- γ roulette above 1.5 without post-agg validation; γ=1.75 specifically (0.747)
- Trigger jitter (0.751 — closest miss, still discard)
- Soft cosine, Multi-Krum, plain median defenses

---

## 6. Reading list (skim → one implementation idea)

| Resource | Takeaway |
|----------|----------|
| [Bagdasaryan — How To Backdoor FL](https://proceedings.mlr.press/v108/bagdasaryan20a.html) | Model replacement, constrain-and-scale |
| [DBA (ICLR 2020)](https://openreview.net/forum?id=rkgyS0VFvr) | Per-client trigger splits beat identical poison |
| [Neurotoxin (ICML 2022)](https://proceedings.mlr.press/v162/zhang22w.html) | Durable backdoor via gradient masking |
| [FL backdoor survey](https://arxiv.org/abs/2303.02213) | Taxonomy, metrics, defense phases |
| [backdoors101](https://github.com/ebagdasa/backdoors101) | Reference FL backdoor code |
| [fedlearn-backdoor-attacks](https://github.com/mtuann/fedlearn-backdoor-attacks) | Modular attack/defense eval |

**Reading protocol (45 min):** abstract → method figure → which aggregator they test → one sentence: “We test X by changing Y in our repo.”

---

## 7. Commands cheat sheet

```bash
# Attack train (frozen keep recipe)
python attack/attack_baseline.py --mode backdoor --scale-factor 1.5 \
  --max-images 1400 --epochs 6 --lambda-bd 2.5 --lambda-reg 10 \
  --poison-mode all --train-scope full --cases 1,2,3

python attack/create_attack_submission.py
python attack/validate_attack_submission.py

# Defense validate before portal
python defense/test_defense_submission.py --submission defense_submission.py

# Held-out proxy (malicious only — not portal-realistic alone)
python attack/eval_heldout_asr.py --models-root participant_models
```

### Defense variants on disk
| File | Algorithm |
|------|-----------|
| `defense_submission.py` | Coordinate median (portal 2026-07-20: ASR 0.895) |
| `defense_submission_median.py` | Median (source copy) |
| `defense_submission_multikrum.py` | Multi-Krum (portal: ASR 0.905) |
| `defense_submission_trimmed.py` | **Trimmed-mean hybrid — tomorrow slot #1** |

```bash
# Post-aggregation attack proxy (use before portal)
python attack/eval_post_agg.py --models-root participant_models_dba \
  --aggregators fedavg,median,multikrum,trimmed_mean

# Local defense proxy
python defense/eval_local.py --submission defense_submission_trimmed.py \
  --malicious-root participant_models_dba
```

---

## 8. Personal notes (your section)

*Add lessons, paper quotes, mentor advice, and “aha” moments below.*

### Template for new entries

```markdown
#### YYYY-MM-DD — Short title
**Situation:** …
**Learned:** …
**Action next time:** …
```

---

*Last updated: 2026-07-21 (docs + gitignore for teammate submit; keep still 0.752)*
