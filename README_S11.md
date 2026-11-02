## Session 11: Optimizers and Learning-Rate Schedules

The same small decoder-only model as Sessions 9 and 10, now used to look inside the optimiser:
Adam reproduced by hand, bias correction switched off, the update-to-weight ratio logged for every
layer at every step, cosine against WSD stopped early, and a learning-rate sweep at four widths
used to estimate the learning rate for width 4,096.

**Run it:** [session11.ipynb](session11.ipynb) runs everything top to bottom, with all outputs saved.
Earlier write-ups: [README.md](README.md) (Session 9), [README_S10.md](README_S10.md) (Session 10).

| | |
|---|---|
| Model | 8 layers, D=384, 6 heads (head_dim 64), context 256, RoPE, RMSNorm, pre-norm, SwiGLU (1,024), 20,453,760 parameters |
| Data | Same as Session 9: FineWeb-Edu (sample-10BT), 20M train + 0.5M validation tokens, own BPE tokenizer, V = 8,192 |
| Optimiser | AdamW (0.9, 0.95), weight decay 0.1 on matrices only, grad clip 1.0, bf16 autocast |
| Batch | 32 sequences × 256 tokens per optimiser step |
| Hardware | NVIDIA RTX 5060 Laptop GPU (8 GB), Windows 11, PyTorch 2.11 + CUDA 12.8 |

#### The five tasks at a glance

| # | Task | Result |
|---|---|---|
| 1 | Adam by hand | hand vs PyTorch fp64: **identical** (difference 0.0); our AdamW vs PyTorch's on the full model: within 6e-8 |
| 2 | Bias correction off | β₂ 0.999: first step **3.16× too large**, peaks at 6.6×, needs **~3,900 steps** to fade to 1%; training loss ends **0.86 nats worse**. β₂ 0.95: fades in **76 steps** |
| 3 | Update-to-weight ratio | warmup stops changing it at **step 77–129** for embedding, head and queries; for `v_proj`, `o_proj` and `down` it **never** does in 600 steps. Final val loss 5.033 with warmup vs **5.452** without |
| 4 | Cosine vs WSD at step 200 | WSD **5.841**, cosine 5.873, cosine planned for 200 steps 5.966. **Keep WSD** |
| 5 | LR sweep at widths 256–1,024 | best lr ∝ width^**−1.31** (R² 0.95); estimate for 4,096: **~1.2e-4** (range 8e-5 to 1.9e-4), moderate-to-low confidence |

Full outputs: [runs/notebook/](runs/notebook/) (`s11_1_adam.txt` … `s11_5_sweep.txt`). Data and plots: [runs/session11/](runs/session11/).

#### Code
- [train/adam_by_hand.py](train/adam_by_hand.py): Adam for one weight in plain Python floats, no torch. Also the bias-correction factor, the parabola fit for the sweep minima and the log-log fit for the width trend.
- [train/optim.py](train/optim.py): our own AdamW (same arithmetic as PyTorch's, plus a `bias_correction` switch), decay groups, the cosine / WSD / constant schedules, and `UpdateRatio`, a step hook that records ‖Δw‖ / ‖w‖ for every parameter tensor at every step.
- [train/loop.py](train/loop.py): two small hooks added to `run_training`: pass in your own optimiser, and a hook called before and after every step. Session 10 behaviour is unchanged.
- [experiments/session11.py](experiments/session11.py): one section per task; `--quick` for a 3-minute smoke test.

### Task 1: Adam by hand, checked against PyTorch

One weight starting at 1.0, the notes' five gradients, lr 1e-3, betas (0.9, 0.999), eps 1e-8:

| t | g | m | v | m̂ | v̂ | step | w |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.050000 | 0.00025000 | 0.500000 | 0.250000 | −0.00100000 | 0.999000 |
| 2 | 0.40 | 0.085000 | 0.00040975 | 0.447368 | 0.204977 | −0.00098813 | 0.998012 |
| 3 | 0.60 | 0.136500 | 0.00076934 | 0.503690 | 0.256703 | −0.00099414 | 0.997018 |
| 4 | 0.45 | 0.167850 | 0.00097107 | 0.488078 | 0.243132 | −0.00098985 | 0.996028 |
| 5 | 0.55 | 0.206065 | 0.00127260 | 0.503199 | 0.255030 | −0.00099642 | 0.995031 |

| Compared against the hand calculation | Largest relative difference |
|---|---|
| `torch.optim.AdamW`, fp64, no decay | 0 (identical in every column) |
| `torch.optim.AdamW`, fp64, decay 0.1 | 0 (identical) |
| `torch.optim.AdamW`, fp32 | 1.6e-8 in m, 8.7e-8 in v, **1.4e-5 in the step** |
| our `AdamW`, fp64, decay 0.1, and with bias correction off | 0 (identical) |
| our `AdamW` vs PyTorch's on the 20.5M-parameter model, 5 real steps | largest weight difference 6e-8, i.e. 6e-5 of the update |

- **The gradient sets the direction, the learning rate sets the distance.** The gradients range from 0.40 to 0.60, while every step is within 1.2% of lr.
- **fp32 loses most digits in the step, not in m or v.** The step is the difference of two numbers near 1.0, so it keeps only ~5 of fp32's 7 digits. This is why weights are kept in fp32, not bf16 (Session 10, Task 6).
- **Decoupled decay adds exactly −lr × λ × w** = −1.0e-4 at step 1, nothing divided by √v̂.

### Task 2: Bias correction off

[Bias correction plot](runs/session11/bias_correction.png)

**By hand.** Without correction, the step is too large or too small by the factor (1 − β₁ᵗ) / √(1 − β₂ᵗ), which depends only on the step number:

| β₂ | Step 1 | Peak | Within 10% from step | Within 1% from step |
|---|---|---|---|---|
| 0.999 (notes) | **3.16×** too large | 6.57× at step 12 | 1,751 | **3,925** |
| 0.95 (our runs) | **0.45×** (too small) | 1.10× at step 20 | 6 | **76** |

With β₂ = 0.999 the gap *widens* for the first 12 steps before it starts closing: m has caught up after ~10 steps, v needs ~1,000. With β₂ = 0.95 the two averages catch up at similar speeds and the error is small and short-lived.

**On the real model.** 300 steps, constant lr 1e-3 and no warmup (so a tiny warmup lr cannot hide the effect), same weights and batches. Median update-to-weight ratio over all matrices:

| Step | β₂ 0.999 ON | OFF | OFF/ON | formula | β₂ 0.95 ON | OFF | OFF/ON | formula |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.99e-2 | 1.52e-1 | **3.05** | 3.16 | 4.99e-2 | 2.22e-2 | **0.44** | 0.45 |
| 2 | 3.51e-2 | 1.46e-1 | 4.16 | 4.25 | 3.51e-2 | 2.15e-2 | 0.61 | 0.61 |
| 5 | 2.07e-2 | 1.26e-1 | 6.08 | 5.80 | 2.14e-2 | 1.91e-2 | 0.89 | 0.86 |
| 10 | 1.22e-2 | 6.61e-2 | 5.41 | 6.53 | 1.24e-2 | 1.35e-2 | 1.09 | 1.03 |
| 50 | 1.15e-2 | 4.14e-3 | **0.36** | 4.50 | 1.04e-2 | 1.02e-2 | 0.98 | 1.00 |
| 100 | 1.18e-2 | 2.71e-3 | **0.23** | 3.4 | 9.69e-3 | 9.51e-3 | 0.98 | 1.00 |
| 300 | 1.08e-2 | 4.83e-3 | 0.45 | 1.9 | 8.62e-3 | 8.90e-3 | 1.03 | 1.00 |

| | β₂ 0.999 | β₂ 0.95 |
|---|---|---|
| Training loss (EMA) at step 300, ON | 5.801 | 5.809 |
| Training loss (EMA) at step 300, OFF | **6.660** (+0.86) | **5.626** (−0.18) |
| Step sizes back within 10% of ON | never (in 300 steps) | from step **66** |

- **For the first ~5 steps the measured ratio matches the formula** (3.05 vs 3.16, 0.44 vs 0.45). After that the two runs are in different places, so the formula no longer describes the comparison.
- **β₂ 0.999 OFF: oversized early steps cause lasting damage.** At step 5 the OFF run's grad norm jumped to 11.3 (ON: 1.2). After that its steps became *smaller* than ON's (0.23× at step 100) even though the formula says they should still be 3× larger. The most likely reason: that gradient spike's square sits in v, which with β₂ = 0.999 remembers ~1,000 steps, and keeps shrinking every later step. The run is 0.86 nats behind at step 300 and the gap is not closing.
- **β₂ 0.95 OFF ended up better (−0.18).** Its first steps are 0.45×, 0.6×, 0.7×… of the corrected size, which acts like a short built-in warmup. These runs deliberately had no warmup, so OFF effectively got one and ON did not. This doesn't argue for switching correction off; it is Task 3's warmup effect seen from another side.
- **When does it stop mattering?** For the step *sizes*: ~76 steps at β₂ 0.95 and ~3,900 at β₂ 0.999 (in theory), 66 steps measured at β₂ 0.95. For the *loss*: not within 300 steps for either, because any distance gained or lost early stays in the weights.

### Task 3: Update-to-weight ratio for every layer

[Update-to-weight ratio](runs/session11/update_ratio.png)

Two 600-step runs, same weights and batches. Run A warms up over 100 steps; run B starts at the peak lr of 1e-3. From step 101 both follow the identical cosine, so warmup is the only difference. The ratio ‖w_after − w_before‖ / ‖w_before‖ is logged for all 75 parameter tensors at every step ([ratio_warmup.csv](runs/session11/ratio_warmup.csv), [ratio_no_warmup.csv](runs/session11/ratio_no_warmup.csv)).

**Step 1, layer 0:**

| Weight | With warmup | Without warmup |
|---|---|---|
| `o_proj`, `down` (write into the residual stream) | 2.0e-3 | **2.0e-1** (20% of the weight in one step) |
| `q_proj`, `lm_head` | 5.0e-4 | 5.0e-2 |
| `tok_emb` | 2.8e-4 | 2.8e-2 |

`o_proj` and `down` start 4× smaller than the other matrices (std 0.02/√16 instead of 0.02, Session 9's init), but Adam gives every weight about the same absolute step, so their *relative* change is 4× larger.

**When warmup stops changing the ratio** (median over the 8 layers, 11-step rolling median, warmup run within 10% of the no-warmup run from then on):

| Weight kind | Converges from step | Warmup run peaks at step |
|---|---|---|
| `lm_head` | **77** | 97 |
| `tok_emb` | **81** | 95 |
| `q_proj` | **129** | 103 |
| `k_proj` | 417 | 99 |
| `up`, `gate` | 534, 567 | 167, 166 |
| `v_proj`, `o_proj`, `down` | **never** (600 steps) | 173, 104, 89 |

- **The answer to the task is about step 80–130**: the embedding, the head and the queries line up with the no-warmup run within 30 steps of the warmup ending at step 100. Most kinds peak within a few steps of step 100, as expected when the learning rate peaks there.
- **For the layers that write into the residual stream, warmup's effect never fades.** `o_proj` and `down` (and `v_proj`) take ~20–40% larger relative steps in the warmup run all the way to step 600. The no-warmup run's first steps moved those small weights by up to 20%, so they grew larger and every later step is a smaller fraction of them. The two runs are not on the same path again.
- **That difference shows up in the loss: validation 5.033 with warmup, 5.452 without (+0.42 nats).**
- **The ratio is ~10× the notes' 1e-3 target** for most of the run (~1e-2 around step 100–200) and only reaches ~1e-3 as the cosine finishes. At D = 384 the init scale is much larger than at D = 4,096, so the same lr is a larger fraction of each weight (notes, Section 9). Still, together with Task 5 this suggests 1e-3 is on the high side of comfortable for this model.

### Task 4: Cosine vs WSD, stopped at step 200

[Schedules](runs/session11/schedules.png)

Peak lr 1e-3, 30 warmup steps, both decaying to 10% of peak. Cosine and WSD are planned for 300 steps; WSD holds the peak until step 270 and decays linearly over the last 30. A third run is cosine planned for 200 steps from the start. Validation on 20 fixed batches.

| Run | lr at step 200 | Val @ 100 | Val @ 150 | **Val @ 200** | Val @ 300 |
|---|---|---|---|---|---|
| cosine, planned 300 | 3.7e-4 | 6.265 | 6.031 | 5.873 | 5.704 |
| **WSD, planned 300** | 1.0e-3 | 6.251 | 6.022 | **5.841** | **5.538** |
| cosine, planned 200 | 1.0e-4 | 6.272 | 6.060 | 5.966 | – |

- **At step 200, WSD is 0.032 better than cosine** even though it hasn't started decaying. I expected cosine to lead mid-run because its lr is already lower. It didn't: at this short horizon and lr 1e-3, a higher lr makes faster progress (the Task 5 sweep puts the best lr for width 384 near 1.4e-3, above 1e-3).
- **The cosine planned for 200 steps is the worst at step 200 (+0.094 vs cosine-300).** The notes say a run stopped early is worse than one planned for that length. Here it went the other way: planning a short cosine means decaying sooner, and at this lr and length, decaying sooner cost more than it gained.
- **At step 300 WSD's short decay brings it to 5.538, 0.166 below cosine.** The gap was 0.066 at step 250 and grew to 0.166 during WSD's 30-step decay, the "decay suppresses the noise" effect the notes describe.
- **I would keep the WSD model.** It is better at step 200, and its step-200 checkpoint is still at full lr, so it can be decayed for a few dozen steps into a finished model or trained further. A cosine model stopped at step 200 is already cooled and can't usefully be continued. Not measured here: actually branching the decay from WSD's step-200 checkpoint.
- One seed, 300 steps, ~2.5M tokens. The differences are larger than the 0.01–0.02 run-to-run noise, but not by a wide margin at step 200.

### Task 5: Learning-rate sweep at widths 256, 384, 512 and 1,024

[LR sweep](runs/session11/lr_sweep.png)

8 layers and head_dim 64 at every width; FFN ≈ 8/3 × D. Standard parameterisation (init std 0.02 at every width, not muP). 7 learning rates spaced ×2 apart, 600 steps each (4.9M tokens), warmup 100, cosine to 10%, same seed and batches. Score: validation loss at the end. The minimum is a parabola through the best grid point and its two neighbours (in log lr).

Validation loss after 600 steps (**bold** = best on the grid):

| lr | w=256 (10.6M) | w=384 (20.5M) | w=512 (34.1M) | w=1024 (118.0M) |
|---|---|---|---|---|
| 2.5e-4 | 5.675 | 5.395 | 5.218 | 4.979 |
| 5e-4 | 5.324 | 5.139 | 4.994 | **4.906** |
| 1e-3 | 5.131 | **5.011** | **4.939** | 4.962 |
| 2e-3 | 5.067 | 5.015 | 4.973 | 5.145 |
| 4e-3 | **5.027** | 5.021 | 5.115 | 5.616 |
| 8e-3 | 5.119 | 5.370 | 5.526 | 6.181 |
| 1.6e-2 | 5.634 | 6.009 | 6.423 | 6.409 |

| Width | Fitted best lr | Notes' table (SP) |
|---|---|---|
| 256 | **3.5e-3** | 3.0e-3 |
| 384 | **1.4e-3** | – |
| 512 | **1.1e-3** | 1.5e-3 |
| 1,024 | **5.2e-4** | 7.5e-4 |

- **The minimum moves left as width grows, as the notes predict.** The fit through all four is lr* ∝ width^**−1.31** (R² 0.95), slightly steeper than the notes' width^−1. None of the minima sits at the edge of the grid.
- **Width 384 is nearly flat**: 5.011 / 5.015 / 5.021 from 1e-3 to 4e-3. Its fitted minimum is the least certain of the four, and the curve explains why 1e-3 has been a reasonable choice for this model.
- **Wider models are more sensitive to a too-high lr.** At lr 4e-3 width 256 is at its best (5.027) while width 1,024 is already 0.71 nats above its own best. The same absolute lr that suits a small model is far too large for a wide one.
- **Wider is better at the best lr** (5.027 → 5.011 → 4.939 → 4.906), with diminishing returns at this token budget.

**Estimate for width 4,096:**

| Method | lr at 4,096 |
|---|---|
| Log-log fit through all four widths | 7.7e-5 |
| 1/width from width 1,024 | 1.3e-4 |
| Slope of the two widest (512 → 1,024) | 1.2e-4 |
| Notes, Section 12 | 1.9e-4 |

**I would use about 1.2e-4.** The two estimates built from the widest models agree (1.2–1.3e-4). The full fit gives 7.7e-5 because it is pulled by width 256, whose minimum sits highest. Everything falls between 8e-5 and 1.9e-4, a factor of 2.4.

**Confidence: moderate at best, low for any single number.** The run itself passed every check (no minimum at an edge, R² 0.95, its own three estimates within 1.7× of each other). But:
- it extrapolates **4×** beyond the widest model measured;
- one seed, and the loss is flat near the minimum (at width 384 a 4× range of lr is within 0.01 nats), so each minimum is uncertain by roughly ±30–50%;
- 600-step runs favour higher learning rates than a full-length run would;
- in standard parameterisation the best lr also depends on depth and token budget, which would change at 4,096.

The notes' point stands: muP would turn this extrapolation into a direct reading.

**Wall time.** The 28 runs logged 9.4 hours in total, but runs of the same size took 0.6–3.6 minutes when the laptop stayed awake. The rest was the laptop sleeping or hibernating mid-run (single runs logged 25–131 minutes). The losses are unaffected. Keep the laptop plugged in with sleep disabled for long sweeps.

### What this changes for the project
- **Keep AdamW with bias correction and β₂ = 0.95.** Correction off is a lottery that depends on β₂ (Task 2).
- **Keep warmup**: it was worth 0.42 nats at 600 steps, and the residual-stream layers never recovered without it (Task 3).
- **Switch the default schedule to WSD** (Task 4).
- **Peak lr for width 384: about 1.4e-3**, with 1e-3 to 2e-3 all within 0.005 nats (Task 5).

### Reproduce

From the mini_llm folder, with the Session 9 data already in data/prepared/:

```
uv sync
uv run python -m experiments.session11 --quick      # smoke test, about 3-4 minutes
```

Then open session11.ipynb, select .venv as the kernel and **Run All**. Without sleep interruptions it takes about 1.5 hours, mostly the sweep. Set Windows to never sleep while plugged in. Or run by section:

```
uv run python -m experiments.session11 --section adam       # Task 1
uv run python -m experiments.session11 --section bias       # Task 2
uv run python -m experiments.session11 --section ratio      # Task 3
uv run python -m experiments.session11 --section schedule   # Task 4
uv run python -m experiments.session11 --section sweep      # Task 5 (resumable)
```

The sweep appends each finished run to runs/session11/sweep.csv. If it's interrupted, running it again continues from where it stopped (`--fresh-sweep` starts over). Losses can differ by about 0.01–0.02 between runs because some GPU kernels are not bit-for-bit deterministic.

### Files added in Session 11

```
mini_llm/
├── README_S11.md            # this write-up
├── session11.ipynb          # runs the whole assignment, outputs saved
├── train/
│   ├── adam_by_hand.py      # Adam in plain Python + small fitting helpers
│   ├── optim.py             # AdamW with bias-correction switch, schedules, UpdateRatio
│   └── loop.py              # + opt= and step_hook= in run_training
├── experiments/
│   └── session11.py         # Tasks 1-5
└── runs/
    ├── notebook/            # s11_1_adam.txt ... s11_5_sweep.txt
    └── session11/           # bias_model.csv, ratio_*.csv, ratio_summary.json, schedules.csv,
                             # sweep.csv, sweep_summary.json, and the four plots
```
