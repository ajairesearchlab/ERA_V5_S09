# Session 14: Dense model → Mixture-of-Experts

The Session 11 model is trained as a normal dense model, converted into a mixture-of-experts (MoE) by partition and drop-upcycling, and trained further. Over the same 2,000 extra steps, a dense copy of the model is trained on the same batches for comparison.

**Run it:** [session14.ipynb](session14.ipynb) runs everything top to bottom.
Earlier write-ups: [README.md](README.md) (Session 9), [README_S10.md](README_S10.md) (Session 10), [README_S11.md](README_S11.md) (Session 11).

| | |
|---|---|
| Dense model | 8 layers, D=384, 6 heads, context 256, RoPE, RMSNorm, SwiGLU 1,024 wide, **20,453,760** parameters |
| MoE model | Same model, but each block's SwiGLU is replaced by **16 experts × 256 wide, top-4** (4 × 256 = 1,024 = the dense width). **48,814,464** total, **20,502,912** active per token |
| Data | FineWeb-Edu (sample-10BT), own 8K BPE tokenizer, train.bin extended from **20M to 60M tokens** (the old 20M and val.bin reproduced exactly) |
| Optimiser | AdamW (0.9, 0.95), weight decay 0.1 on matrices, grad clip 1.0, bf16 autocast; router in fp32 |
| Batch | 32 sequences × 256 tokens = 8,192 tokens per step |
| Hardware | NVIDIA RTX 5060 Laptop GPU (8 GB), Windows 11 |

## Results at a glance

| # | Task | Result |
|---|---|---|
| 1 | Train dense | 2,000 steps, warmup then flat lr 1e-3: validation **4.355** |
| 2 | Convert | Copy-0 experts reproduce the dense FFN to **6.5e-7**. Loss right after conversion: r=0 **+0.27**, r=0.5 **+1.10** |
| 3 | Keep training | Both MoEs keep learning: **3.981** (r=0) and **3.985** (r=0.5), **0.37 below** the checkpoint. Dense on the same batches: **3.971** |
| 4 | Load balance | MaxVio (median over layers) falls below 0.5 by **step ~24** and ends at **0.10–0.13**. No starved experts |
| 5a | Balancing off | MaxVio rises to **2.9**, **73 of 128** experts starved and **28** get no tokens at all. Loss at step 600 is 0.03 *lower* (see below) |
| 5b | What experts learned | Mostly **token types**: whitespace (lift up to ~3.9×), digits (~3.5×), punctuation (~2.5×); weak for ordinary words (~1.3–1.9×) |

**The assignment's requirement is met.** After conversion, both MoE models keep training and end well below the dense checkpoint they started from. **They do not beat the dense model** at this length: they close most of the gap the conversion opened but finish 0.010–0.014 behind.

## Code

- [model/moe.py](model/moe.py): `MoEFFN` (router, top-k with a bias that only affects the choice, SwiGLU experts, dropless dispatch and combine), `upcycle` (dense → MoE), `param_counts`.
- [train/moe_hook.py](train/moe_hook.py): `MoEBalancer`, a Session 11 `step_hook`. After each step it updates the per-expert biases from the whole batch's load and logs load, MaxVio, starved and zero-token experts, and bias range per layer.
- [data/extend_data.py](data/extend_data.py): extends train.bin with the saved tokenizer, and writes nothing unless the existing data is reproduced token for token.
- [experiments/session14.py](experiments/session14.py): one section per task, `--quick` smoke test, finished sections reused on re-runs.
- [config.py](config.py): adds `MoEConfig`. Nothing else changed; `train/loop.py` and `model/model.py` are untouched.

## Task 1: Dense model

2,000 steps (16.4M tokens), 100 warmup steps, then a constant learning rate of 1e-3: the "stable" part of WSD, with no decay. This follows Session 11 Task 4: a checkpoint taken before the decay can keep training. Validation loss went from 9.09 to **4.355**, at about 154 ms per step.

## Task 2: Conversion

**How the experts are built.** Each block's 1,024-wide SwiGLU is copied 4 times. Each copy's neurons are shuffled and cut into 4 slices of 256 (the gate and up rows plus the matching down columns), giving 16 experts. With r=0.5, half the neurons of every expert are then redrawn at random with the original matrix's standard deviation (drop-upcycling). Attention, norms, embeddings and the head are copied unchanged.

**Router.** One 16 × 384 matrix per layer, fp32, softmax over 16, initial std 0.002 (one tenth of the usual). The 4 chosen scores are rescaled to sum to 1 and multiplied by a routed scale of 1,024 / 256 = **4**. Each expert holds a random quarter of the neurons, so four experts weighted about 1 each cover every neuron once on average, and the expected output matches the dense layer.

| | Dense | MoE |
|---|---|---|
| Total parameters | 20,453,760 | 48,814,464 (2.39×) |
| Active per token | 20,453,760 | 20,502,912 (+49,152: the routers) |
| Training state (16 B / weight) | 0.30 GiB | 0.73 GiB |

**Check.** Summing the 4 experts cut from copy 0, each with weight 1, must give back the dense FFN exactly:

| | Largest difference, relative to the output |
|---|---|
| r = 0 | **6.5e-7** (fp32 rounding: the conversion is correct) |
| r = 0.5 | 0.85 (expected: half of every expert was redrawn) |

**Loss right after conversion (validation, no training):**

| | Validation | vs dense |
|---|---|---|
| Dense checkpoint | 4.355 | |
| MoE r=0 | 4.628 | **+0.27** |
| MoE r=0.5 | 5.451 | **+1.10** |

- **r=0 is right on average but not for each token.** Any 4 experts the router picks are 4 random quarters of the network. They can come from different copies, so they overlap in some neurons and miss others. The output is correct in expectation but noisy token by token, and that costs 0.27.
- **r=0.5 throws away half of what the dense model learned**, so it starts much higher.
- **The routing was uneven from the start.** MaxVio at step 0 was 0.4 in layer 0 and 1.3–2.5 in layers 1–7, so the busiest expert already received 2–3.5× its share before the router had learned anything.

## Task 3: Keep training

Three branches from the step-2,000 checkpoint, each with a fresh AdamW and the **same 2,000 new batches** (16.4M tokens). Schedule: WSD with 100 warmup steps, lr 1e-3 held flat, then a linear decay to 10% over the last 200 steps.

[Continued training](runs/session14/continue.png)

Validation loss:

| Step | Dense | MoE r=0 | MoE r=0.5 | r=0 − dense | r=0.5 − dense |
|---|---|---|---|---|---|
| 0 | 4.355 | 4.628 | 5.451 | +0.272 | +1.095 |
| 100 | 4.299 | 4.413 | 4.539 | +0.114 | +0.240 |
| 500 | 4.257 | 4.332 | 4.380 | +0.075 | +0.123 |
| 1,000 | 4.192 | 4.238 | 4.262 | +0.046 | +0.070 |
| 1,500 | 4.135 | 4.162 | 4.176 | +0.027 | +0.042 |
| 1,800 | 4.105 | 4.127 | 4.135 | +0.022 | +0.030 |
| **2,000** | **3.971** | **3.981** | **3.985** | **+0.011** | **+0.014** |

| Branch | Back below the checkpoint (4.355) at | Final vs checkpoint | ms / step |
|---|---|---|---|
| Dense | step 100 | −0.384 | 172 |
| MoE r=0 | step 500 | −0.374 | 550 |
| MoE r=0.5 | step 600 | −0.370 | 658 |

- **The MoEs train and keep reducing the loss.** Both drop steeply in the first 100 steps, then fall steadily and take the full benefit of the final decay.
- **The gap to dense shrinks the whole way and never reverses.** It goes from 0.27 to 0.011 (r=0) and from 1.10 to 0.014 (r=0.5). At 2,000 steps the 2.4× larger MoE has not yet turned its extra parameters into a lower loss than dense.
- **r=0 leads r=0.5 throughout**, but the lead shrinks from 0.82 at step 0 to 0.004 at the end. The cost of redrawing half the neurons is nearly gone by step 2,000. In the published drop-upcycling results, r=0.5 comes out ahead only after much longer training than this.
- **The dense branch got worse briefly at step 200 (4.299 → 4.317).** Its fresh optimiser restarts the warmup at the full lr of 1e-3, after a phase that had already been running at that lr.
- **The MoE is 3.2–3.8× slower per step**, although the work per token is about the same. My `MoEFFN` runs the 16 experts one after another in a Python loop, with many small matrix products and index operations. A grouped-matmul kernel (as in MegaBlocks) would remove most of this. Measured in wall-clock time, dense is clearly better here.
- One seed. At step 2,000 the gaps (0.010–0.014) are within the 0.01–0.02 run-to-run noise measured in Session 11, so "the MoE matches dense" is as strong a claim as this run supports.

## Task 4: Load balance during training

Balancing: a per-expert bias added to the scores only when choosing the top 4, updated after every step by ±0.001 from the **whole batch's** load (both micro-batches), plus a per-sequence balance loss with weight 1e-4. No tokens are dropped.

[Routing](runs/session14/routing.png)

| | MoE r=0 | MoE r=0.5 |
|---|---|---|
| MaxVio (median layer) stays below 0.5 from | step 24 | step 23 |
| MaxVio at step 2,000, by layer | 0.07–0.23 | 0.06–0.17 |
| Busiest expert at step 2,000, as a multiple of even | 1.07–1.23× | 1.06–1.17× |
| Starved (< 10% of even) or zero-token experts | 0 | 0 |

- **The bias corrects the imbalance within about 25 steps** and keeps every layer within about ±20% of even for the rest of the run. The heatmaps of the last 100 steps are almost uniformly 1.0.
- **Layer 0 is the least even** for r=0 (MaxVio 0.23 at the end). This matches the notes: routing in the first layer settles most slowly.
- The worst layer at any step jumps around 0.2–0.6 from step to step. That is batch noise in an 8,192-token batch, not a trend.

## Task 5a: Balancing switched off

The same r=0.5 conversion, schedule and batches, with the bias frozen at 0 and no sequence loss, for 600 steps.

[No balancing](runs/session14/nobalance.png)

| | Balanced | Off |
|---|---|---|
| MaxVio, median layer, step 600 | 0.12 | **2.91** |
| Busiest expert by layer (× even) | ~1.1 | 2.9–**4.0** |
| Starved experts (of 128) | 0 | **73** |
| Experts with no tokens at all | 0 | **28** |
| Validation at step 600 | 4.352 | **4.321** (−0.031) |

- **The collapse from the notes happens, and quickly.** Starved experts rise steadily from step ~15 to 73 by step 600, and the curve hasn't flattened yet.
- **4.0× even is the ceiling.** With top-4 of 16, an expert that every token picks gets exactly 1/4 of the slots. In 6 of 8 layers the busiest experts sit at 3.9–4.0×, so **nearly every token picks the same 4 experts**.
- **That is why the loss is not worse.** When every token uses the same 4 experts, the layer becomes a fixed 1,024-wide dense FFN, which is the model that trains best at this length (Task 3). The collapsed MoE has quietly turned back into a dense model and pays nothing for routing. In return, 75% of the expert parameters stop learning, so the 2.4× capacity is wasted.
- This doesn't show that balancing hurts. It shows that **at 600 steps the loss can't tell a collapsed MoE from a dense one**. Collapse costs capacity, which would show later or at scale, and in a multi-GPU setup it would cost speed (one GPU doing all the work).
- The no-balance run took 324 ms per step against 657 balanced. The 28 empty experts are skipped and the starved ones do very little, but these runs were on different days, so the timing isn't a clean comparison.

## Task 5b: What the experts learned

Each token in 40 validation batches is put in one of 8 categories by its own string. For each category and expert, **lift** = P(category | expert) / P(category). A lift of 1.0 means no preference, and 4.0 would mean an expert receives nothing but that category.

[Specialisation](runs/session14/specialisation.png)

Strongest preference (MoE r=0.5 after training):

| Category | Layer 0 | Layer 4 | Layer 7 |
|---|---|---|---|
| whitespace / newlines | **3.63** | **3.41** | **3.75** |
| digits | 1.60 | **2.96** | **3.11** |
| punctuation | 1.86 | 2.32 | 2.54 |
| Capitalised words | 1.29 | 1.99 | 2.34 |
| function words (the, of, and…) | 1.54 | 1.92 | 2.19 |
| word start | 1.31 | 1.24 | 1.87 |
| word piece | 1.35 | 1.38 | 1.51 |

- **The experts sort by token type, as the notes report for ST-MoE and Mixtral.** One expert per layer specialises in whitespace in every layer. Digits get their own expert from about layer 2. Ordinary word tokens are spread nearly evenly.
- **Specialisation gets stronger in later layers** for most categories (digits 1.6 → 3.1, Capitalised 1.3 → 2.3).
- **Limit:** the categories are only token types, so this doesn't test for subject-level experts. With balancing over 32 mixed FineWeb-Edu sequences per step, the notes would not expect strong subject experts anyway.

## What this changes for the project

- **Conversion works, and r=0 is the better choice at this budget.** It starts closer and stays ahead; r=0.5 needs a longer run to pay off.
- **Keep the bias balancer.** It is cheap, settles in ~25 steps, and without it 73 of 128 experts starve within 600 steps.
- **A 2,000-step continuation is too short for the MoE to beat dense.** The next test would be a longer phase 2 (5–10K steps), or an MoE trained from scratch at the same active size.
- **Speed is the real cost on this setup.** A grouped expert kernel would be needed before any MoE run is worth it on wall-clock time.

## Reproduce

From the mini_llm folder, with the Session 9 data in data/prepared/:

```
uv sync
uv run python -m data.extend_data                 # once: 20M -> 60M train tokens (~5 min, needs internet)
uv run python -m experiments.session14 --quick    # smoke test, a few minutes
```

Then open session14.ipynb, select .venv as the kernel and **Run All** (about 1 hour on the RTX 5060; keep the laptop plugged in with sleep disabled). Or by section:

```
uv run python -m experiments.session14 --section dense      # Task 1
uv run python -m experiments.session14 --section convert    # Task 2
uv run python -m experiments.session14 --section continue   # Task 3
uv run python -m experiments.session14 --section routing    # Task 4
uv run python -m experiments.session14 --section nobalance  # Task 5a
uv run python -m experiments.session14 --section special    # Task 5b
```

Finished training sections are reused on later runs; add `--fresh` to retrain. Checkpoints go to `checkpoints/s14_*.pt` and are not in the repository.

## Files added in Session 14

```
mini_llm/
├── README_S14.md                # this write-up
├── session14.ipynb              # runs the whole assignment
├── config.py                    # + MoEConfig
├── data/
│   └── extend_data.py           # grow train.bin with the saved tokenizer
├── model/
│   └── moe.py                   # MoEFFN, upcycle, param_counts
├── train/
│   └── moe_hook.py              # MoEBalancer (bias update + routing stats)
├── experiments/
│   └── session14.py             # Tasks 1-5
└── runs/session14/              # dense_phase1.csv, cont_*.csv, moe_stats_*.csv, nobal_*.csv,
                                 # summary.json, continue.png, routing.png, nobalance.png,
                                 # specialisation.png
```
