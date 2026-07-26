# Quaternary Language Models — Research Log

Discrete (2-bit / quaternary) neural networks applied to language modeling on
WikiText-2, in the style of `qnn2.cpp`. The goal is to see how far a language
model built entirely from **4-level weights** and a **non-gradient, argmax/Hebbian
update rule** can go — and to understand the failure modes that show up when you
throw away real-valued weights and backprop.

---

## 1. The core method

Every weight is one of **four levels `{0, 1, 2, 3}`** (2 bits). There is no
floating point in the model and no gradient descent.

### Weights and accumulators
Each weight is backed by a persistent **integer accumulator**. The weight is just
*which band of the accumulator's range* it currently sits in:

```
 accumulator:  -T ......... -T/2 ......... 0 ......... +T/2 ......... +T
               └── w=0 ───┘└── w=1 ───┘└── w=2 ───┘└── w=3 ───┘
                (bottom rail)                          (top rail)
```

- `T` = **threshold** = global weight inertia.
- Level boundaries at `-T/2, 0, +T/2`; the accumulator clamps at `±T`.
- The accumulator **does not reset** — it's a live position, integrating pushes
  over many steps (like a momentum buffer). A weight "flips" when its accumulator
  crosses a boundary.

### Primitives (from qnn2.cpp)
- `qpolarity(v) = 2v - 3` maps `{0,1,2,3}` → `{-3,-1,1,3}`.
- `qscore(input, weight)` — an integer "product": rewards same-side polarized
  agreement, penalizes disagreement.
- **Forward**: pre-activation `= Σ_i qscore(input_i, weight_i)`, normalized by a
  per-neuron EMA of its mean/deviation (`run_mean`/`run_dev`, decay 1/64), then
  quantized to an activation level.

### Updates (no gradients)
- A **sign/Hebbian push** into the accumulators. The LM head contrasts the target
  class against the argmax (predicted-wrong) class. Q/K learn via a Hebbian router
  surrogate (no softmax derivative).
- This greedy, ~1-bit-per-weight rule is the source of most of the interesting
  failure modes below.

### Metrics — always report both
1. **argmax next-token accuracy** (top-1, 1000-class BPE vocab). Random ≈ 0.1%.
2. **Calibrated cross-entropy / perplexity**. Integer scores aren't logits, so eval
   fits the best softmax temperature `T` and reports CE (nats) + perplexity.
   Random = `ln(1000)` = 6.908 nats, ppl 1000.

> **Methodology warning:** accuracy-only checkpointing selects *uncalibrated
> spikes* (where the fitted `T` rails and the score magnitudes carry no probability
> info). **Always co-report CE.** The CE metric is what caught several false
> "breakthroughs."

---

## 2. Files

| File | What it is |
|---|---|
| `qllm.cpp` | Baseline quaternary LM (embedding + MLP head). ~2.0% acc. |
| `qllm_attn.cpp` | Attention wired in, **per-position** design. Best small config ~2.8%. Reference. |
| `qllm_attn_xf.cpp` | **Main file.** Interleaved transformer (attn+FFN blocks). All experiments this session are argv-gated knobs on this one source. |
| `fourbit/qllm_attn_4bit.cpp` | 4-bit (L-level) generalization; `bits=2` is bit-identical to xf. |

**Executables** are all built from `qllm_attn_xf.cpp`, each adding knobs. Every knob
defaults to OFF and is **bit-identical to the base** in that state (verified per
build):

| exe | adds |
|---|---|
| `qllm_attn_xf.exe` | base (argv 1–13) |
| `qllm_attn_xf_trust.exe` | per-neuron trust region (argv 14) |
| `qllm_attn_xf_tele.exe` | per-component telemetry + eval-override + leak + reset (argv 15–19) |
| `qllm_attn_xf_cap.exe` | + reset-fire-div + clamp% (argv 19–20) |
| `qllm_attn_xf_gate.exe` | + saturation-gate mode (argv 21) |
| `qllm_attn_xf_batch.exe` | + multi-window minibatch (argv 22) — **the full stack** |

### argv reference (qllm_attn_xf_batch.exe = all of them)
```
1  iters                 8  context length        15 telemetry (0/1)
2  textfile              9  threshold             16 eval-interval override
3  blocks               10  threshold-ramp iters  17 leak divisor
4  N (neg asymmetry)    11  boost_max             18 reset-on-flip (0/1)
5  n_embed              12  positions_per_step    19 reset fire divisor
6  mlp_layers (1=1:1)   13  init (0/1{1,2}/2)     20 clamp % of threshold
7  seed                 14  trust-region frac     21 satgate (0 orig/1 one-sided/2 open)
                                                   22 minibatch B
```

---

## 3. Results at a glance

| Config | argmax | ppl | stable? |
|---|---|---|---|
| Random baseline | 0.1% | 1000 | — |
| `qllm.cpp` baseline | 2.0% | — | yes |
| Per-position attention (n32) | 2.8% | 849 | oscillates |
| **+ head-only asymmetric negative (N=4)** | 2.8% | **454** | **yes** |
| **5-block 1:1 interleave, N=32, n64** | 2.8% | **435** ← small-model best | yes |
| n_embed 128 (naive) | <1% | ~1000 | **collapses to random** |
| **n_embed 128 + satgate + N32 + minibatch B≥4** | ~2.4% | **~506–567** (still descending) | yes |

Two independent axes emerged:
- **argmax accuracy** plateaus at **~2.8%** and is **optimizer-bound** — depth,
  width, finer weights, gating all fail to move it.
- **Calibration (CE/ppl)** is *separable* and *very* improvable — this is where all
  the real progress happened.

---

## 4. What worked

- **Per-position design.** Attention only helps when a shared head predicts token
  t+1 from `attn_out[t]` *alone*, making attention the only cross-position path.
  (In a flatten-then-MLP design the MLP already mixes positions → attention is
  redundant and lost.) 2.7–2.8% vs 2.0% baseline.
- **Decorrelation via position subsampling.** Training ~4 of 16 positions/step
  reduces the correlated aggregate push and tames the bang-bang oscillation. + best-
  checkpoint tracking on a fixed eval set.
- **Head-only asymmetric negative flip-threshold (N).** *The* calibration
  breakthrough. Softening only the head's single push-away (negative-class) update
  by a factor N (it accumulates N× slower) nearly **halved perplexity (849→454)**
  and **eliminated the collapse**. The negative class churns every step as the
  model oscillates — it's the noisiest, most self-correlated update, so damping
  *only* it stabilizes the whole system. First lever to move the CE ceiling.
- **Depth damping law `N ≈ 2^blocks`.** In the 1:1 interleaved transformer, each
  added block is another site where oscillation can inject a sign flip, so the
  negative damping must ~double per block. **5 blocks / N=32 / n64 = ppl 435**, the
  standing best. (Peaks at 5; 6 blocks reverses — attention saturated on ctx=16.)
- **Threshold 512** is the small-model optimum (stability improves monotonically up
  to 512, no peak cost).
- **Saturation-gate fix (wide model).** Making the output-saturation *update gate*
  one-sided/open lets railed units recover — stops the catastrophic collapse-to-
  random (see §6).
- **Multi-window minibatch (wide model).** *The* wide-model breakthrough (see §6).

## 5. What failed (and why it's informative)

| Attempt | Result | Lesson |
|---|---|---|
| Band-flip hysteresis (Schmitt trigger) | failed | The collapse is a **system-level correlated cascade**, not per-weight chatter at boundaries. |
| EMA direction gate / LR schedule | neutral / over-damped | Damping the whole update rule interferes; can't beat the peak. |
| Top-K negatives (K>1) | negative | K=1 is a unique working point; extra negatives overload the bang-bang edge. |
| BiT `{0,1}` attention | neutral | Graded qscore magnitudes carry ~no info beyond the above-mean mask. |
| Network-wide (not head-only) asymmetry | hurts | The head's negative is special (targets a specific wrong *class*); damping graded-signal layers breaks their balance. |
| Softmax-CE head | competitive, no win | Only the head is graded; blocks/embedding still use the sign rule. |
| 4-bit finer weights | worse & undertrained | Each added bit needs proportionally more inertia; doesn't beat 435. Reinforces **optimizer-bound, not capacity-bound**. |
| Wider n_embed (naive) | collapses to random | See the whole wide-model saga, §6. |

---

## 6. The n_embed-128 (wide model) saga — this session

The wide model always reached a foothold, then **drifted back to random**. Turned
out to be **three separate failure modes stacked**, each needing its own fix.

### Ruling things out
- Not signal dilution (gates are relative/mean-based → width-invariant).
- Not int overflow, not update starvation (more positions didn't help).
- **Shallow-wide falsified**: 2- and 3-block wide models collapse anyway (later,
  and with *worse* footholds) → the instability is **depth-independent**.
- Threshold sweep: 512 = deepest foothold (767, then cliff); 1024 = bounded
  oscillation (834–926) but shallower foothold. A pure trade-off, no escape.

### Diagnosis via per-component telemetry
Instrumented every update site (emb / q / k / v / out / ffn / head) to log
flip-rate, flip-coherence, and saturation each eval window. Result:
- flip-rate and coherence **flat** everywhere → **not** a churn explosion, **not** a
  coherence cascade.
- **Saturation localized**: the attention `out_proj` and block FFN accumulators
  **progressively rail** (pin at ±threshold), climbing 0 → ~6%, **width-specific**
  (n64 stays ~0). Embedding and Q/K router **exonerated** (flat).
- So the collapse is **burn-in / railing** of two specific components.

### Anti-rail experiments (all instructive, none the final answer)
- **Leak** (proportional accumulator decay) → over-damps to random. Drags every
  weight toward center each step; can't hold learned weights.
- **Reset-on-flip** (integrate-and-fire; accumulator resets at fire, *cannot rail*)
  → drives layer saturation to **exactly 0** (validates the diagnosis) but is a
  **weak learner** (~20× fewer flips, plateaus ~920). Only bootstraps at low
  threshold.
- **Soft cap** (clamp accumulator at 70% of threshold) → preserves the deep
  foothold (770) but **still collapses**. ⟹ **burial depth is *not* the trap** — the
  leak/cap/reset family were all working the wrong variable.

> **Key insight:** railing *is* weight confidence / hysteresis. The normal scheme
> rails *because* it lets weights build deep conviction — which is what makes it
> learn well *and* what traps it. You can't delete railing; you must **bound** it.

### The sharper culprit: the output-saturation update gate
The update path *skips* a unit when its output activation rails at 0 or 3
(`if attn_out/own_act/V ∈ {0,3} continue`). So a railed unit can **never be pushed
back off the rail — it freezes.** That's why capping accumulator depth did nothing
(the weight still reaches the extreme level), and why reset "worked" (it keeps
weights jittering so outputs never *stay* saturated).

- **Saturation-gate fix** (one-sided: allow relieving pushes, block deepening ones;
  or fully open): **stops the full collapse-to-random.** Telemetry confirms units
  recover (out/ffn saturation spikes to 0.18 then *drops* to 0.012). But it unmasks
  a **large bounded oscillation** (~765–955) — so the freeze-gate was **part** of
  the trap, not all of it. Underneath is still the bang-bang.

### Composing the fixes → the breakthrough
- gate + thr1024 → **froze** (over-damped bootstrap). High threshold buys stability
  by *killing plasticity* — exactly wrong for a big model that needs more training.
- gate + N32 → **min 735** (best floor yet), but a wild swing (735↔988).
- **gate + N32 + multi-window minibatch (B independent windows)** → **the win.**

**Multi-window minibatch** = B *independent* context windows forward+backward'd at
**frozen weights**, pushes summed into the accumulators, then **one commit**. Noise
from independent windows averages out (true variance reduction); boundary weights
stop flip-flopping. This is **distinct from `positions_per_step`**, which adds
*correlated* pushes from *one* window (and indeed pos=4 was *worse*, ppl 796/wild).

Head-to-head (gate-open + N32 + thr512 + n128, all else equal):
| | min ppl | trajectory | test acc |
|---|---|---|---|
| pos=4 (correlated, 1 window) | 796 | wild swings to 1000 | ~1% |
| **B=4 (independent windows)** | **567** | steady descent, still falling at 20k | **2.43%** |
| B=8 | 542 | still falling | — |
| B=4, long run | **506** @ iter 57.5k (killed by session restart) | still slowly descending | — |

Bigger B ⇒ lower floor per update step (more averaging), at the cost of more
forward passes per step.

> **n_embed 128 was never a capacity wall.** It was
> **collapse (freeze-gate) + oscillation (bang-bang) + noise/undertraining**, and
> all three now have fixes: **satgate + N32 + minibatch.** The wide model is now
> **undertrained, not broken** — it keeps descending with more iters.

---

## 7. Lessons learned (the meta-findings)

1. **The bang-bang collapse is a system-level correlated cascade**, not weight-level
   chatter. Per-weight tricks (hysteresis) can't fix it; reducing *aggregate update
   correlation* can.
2. **The negative-class update is the destabilizer.** Softening *only* it
   (head-only asymmetry, N) is the single most effective calibration lever.
3. **Damping must track depth** (`N ≈ 2^blocks`).
4. **The ~2.8% argmax ceiling is optimizer-bound, not capacity-bound.** Width,
   depth, and finer weights all add representational capacity the argmax rule can't
   fill. Only a genuinely different (gradient-like) update or finer effective
   plasticity could move it — none tried has.
5. **Calibration is separable from argmax** and is where nearly all progress lives
   (849 → 435).
6. **Methodology matters and bites:**
   - acc-only checkpointing selects uncalibrated spikes → **co-report CE**;
   - **eval mutates training state** (the `run_mean`/`run_dev` EMAs update during
     eval), so eval frequency is silently a hyperparameter;
   - keep a **bit-identical control** for every new knob;
   - peak timing is **data-stream-locked** (moves with the sampler seed), so genuine
     quality is seed-dependent (~0.3–2.8%); seed 1337 was a lucky high.
7. **Wide models fail in layers.** Don't look for one bug — instrument and peel
   (collapse → oscillation → noise).
8. **Independent-window minibatch (variance reduction) ≠ positions_per_step
   (correlated).** The former is what finally trains wide models; the latter makes
   things worse.
9. **For big models, don't buy stability with inertia** (high threshold freezes
   learning) — buy it with **variance reduction** (batch). Big models need to train
   *more*, not slower.

---

## 8. Current best configs & open directions

**Small model (best overall):**
```
qllm_attn_xf.exe 40000 wikitext2_train.txt 5 32 64 1 1337 16 512 0 8 4 0
# 5 blocks, N=32, n_embed 64, 1:1 interleave, thr 512  ->  ppl 435, argmax 2.8%, stable
```

**Wide model (n_embed 128, the breakthrough recipe):**
```
qllm_attn_xf_batch.exe 100000 wikitext2_train.txt 4 32 128 1 1337 16 512 0 8 1 1 \
                       0 0 0 0 0 1 100 2 4
# gate-open (satgate=2) + N32 + thr512 + {1,2} init + minibatch B=4
# -> ppl ~506 and still descending; UNDERTRAINED, wants many more iters
```

**Open directions:**
- **Train the wide model much longer** (100k+; it was still descending at 57.5k when
  a session restart killed it) and **sweep B** (8, 16, 32) — it hasn't plateaued.
- **CPU-multithread the minibatch** (independent windows, per-thread accumulator
  deltas) to make large B / bigger models practical — cheap, before any GPU work.
- **Scale the model up** (n_embed 512+) now that wide models train — capacity that
  used to just collapse might finally help.
- **Per-layer damping** — telemetry shows railing is localized to out_proj+FFN, so
  targeted per-component N/threshold is the natural extension.
- The **2.8% argmax ceiling** remains untouched by anything structural — it likely
  needs a fundamentally different (gradient-like) update rule or genuinely finer
  effective weights.

---

## 9. Data

WikiText-2-v1 train text is exported once from the local HuggingFace arrow cache to
`wikitext2_train.txt`; the C++ programs run their own BPE tokenizer (vocab 1000) on
it.

---

*This log reflects an exploratory research thread; numbers are seed- and
schedule-dependent (see §7.6). Reproduce with the exact argv strings above.*
 
