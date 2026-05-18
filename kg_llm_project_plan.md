# LLM-based Knowledge Graph Completion: Project Plan

## Context for the assistant

I'm a software engineer at Microsoft (2+ years working on AOSP), and a Year 1 PhD student at Alexandru Ioan Cuza University in Iași. My PhD is structured around a Clifford-algebraic spine connecting knowledge graph embeddings, quantum walks, and generative/uncertainty modeling. The four-year arc runs from classical Clifford KGE benchmarked against ComplEx/RotatE/QuatE/Keci (Year 1), through quantum circuit compilation on simulators (Year 2), to QAE-accelerated link prediction for drug repurposing on biomedical KGs (Year 3, applied direction N28), with a Year 4 fork. Applied direction N31 (hypothesis generation from literature KGs) is also a natural destination.

My Master's thesis covered sample-efficient RL via knowledge transfer: transfer learning in Unity multi-stage environments, sequence modeling for offline RL (Decision Transformer, Decision ConvFormer, Decision Mamba, Decision Griffin on D4RL), and a hierarchical chain-of-thought Transformer with cross-episodic curriculum and teacher-student distillation on SMAC. I have three publications by Master's. I've also competed in the AI Mathematical Olympiad on Kaggle (the Progress Prize 2 round), working on efficient LLM inference and orchestration — loading and using models like Qwen and DeepSeek-Math under tight memory and runtime constraints, prompt engineering, majority voting, self-consistency. I have **not** fine-tuned LLMs hands-on before this project — that's the gap this project is designed to close.

**Goals for this project:** dual purpose. (1) Portfolio piece for job applications targeting Applied Scientist roles (Microsoft / Amazon / Meta), Frontier Lab Research Engineer roles (Anthropic / DeepMind / FAIR / MSR), and AI engineering roles broadly. I'm also open to research-track positions, but those typically require my PhD complete, so this project is primarily for the non-PhD-required roles I can apply to now. (2) Foundational work for my PhD Year 1 confirmation report — the KGE baselines (ComplEx, RotatE, TransE, QuatE on FB15k-237) are on my PhD roadmap regardless, so this project structure makes them double-duty.

**Resources:** ~10–15 hrs/week available. Kaggle T4 (16GB, with boosted quota from AIMO registration), Colab as backup, personal GTX 1650 (4GB) for code development only — not training. Copilot CLI available as coding assistant; using it for boilerplate but writing the conceptually important pieces myself.

**Working preferences:** I value honest pushback over agreement. If I'm about to do something wrong or suboptimal, say so directly. Concrete advice over abstract advice. When I'm stuck, I'd rather you ask me a clarifying question than guess.

---

**Working title:** *When do LLMs beat classical KGE for knowledge graph completion? An empirical study with post-training.*

A two-phase, eight-week project that doubles as portfolio (job applications) and PhD groundwork (Year 1 baselines + path to Year 3 biomedical work).

---

## 1. Project at a glance

You will:

1. Train classical KGE baselines (ComplEx, RotatE, TransE, QuatE) on FB15k-237 — these are baselines you owe your PhD anyway.
2. Fine-tune a small open LLM (Qwen2.5-1.5B-Instruct) for KG completion via SFT, evaluate against the KGE baselines using filtered Hits@k and MRR.
3. Construct a preference dataset using *KGE-mined hard negatives* — using the embeddings from step 1 to find difficult-to-distinguish wrong answers. This is the project's clever bit and the angle that makes it research-grade rather than tutorial-grade.
4. Run DPO (and optionally GRPO) on this preference data, evaluate again, analyze when LLMs win and when KGE wins.
5. Port the entire pipeline to a biomedical KG (Hetionet) to align with your Year 3 drug repurposing direction.
6. Ship: clean GitHub repo, technical report, optional Hugging Face Space demo.

The whole thing is designed so that **week 4 is a complete shippable milestone** — if you decide at that point that a job application window has opened or that PhD work needs more attention, you can stop with a presentable artifact.

---

## 2. What this project covers

Mapping to the checklist:

| Section | Coverage | How |
|---|---|---|
| §2 Transformer internals | Strong | Tokenization for structured inputs, chat templates, KV caching for batch eval |
| §3 Post-training | **Comprehensive** | SFT, DPO, optional GRPO, preference data construction, capability preservation |
| §4 Evaluation | **Comprehensive** | Standard KG metrics, custom evals, contamination checks, statistical significance |
| §5 Inference | Moderate | Quantization (QLoRA), batched generation, possibly vLLM for eval |
| §6 RAG/agents | Light (stretch) | Possible KG-as-retrieval extension |
| §7 Knowledge graphs | **Strong** | Both classical and modern approaches |
| §8 Engineering | **Strong** | HuggingFace stack, W&B, configs, reproducibility |
| §9 Reading | Moderate | You'll engage with ~15 papers seriously |

The gaps it deliberately doesn't fill: agents, RAG production patterns, multi-modal. If those become priorities, they're separate follow-up projects.

## 3. What this project signals to hiring managers

For Applied Scientist roles: it shows you can take a research problem end-to-end — formulate it, build infrastructure, run baselines, iterate, evaluate honestly. The fact that it produces real Hits@k tables comparable to published numbers is unusually concrete for a portfolio piece.

For Frontier Lab RE roles: it shows post-training fluency (SFT + DPO + eval), which is the actual job. The KGE-mined hard negatives angle shows you can think about training data design, not just plug into a recipe.

For research-track positions later: depending on what you find, this is plausibly publishable at ISWC, EMNLP Findings, or as a workshop paper at NeurIPS/ICLR. Even if you don't submit, it strengthens your PhD Year 1 confirmation.

The narrative that ties it together for a CV: *"I systematically compared LLM-based and embedding-based approaches to a foundational structured-prediction task, designed a novel preference data construction method using classical methods to bootstrap LLM training, and characterized the regime boundaries between the two paradigms."* That's a paragraph that reads well to anyone hiring for ML.

---

## 4. Hardware and infrastructure setup

**Compute reality:** Your GTX 1650 (4GB) is for code development only — not training. Real training runs on Kaggle T4 (16GB, the AIMO-boosted quota helps a lot here) and Colab as backup.

**Model choice (in order of preference):**
- **Primary: Qwen2.5-1.5B-Instruct.** Strong instruction following, 32K context, well-supported. Fits comfortably in 16GB with QLoRA.
- **Fallback: Llama 3.2 1B-Instruct.** If you hit memory issues or want faster iteration.
- **Stretch upgrade: Qwen2.5-3B-Instruct.** Only after you've shown the 1.5B pipeline works end-to-end.

**Stack:**
- `transformers` (model loading, generation)
- `trl` (SFTTrainer, DPOTrainer, GRPOTrainer)
- `peft` (LoRA/QLoRA)
- `bitsandbytes` (4-bit quantization)
- `datasets` (data pipelines)
- `accelerate` (training launch)
- `pykeen` or `libkge` for KGE baselines
- `wandb` for tracking — set this up day one
- `lm-evaluation-harness` for general benchmarks (catastrophic forgetting checks)

**Configs:** use Hydra or simple YAML files. Don't write configs in argparse — you'll regret it when you have 30 runs.

**Repo layout (start clean):**
```
kg-llm-completion/
  configs/         # YAML configs for each experiment
  data/            # Data loading and preprocessing
  kge/             # Classical KGE baselines (or wrap PyKEEN)
  llm/             # SFT, DPO, GRPO training scripts
  eval/            # Evaluation harness (Hits@k, MRR)
  scripts/         # Entry points, sweeps
  notebooks/       # EDA, analysis
  reports/         # Writeups, figures
  README.md        # Project overview, results table
```

**Copilot CLI usage:** delegate the boilerplate (data loaders, trainer setup, plotting) but write the conceptually important pieces yourself — the DPO data construction, the eval harness, the analysis. Hiring managers can usually tell the difference between someone who has built something and someone whose AI built it.

---

## 5. Phase 1: FB15k-237 (Weeks 1–4)

### Week 1: Foundations and KGE baselines

**Goal:** working data pipeline, KGE baselines training, repo structure.

**Day 1–2: Setup and data.**
- Create the repo, set up environment (likely a `requirements.txt` plus a Kaggle/Colab notebook that mirrors it).
- Download FB15k-237 from the standard source. Confirm splits: 272,115 train / 17,535 valid / 20,466 test triples; 14,541 entities; 237 relations.
- Build a data loading module that gives you `(head_id, relation_id, tail_id)` tuples and entity/relation name lookups. The named version (entities have human-readable names like "Albert Einstein") is what makes LLM training tractable.
- Implement filtered evaluation correctly. This is more error-prone than it looks: for each test triple `(h, r, t)`, you rank `t` against all entities, but you remove from the candidates any entity `t'` where `(h, r, t')` exists in train+valid+test (because those are also correct, just not the one being tested). This is the standard "filtered" protocol and getting it wrong invalidates results.

**Day 3–5: KGE baselines.**
- Use PyKEEN to train ComplEx, RotatE, TransE, QuatE on FB15k-237.
- Use standard hyperparameters from the original papers (PyKEEN has reasonable defaults). Aim for ~24 hours of total training across all four baselines (run them sequentially or use Kaggle's session limits creatively).
- Sanity check: your numbers should be in the ballpark of published results. ComplEx should hit roughly MRR ~0.32, RotatE ~0.34 on FB15k-237 filtered. If you're way off, debug before moving on.
- Save the trained embeddings — you'll need them for hard negative mining later.

**Day 6–7: Eval harness for LLMs.**
- Design how the LLM will be evaluated for KG completion. Two approaches:
  - **Scoring approach (recommended):** for each test triple `(h, r, ?)`, format a prompt and compute the model's log-probability of each candidate tail entity. Rank by log-prob. This is closer to how KGE evaluates and more directly comparable.
  - **Generation approach:** sample completions and see if `t` appears. Faster but less directly comparable to KGE numbers.
- Implement the scoring approach. For 14,541 candidates per query, you'll need to batch efficiently. Use the model in eval mode with KV caching across the shared prefix (the prompt up to where the entity goes).
- Verify the harness on the base (un-fine-tuned) Qwen2.5-1.5B. Expect terrible numbers — that's the point. Document them as the zero-shot baseline.

**Deliverable end of week 1:** baseline results table (4 KGE methods + 1 zero-shot LLM), data pipeline, eval harness. Commit and push.

### Week 2: Supervised fine-tuning

**Goal:** SFT model trained, evaluated, comparable numbers to baselines.

**Day 1–2: Data formatting for SFT.**
- Convert each training triple into prompt-response pairs. Two reasonable formats:
  - **Tail prediction:** "Given the entity *Albert Einstein* and the relation *place of birth*, predict the entity. Answer: *Ulm*."
  - **Head prediction:** "Given the entity *Ulm* and the relation *place of birth* (inverse), predict the entity. Answer: *Albert Einstein*."
- Train on both directions. This roughly doubles your data and matches how KGE evaluates (head and tail prediction).
- Use Qwen's chat template properly. Wrap in `<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`. Mask the loss on the prompt tokens — only compute loss on the answer.
- Watch out for the tokenization of entity names. Some entities have multi-token names; this is fine but make sure the tokenizer round-trips correctly.

**Day 3–5: SFT training.**
- QLoRA setup: 4-bit base model + LoRA adapters on Q, K, V, O projections (and optionally MLP projections). Start with rank 16, alpha 32.
- Hyperparameters to start with: learning rate 2e-4, batch size 4 with gradient accumulation 8 (effective 32), 1–2 epochs over the dataset, cosine schedule, 100-step warmup.
- Train on Kaggle. You may need to chunk the run across multiple sessions; save checkpoints.
- Monitor: training loss, gradient norms, eval loss on a small validation subset (don't run full eval every step).

**Day 6: First evaluation.**
- Run the SFT model through your eval harness. Get Hits@1, Hits@3, Hits@10, MRR (filtered).
- Compare to KGE baselines. Realistic expectation: SFT will likely *underperform* the strongest KGE baseline at this stage, possibly significantly. That's fine and expected — DPO is what's expected to close the gap.
- Run a small subset of MMLU or BBH to check for catastrophic forgetting.

**Day 7: Analysis.**
- Look at qualitative outputs. Where does the SFT model fail? Common failure modes: hallucinating entities not in the KG, wrong relation interpretation, overconfident wrong answers.
- This analysis seeds your DPO data construction strategy.

**Deliverable end of week 2:** SFT model, results compared to KGE, qualitative failure analysis written up.

### Week 3: DPO and the hard-negative-mining angle

**Goal:** preference data constructed, DPO model trained, results.

**Day 1–2: Hard negative mining (the cleverness).**

This is the most original part of the project. Standard DPO uses human or AI feedback for preference pairs. We construct preferences programmatically using KGE embeddings:

For each training triple `(h, r, t)`:
- The **chosen** completion is `t`.
- The **rejected** completion is a hard negative — an entity `t'` that is similar to `t` but doesn't form a valid triple with `(h, r)`.

Hard negative selection strategies (run all and compare):

1. **KGE-similarity-based:** use ComplEx or RotatE embeddings. Find the top-K entities closest to `t` in embedding space, filter to those that don't form a valid `(h, r, ?)` triple anywhere in the KG, pick one as the rejected.

2. **Type-constrained random:** sample any entity of the same type as `t` that doesn't form a valid triple. Easier baseline.

3. **Model-mistake-based:** generate completions from the SFT model, find ones that are wrong, use those as rejected. Adaptively hard.

The KGE-similarity strategy is the unique angle. The story: *"We use classical KGE precisely where it shines — discriminating subtle entity differences in geometric space — to construct preferences that teach the LLM what classical methods know implicitly."* That's a paragraph that reads well in a paper.

Generate, say, 50K–100K preference pairs from the train set. Store as a HuggingFace dataset with `chosen` and `rejected` fields plus the prompt.

**Day 3–5: DPO training.**
- Use `trl.DPOTrainer`. Reference model is your SFT checkpoint. Beta typically 0.1.
- LoRA on top of the SFT-LoRA (you can either merge SFT first then add new adapters, or stack adapters). Merging is simpler.
- Hyperparameters: learning rate 5e-7 (yes, much smaller than SFT — DPO is sensitive to high LRs), 1 epoch typically enough. Watch for the chosen-rewards and rejected-rewards diverging — that's healthy. Watch for both dropping — that's degradation.
- Monitor: KL divergence from reference, reward margins. Save checkpoints every ~500 steps so you can pick the best.

**Day 6: Evaluation.**
- Full eval: Hits@1/3/10, MRR filtered, compared to SFT and KGE baselines.
- General capability check: MMLU subset. If MMLU dropped substantially from base → this is the alignment tax, document it.
- Compare hard-negative strategies (1, 2, 3 above): which produced the best post-DPO model?

**Day 7: Ablations.**
- Without DPO (SFT only)
- DPO with random negatives vs KGE-mined negatives
- Different beta values

**Deliverable end of week 3:** DPO model with results table comparing 5+ configurations. The KGE-mined-negatives result is the headline.

### Week 4: Comprehensive evaluation and writeup

**Goal:** publishable-quality artifact, ready to share.

**Day 1–2: Deep evaluation.**
- Statistical significance: bootstrap confidence intervals on Hits@k. Are differences between methods real?
- Per-relation analysis: for which relations does the LLM beat KGE? For which does KGE beat LLM? This is the substantive contribution.
- Specific predictions: cherry-pick (and balance) examples where LLM is right and KGE wrong, vice versa, both wrong, both right. Qualitative section in writeup.
- Contamination check: are FB15k-237 triples in Qwen's pretraining? You can check by looking at zero-shot performance on subsets — if certain entities are massively easier than others, that's a clue. Document caveats.

**Day 3–4: Writeup.**
- Aim for a 6–10 page technical report (NeurIPS-style) or a 3000-word blog post. The format matters less than the content.
- Structure: motivation → related work (brief) → method (data, training, evaluation) → results → analysis → limitations → conclusion.
- Include negative results. "We tried X and it didn't work because Y" is more impressive than a sanitized success story.
- Lead the analysis section with the per-relation breakdown — that's the most interesting finding.

**Day 5: Repo cleanup.**
- README with results table at the top, then how-to-reproduce.
- Clean up notebooks. Remove dead code. Make sure `python scripts/train_sft.py --config configs/qwen_sft.yaml` actually works from a fresh clone.
- Pin versions in `requirements.txt`. Reproducibility is a ranked-by-most-hiring-managers signal.

**Day 6–7: Polish and announce.**
- Optional: Hugging Face Space with a small demo (input a head and relation, see top-10 predictions).
- Push model checkpoint to HF Hub.
- Post writeup somewhere visible (personal blog, Twitter/X, LinkedIn, or just the GitHub README).
- This is when you can start using it on applications. The Phase 1 alone is a complete portfolio piece.

**Deliverable end of week 4:** complete, shippable Phase 1 — repo, model, writeup. You could stop here if needed.

---

## 6. Phase 2: Biomedical KG (Weeks 5–8)

This phase is shorter relative to its scope because the infrastructure from Phase 1 is mostly portable. The new work is: different KG, domain-specific challenges, deeper analysis.

### Week 5: Setting up Hetionet

**KG choice:** **Hetionet** is the recommended starting point — public, widely used in biomedical KG research, ~47K nodes (genes, diseases, drugs, etc.) and 2.25M edges across 24 metaedge types. Smaller and more focused than PrimeKG, which is good for the timeline.

- Download Hetionet from the official source (it's MIT-licensed).
- Construct train/valid/test splits if not provided. Use a held-out edge approach (e.g., 80/10/10).
- The interesting prediction targets are typically Compound–Treats–Disease (drug repurposing), Gene–Associates–Disease, Compound–Causes–Side Effect.
- Build the same data loaders as Phase 1, with biomedical entity name resolution.

### Week 6: Port and SFT

- Re-train KGE baselines on Hetionet (PyKEEN handles this fine).
- Run SFT with the Phase 1 pipeline, no major changes needed.
- Biomedical entities have technical names — "*Phenylpropanolamine*", "*HMOX1*", "*Coronary artery disease*". Some pretraining knowledge will help, but expect more confusion than FB15k-237.
- Evaluate.

### Week 7: DPO with biomedical hard negatives

- Same hard-negative-mining strategy as Phase 1, applied to Hetionet.
- Additional opportunity: type-aware hard negatives matter more here. A drug as a hard negative for a disease prediction is uninformative; another drug is the right comparison.
- DPO training, evaluation.

### Week 8: Writeup, comparison, shipping

- Combined writeup covering both Phase 1 and Phase 2. The story now has a generalization arc: "the recipe transfers from general-domain to biomedical with measurable success/limits."
- Specifically focus on drug repurposing predictions — concrete examples where the model recovers known repurposing candidates is a memorable result.
- This is the writeup that goes in your PhD Year 1 report's appendix and on your CV.

**Final deliverable:** two-phase project, comprehensive results, clean repo, polished writeup, model artifacts on HF Hub.

---

## 7. Stretch goals (only after MVP is shippable)

These are not on the critical path. They're projects that grow out of this one.

**GRPO instead of/alongside DPO.** Implement GRPO training using the verifiable reward of "is this a valid triple in the KG?" This is conceptually closer to your RL background and to current frontier research (R1-style training). One week of additional work. Excellent CV signal for frontier-lab roles.

**Hybrid LLM + KGE model.** Inject KGE embeddings as soft prompts or as additional conditioning into the LLM. This is where your future Clifford-KGE work will plug in — once your PhD produces Clifford embeddings, swap them in for ComplEx. This is a publishable extension.

**Multi-hop reasoning evaluation.** Move from one-hop completion to multi-hop questions. "What diseases are treated by drugs that target genes associated with hypertension?" Tests whether the LLM has learned compositional reasoning.

**Tool use formulation.** Rather than predicting entities directly, have the LLM call the KGE model as a tool. Connects to the agent skillset on the checklist.

**Quantum angle (Year 2 territory).** Once your PhD has quantum circuit compilation results, generate training data for an LLM that translates natural language to quantum circuits, then use the KG-LLM pipeline as the training methodology. Don't pursue this in the next 8 weeks; flag it as a future paper.

---

## 8. Risks and mitigations

**Memory issues on T4.** With a 1.5B model and QLoRA, you have headroom but it's tight. Mitigations: reduce batch size and use gradient accumulation, lower max sequence length to 512 (most KG triples don't need more), drop to Llama 3.2 1B if needed.

**KGE baselines don't reproduce.** Your numbers should match published ones within ~5% relative. If they don't, the issue is usually in eval (filtered vs raw, head + tail averaged correctly). PyKEEN has a verified evaluation function — use it.

**SFT degrades general capability badly.** This is the alignment tax. If MMLU drops by more than ~20%, you've over-trained. Mitigations: lower LR, fewer epochs, smaller LoRA rank, mix in some general instruction data.

**DPO collapses.** The dreaded scenario where DPO training "succeeds" by training loss but the model becomes worse at everything. Mitigation: aggressive eval throughout training, save many checkpoints, pick the best — not the last.

**Time slip past 8 weeks.** Almost guaranteed. The 8-week plan has real slack: Phase 1 alone (4 weeks) is a complete project. If Phase 2 has to shrink to 2 weeks, that's fine — focus on showing the recipe ports, even if the analysis is shallower.

**Boredom or motivation drop.** Real risk on solo projects. Mitigation: weekly artifact (commit, plot, draft section). Each week should produce something visible. Don't go three weeks without a visible result.

**The result is uninteresting.** Possible: SFT + DPO might just basically match KGE without the LLM offering a clear win. That's still publishable as a careful study. The "negative result" framing is honest and respected. Don't manufacture wins.

---

## 9. Reading list

**Read carefully (foundational for the project):**
- Chen et al., *Decision Transformer* (you know this from your thesis)
- Rafailov et al., *DPO: Direct Preference Optimization* (2023)
- DeepSeek-R1 paper (2025)
- KG-BERT (Yao et al., 2019)
- KGT5 (Saxena et al., 2022) — direct precursor to your project
- Trouillon et al., *ComplEx* (2016) — your KGE baseline
- Sun et al., *RotatE* (2019) — your KGE baseline

**Skim (context):**
- LLaMA 2 / LLaMA 3 technical reports
- Hu et al., *LoRA* (2021)
- Dettmers et al., *QLoRA* (2023)
- Recent (2024–2025) papers on LLMs for KG completion — search Semantic Scholar with "LLM knowledge graph completion" filtered to recent

**Reference (have open during work):**
- HuggingFace TRL documentation
- PyKEEN documentation
- lm-evaluation-harness README

---

## 10. The writeup template

Aim for 6–10 pages or 3000 words. Structure:

1. **Abstract / TL;DR** (1 paragraph). Summarize result, including the per-relation finding.
2. **Introduction.** Why is this question interesting? Why now? KGE has plateaued, LLMs are new — when does each win?
3. **Background.** Brief: KGE basics, modern post-training. Two paragraphs each. Don't write a full survey.
4. **Method.** Data, models, training (SFT, DPO, hard negative mining). The hard-negative-mining section is the original contribution — give it space.
5. **Experiments.** Setup, KGE baselines, SFT, DPO. One results table that's easy to read.
6. **Analysis.** Per-relation breakdown. Qualitative examples. Where does each method shine? This is the substantive content.
7. **Phase 2 (biomedical).** Brief — show the recipe transfers, show one or two interesting drug repurposing predictions.
8. **Limitations.** Be honest. Contamination caveats. Compute limits. Single-seed runs (probably). Generalization questions.
9. **Conclusion.** What we learned. What's next.
10. **Appendix.** Hyperparameters, training curves, additional examples.

Write it for a smart engineer who hasn't read 50 KGE papers. The audience is a hiring manager, not a peer reviewer (peer review can come later).

---

## 11. Success criteria

By end of week 4 (Phase 1 ship):
- [ ] Repo on GitHub with reproducible code
- [ ] At least 3 trained KGE baselines with results matching published
- [ ] SFT model with results
- [ ] DPO model with results, including hard-negative-mining ablation
- [ ] Writeup published (blog or PDF)
- [ ] Model on HF Hub

By end of week 8 (full ship):
- [ ] Phase 2 results on Hetionet
- [ ] Combined writeup with biomedical analysis
- [ ] Per-relation analysis as a first-class result
- [ ] CV updated with project link

For job applications: start sending after week 4. The writeup and repo are enough to be the project on your CV. Phase 2 strengthens it but doesn't gate it.

---

## 12. Day-one action items

To start tomorrow:

1. Create the GitHub repo with the structure above. Empty is fine.
2. Sign up for W&B if you don't have it.
3. Clone the FB15k-237 dataset and look at it. Print 10 random triples. Make sure entity names resolve.
4. Read the DPO paper (Rafailov et al. 2023) front to back. It's short. This is the conceptual core of the project.
5. Skim the KGT5 paper (Saxena et al. 2022) — closest direct precursor to your work.
6. Set up a Kaggle notebook with a base Qwen2.5-1.5B-Instruct loaded in 4-bit. Confirm it generates. This is your sanity check that the environment works before doing anything ambitious.

That's the first session. Two hours, max.
