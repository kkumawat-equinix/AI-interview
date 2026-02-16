# Interview Preparation Kit — Interview-Ready Answers (60-minute)

This file provides concise, interview-ready answers for each major question in the deep-dive guide. Use these as 30–90 second spoken answers, and expand with examples from your experience during interviews.

---

## 1. Prompt Engineering — Answers

- Q: What is the difference between zero-shot, one-shot, and few-shot prompting, and when do you prefer each?
  A: Zero-shot provides only an instruction and relies on the model's pretraining—best when the task is common and you need minimal context. One-shot/few-shot include 1+ examples to demonstrate format or edge behavior—prefer few-shot when format, edge cases, or style are important and you cannot fine-tune.

- Q: Explain chain-of-thought prompting. When does it improve performance and what are its limitations?
  A: Chain-of-thought asks the model to show intermediate reasoning steps, improving multi-step logical tasks (math, reasoning). It helps when models benefit from decomposing problems but increases token use, may leak spurious reasoning, and can still hallucinate incorrect steps.

- Q: How do system messages differ from user messages in chat LLMs?
  A: System messages set global behavior (tone, role, constraints); user messages are task-specific. Use system messages for invariants (safety rules, persona) and user messages for specific prompts and examples.

- Q: What are prompt injection attacks and how do you mitigate them?
  A: Injection occurs when model input contains malicious instructions that override intended behavior. Mitigate with input sanitization, strict system messages, sandboxing tool calls, and defensive parsing/verification of outputs.

- Q: How do you design a prompt that extracts tabular data from unstructured text reliably?
  A: Provide a clear schema, include 1–3 examples (few-shot), constrain output format (JSON/CSV), and add validation steps (ask model to confirm schema compliance). Post-parse with deterministic checks and reject or re-query if invalid.

- Q: How do you A/B test prompts at scale? What metrics do you collect?
  A: Deploy variants to split traffic, track correctness, F1/EM for structured tasks, user satisfaction, latency, token cost, and safety flags. Use statistical tests and automated logging (LangSmith/PromptLayer) to iterate.

- Q: How do you evaluate prompt quality quantitatively and qualitatively?
  A: Quantitative: accuracy/F1, BLEU/ROUGE for text tasks, hallucination rate, cost. Qualitative: human annotation for usefulness, tone, and safety. Combine both in a CI loop.

- Q: How do you ensure prompts do not encourage unsafe or biased outputs?
  A: Start with constrained system messages, filter inputs/outputs, include counterfactual/fairness tests in evaluations, and add human review for edge cases.

---

## 2. LLM Fine-Tuning — Answers

- Q: Explain supervised fine-tuning, instruction tuning, and RLHF. When is each used?
  A: Supervised fine-tuning trains model on labeled input-output pairs for accuracy on a domain. Instruction tuning optimizes models to follow natural instructions. RLHF uses human feedback as a reward signal to align behavior for preference-based objectives. Use supervised for deterministic tasks, instruction tuning for general instruction-following, and RLHF when preferences are complex and difficult to encode.

- Q: What are LoRA and QLoRA? How do they reduce compute and memory requirements?
  A: LoRA injects low-rank adapters into weights, training fewer parameters. QLoRA quantizes weights to 4-bit and fine-tunes adapters, lowering memory and enabling large-model tuning on smaller GPUs.

- Q: Why is tokenization important for domain-specific fine-tuning?
  A: Tokenization affects vocabulary coverage and sequence length. Domain tokens (chemical names, code) can fragment into many subwords—retraining or augmenting the tokenizer reduces token bloat and improves learning efficiency.

- Q: How do you curate and clean training data for instruction tuning?
  A: Deduplicate, remove PII, normalize formats, filter low-quality or contradictory examples, and balance classes. Use automated heuristics plus manual sampling for quality assurance.

- Q: How would you fine-tune a large model on a limited GPU budget?
  A: Use LoRA/PEFT, gradient accumulation, mixed precision, smaller batch sizes, dataset sampling, and checkpoint/resume strategies; or use QLoRA to fit larger models on a single GPU.

- Q: How do you evaluate the tuned model for generalization and failure modes?
  A: Use held-out test sets, adversarial and distribution-shift tests, human evaluation, and regression tests against prior model outputs. Monitor hallucination, correctness, and safety metrics.

- Q: How do you detect and remove memorized PII from models?
  A: Run canary-style membership tests, search model completions for verbatim dataset strings, and remove offending examples followed by targeted fine-tuning or mitigation via differential privacy when needed.

---

## 3. Multi-Agent Systems — Answers

- Q: What are common agent roles and responsibilities?
  A: Planner (decomposes tasks), Retriever (fetches knowledge), Executor (runs tools/APIs), Critic (validates outputs), and Memory/Logger (state). Each specializes to improve modularity and testability.

- Q: Centralized orchestration vs decentralized coordination — pros/cons?
  A: Centralized orchestration gives global control and simpler debugging but is a single point of failure and may limit scalability. Decentralized emergent coordination scales and can be robust but is harder to reason about and validate.

- Q: How to design agent memory and state?
  A: Use ephemeral short-term memory for context, persistent DB for long-term knowledge, and strict access controls. Store structured summaries and retrieval indices for efficiency.

- Q: How do you detect and recover from agent failures or deadlocks?
  A: Implement heartbeats, timeouts, retries, and supervisor processes that restart or escalate failed agents. Add idempotency and checkpoints for safe retries.

- Q: Compare LangChain, Autogen, and CrewAI briefly.
  A: LangChain provides modular building blocks and flexibility for tool integration; Autogen focuses on conversational agent orchestration; CrewAI offers higher-level structured orchestration. Choose based on maturity, extensibility, and the team's familiarity.

---

## 4. Vector Databases & Embeddings — Answers

- Q: Difference between dense vector search and keyword search?
  A: Dense vector search measures semantic similarity in embedding space and retrieves conceptually related items; keyword search matches lexical tokens and is precise for exact terms. Use hybrid approaches for best precision and recall.

- Q: Popular index types and trade-offs?
  A: Flat (exact, high memory), IVF (inverted file, lower memory, needs quantization), HNSW (fast ANN, good recall/latency trade-off). Choice depends on latency, throughput, and accuracy needs.

- Q: How to choose an embedding model for a domain?
  A: Benchmark candidate models on domain-specific retrieval tasks (recall@k, MRR), consider licensing/cost, and validate against representative queries; fine-tune or instruct-tune embeddings if needed.

- Q: How to monitor vector DB health?
  A: Track recall@k, precision@k, QPS, p99 latency, index size, and index build/refresh times. Alert on drift, latency spikes, and decreased recall.

- Q: How to tune HNSW parameters (M, efSearch, efConstruction)?
  A: Increase `M` for higher recall at cost of memory; raise `efConstruction` for slower builds with better accuracy; set `efSearch` to balance query latency vs recall and tune on representative workloads.

---

## 5. Retrieval-Augmented Generation (RAG) — Answers

- Q: Core components of RAG?
  A: Embed documents, store/index embeddings, retrieve relevant chunks for queries, and generate answers conditioned on retrieved context. Re-ranking and grounding steps improve faithfulness.

- Q: How to choose chunk size and overlap?
  A: Choose chunk sizes that capture semantic units (paragraphs or sections). Use modest overlap to preserve context across chunk boundaries; balance retrieval precision vs cost and latency.

- Q: How to reduce hallucination in RAG?
  A: Ensure high-quality retrieval, re-rank passages, include provenance citations in responses, and add verification steps or fact-checking agents before finalizing outputs.

- Q: How to version knowledge base updates safely?
  A: Use blue/green deployments or staged index updates, snapshot indices, validate on regression test queries, and provide a fallback to a previous snapshot if regressions occur.

---

## 6. Reinforcement Learning (RL) for Agents — Answers

- Q: Core RL concepts succinctly?
  A: Policy maps states to actions; reward signals guide desired outcomes; value functions estimate expected return; discount factor trades present vs future reward; exploration discovers better policies.

- Q: When to use PPO vs DQN?
  A: PPO is a stable on-policy policy-gradient method suited for continuous or large action spaces and modern policy optimization; DQN is off-policy value-based for discrete action spaces.

- Q: How to design rewards for multi-step API-driven tasks?
  A: Combine dense intermediate rewards for subtask progress with sparse final success rewards; add shaping to avoid perverse incentives, and include penalties for costly or unsafe actions.

- Q: How to avoid reward hacking?
  A: Use adversarial testing, human-in-the-loop evaluation, conservative reward shaping, and constraints that penalize undesirable shortcuts.

- Q: How to safely roll out policy changes?
  A: Use shadow rollouts, canary deployments, offline evaluation with logged data, and human monitoring with kill-switches.

---

## How to Use These Answers
- Memorize 3–5 concise answers per topic and back each with one real example or metric from your experience.
- Practice speaking them aloud and mapping each answer to a short system-design sketch or code snippet you can draw during interviews.

If you want, I can now:
- expand these into a one-page printable cheat sheet, or
- convert them into flashcards or a timed mock interview script.
