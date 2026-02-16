# Interview Preparation Kit — Deep Dive (60-minute)

This document expands each focus area from the 60-minute interview kit into concise deep-dive context and grouped deep-level interview questions (fundamentals, design, implementation, evaluation, failure modes, ethics, and project-based prompts). Use these to prepare short technical answers, whiteboard/system-design explanations, and code-level discussions.

---

## 1. Prompt Engineering

Context:
Prompt engineering studies how to structure inputs to LLMs to reliably elicit desired outputs. It covers prompt types (zero-shot, few-shot, chain-of-thought), role and system messages, instruction design, context-window management, and validation strategies for robustness and safety.

Questions — Fundamentals
- What is the difference between zero-shot, one-shot, and few-shot prompting, and when do you prefer each?
- Explain chain-of-thought prompting. When does it improve performance and what are its limitations?
- How do system messages differ from user messages in chat LLMs? Give examples.
- What are prompt injection attacks and how do you mitigate them?

Questions — Design & Strategy
- How do you design a prompt that extracts tabular data from unstructured text reliably?
- Describe techniques for controlling verbosity, tone, and style in generated responses.
- How do you build prompts robust to hallucinations and factual errors?
- How do you design prompts for multi-step tasks where intermediate verification is required?

Questions — Implementation & Tooling
- How do you A/B test prompts at scale? What metrics do you collect?
- Which prompt-testing tools have you used (LangSmith, PromptLayer, etc.) and how do they fit into a CI loop?
- How do you automate prompt optimization while keeping human-understandable intents?

Questions — Evaluation & Metrics
- How do you evaluate prompt quality quantitatively and qualitatively?
- Describe metrics for answer correctness, consistency, and safety. How are they computed?
- How do you set up a feedback loop from users to improve prompts over time?

Questions — Failure Modes & Edge Cases
- Give examples where few-shot prompting fails but fine-tuning succeeds — why?
- How do you handle prompts where model outputs diverge across runs (non-determinism)?
- How do you debug cases where the model ignores constraints (e.g., answer length)?

Questions — Ethics & Safety
- How do you ensure prompts do not encourage unsafe or biased outputs?
- Discuss trade-offs when adding filtering vs. constraining prompts.

Project / Behavioral Prompts
- Walk through a prompt lab you ran: objectives, methodology, findings, and improvements.
- Given a legacy prompt that suddenly produces worse outputs after model upgrade, how do you approach regression analysis?

---

## 2. LLM Fine-Tuning

Context:
Fine-tuning adapts models to domain-specific data or instruction-following behavior. Techniques range from full-parameter tuning to parameter-efficient methods (LoRA, QLoRA, PEFT). Considerations include dataset curation, tokenization, compute cost, evaluation, and risks like overfitting and hallucination amplification.

Questions — Fundamentals
- Explain supervised fine-tuning, instruction tuning, and RLHF. When is each used?
- What are LoRA and QLoRA? How do they reduce compute and memory requirements?
- Why is tokenization important for domain-specific fine-tuning?

Questions — Data & Preprocessing
- How do you curate and clean training data for instruction tuning?
- How do you construct high-quality prompts/targets pairs for supervised fine-tuning?
- How do you handle label noise and contradictory instructions in your dataset?

Questions — Architecture & Efficiency
- Explain how you would fine-tune a large model on a limited GPU budget.
- What trade-offs exist between LoRA, full fine-tuning, and adapters in latency, performance, and stability?
- How do you decide which layers or subsets of parameters to adapt?

Questions — Evaluation & Safety
- How do you evaluate the tuned model for generalization and failure modes?
- Describe guardrails to avoid amplifying bias or memorized private data during fine-tuning.
- How do you set up regression tests across model versions?

Questions — Tools & Libraries
- Walk through a fine-tuning pipeline using Hugging Face transformers + datasets + accelerate.
- What considerations apply to mixed-precision, gradient accumulation, and checkpointing strategies?

Questions — Troubleshooting & Edge Cases
- Models started to hallucinate more after tuning — how would you debug and mitigate?
- How would you detect and remove inadvertent memorized PII from your dataset or model weights?

Project / Behavioral Prompts
- Describe a domain fine-tune you implemented: dataset sourcing, preprocessing, training details, and evaluation metrics.

---

## 3. Multi-Agent Systems (CrewAI / LangChain / Autogen)

Context:
Multi-agent systems orchestrate multiple specialized agents (planner, retriever, critic, executor) to perform complex workflows. Key topics: agent roles, communication protocols, shared memory, failure handling, and orchestration vs. emergent behavior.

Questions — Fundamentals
- What are the common agent roles (planner, executor, retriever, critic)? Describe responsibilities.
- Compare centralized orchestration vs. decentralized emergent coordination.

Questions — Architecture & System Design
- How would you design a multi-agent pipeline for long-form research summarization?
- How do you design agent memory and state: ephemeral vs. persistent; local vs. shared?
- How do you handle agent discovery and service composition in a production system?

Questions — Communication, Protocols & Consistency
- How do you design the communication protocol between agents (message formats, retries, idempotency)?
- How do you ensure consistency and avoid duplicated work across agents?

Questions — Error Handling & Robustness
- How do you detect and recover from agent failures or deadlocks?
- How would you implement sanity checks and cross-agent verification to reduce hallucinations?

Questions — Tooling & Frameworks
- Compare LangChain, Autogen, and CrewAI for building multi-agent solutions. When would you choose one over the others?
- How do you integrate external tools/APIs (search, DB, computation) safely from agents?

Questions — Evaluation & Metrics
- What metrics measure multi-agent system success (task completion, latency, cost, reliability)?
- How do you perform causal analysis when multi-agent workflows produce incorrect outputs?

Questions — Security & Ethics
- How do you prevent agents from leaking sensitive data between tenants or workflows?

Project / Behavioral Prompts
- Describe a multi-agent project you led: design choices, orchestration, failure modes, and lessons learned.

---

## 4. Vector Databases & Embeddings

Context:
Vector databases index embeddings to enable semantic search and retrieval. Key concerns: choice of embedding model, indexing method (Flat, IVF, HNSW), metadata filtering, latency, scaling, and monitoring embedding drift.

Questions — Fundamentals
- Explain the difference between dense vector search and traditional keyword search.
- What are popular index types (Flat, IVF, HNSW)? Discuss trade-offs in latency, recall, and memory.

Questions — Embedding Models & Quality
- How do you choose an embedding model for a specific domain (e.g., biomedical vs. customer support)?
- How do you detect and measure embedding drift over time?

Questions — System Design & Scaling
- Design a low-latency semantic search service for a production application serving 10k QPS.
- How would you shard and replicate vector indices for availability and throughput?

Questions — Indexing Strategies & Tuning
- When do you use approximate nearest neighbor (ANN) vs. exact search?
- How do you tune HNSW parameters (M, efSearch, efConstruction) for your workload?

Questions — Storage & Hybrid Search
- How do you combine BM25/keyword filters with vector search for precision?
- Discuss metadata filtering and dynamic re-ranking strategies.

Questions — Monitoring & Evaluation
- What metrics do you collect to monitor vector DB health (recall@k, QPS, latency, index size)?
- How do you setup labeling and offline evaluation to validate retrieval quality?

Questions — Tools & Trade-offs
- Compare ChromaDB, FAISS, Qdrant, and Weaviate on persistence, metadata, and cloud-readiness.

Project / Behavioral Prompts
- Walk through building a RAG retrieval layer: embedding choice, chunking, index config, and evaluation pipeline.

---

## 5. Retrieval-Augmented Generation (RAG)

Context:
RAG pipelines combine retrieval (embedding + index) with generation to ground LLM outputs in external knowledge. Challenges include chunking, retrieval relevance, latency, hallucination mitigation, and versioning of the knowledge base.

Questions — Fundamentals
- Describe the core components of RAG and the role each plays.
- What are semantic chunking and passage scoring, and why are they important?

Questions — Architecture & Design
- Design a RAG architecture for long-document QA with strict latency SLOs.
- How do you choose chunk size and overlap? What are trade-offs?

Questions — Retrieval Relevance & Filtering
- How do you handle noisy or conflicting documents in retrieval?
- Describe strategies for re-ranking retrieved passages before generation.

Questions — Integration & End-to-End
- How do you integrate streaming LLM outputs with retrieval so partial answers can be generated promptly?
- How do you version and roll out knowledge base updates without breaking production behavior?

Questions — Evaluation & Safety
- How do you evaluate faithfulness of RAG outputs and measure hallucination reduction?
- How do you design fallbacks when retrieval confidence is low?

Questions — Tools & Libraries
- Compare using LangChain vs. LlamaIndex vs. custom pipelines for RAG.

Project / Behavioral Prompts
- Explain a RAG implementation you've built: retrieval strategy, chunking, index tuning, and evaluation.

---

## 6. Reinforcement Learning (RL) for Agents

Context:
RL for agents focuses on optimizing agent policies to maximize long-term objectives (e.g., task success). In LLM systems, RL can tune tool selection, dialog strategies, or verification policies; RLHF is a specific application using human feedback as rewards.

Questions — Fundamentals
- Explain the core RL concepts: policy, value function, reward, discount factor, exploration vs. exploitation.
- Compare on-policy vs. off-policy methods; where is PPO used and why?

Questions — Applying RL to LLM Agents
- How would you formulate reward functions for an agent that must reliably complete multi-step API-driven tasks?
- What issues arise when rewards are sparse or noisy in agent workflows?

Questions — Algorithms & Frameworks
- When would you choose PPO vs. DQN vs. actor-critic architectures for agent problems?
- What frameworks and tools do you use for RL experimentation (OpenAI Gym, RLlib, CleanRL)?

Questions — Safety, Stability & Training
- How do you avoid reward hacking and specification gaming in RL for agents?
- How do you ensure stable training when using learned models of environment or simulators?

Questions — Evaluation & Deployment
- How do you evaluate an RL-improved agent in production? What A/B or offline metrics are useful?
- Describe a strategy to safely roll out policy changes that could degrade user experience.

Questions — RLHF Specifics
- Compare RLHF with supervised fine-tuning and explain when RLHF is necessary.
- What are practical issues when scaling RLHF on large models (data, compute, annotation quality)?

Project / Behavioral Prompts
- Walk through a project where you used RL (or RLHF) to improve agent behavior: reward design, training details, and ROI.

---

## How to Use This Document
- Pick 2–3 topics most relevant to your interviewer's role and prepare concise 2–3 minute explanations for each.
- Practice system design whiteboards for at least one topic (RAG or Multi-Agent or Vector DB) including trade-offs and failure modes.
- Prepare one hands-on story: a project where you owned design, implementation, and iteration; be ready to discuss metrics and postmortems.

If you'd like, I can:
- expand any single topic into a longer list of focused coding or whiteboard questions,
- generate a one-page cheat sheet for on-the-spot answers, or
- produce a printable flashcard set of the most frequent questions per topic.
