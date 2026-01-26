# Interview Questions (Asked)

## Core GenAI / LLM / RAG

1. Explain the project.
2. Explain the transformer architecture.
3. Explain how the sentence “My name is sundaram and i’m from delhi” will pass through an encoder layer.
4. Explain the self-attention mechanism.
5. Explain the difference between LoRA, QLoRA, and PEFT.
6. Explain LLM parameters: Temperature, Top-k, Top-p, Max tokens.
7. How to prevent LLM from accessing PII information?
8. LLM hallucinations and prevention techniques.
9. How to incorporate feedback mechanisms in LLM?
10. Explain all the components of RAG architecture in detail.
11. Explain indexing and vector DB.
12. Explain different types of chunking techniques.
13. What are different ways of improving RAG solutions?
14. How can you create a dataset for LLM evaluation? What are different evaluation techniques?
15. Different agentic frameworks.
16. How to implement LangGraph-based agentic workflow?
17. What are different components of a LangGraph solution?
18. When to use agents vs when to do tool calls?
19. How can you handle tool-use errors inside an agent loop?
20. Prompting and how to enhance it.
21. Explain few-shot and zero-shot prompting.
22. Explain BERT architecture.
23. AWS/Azure pipeline for LLM deployment.
24. Can you talk about MCP? How does it work?

### Short answers (simple + interview ready)

1) **Explain the project.**

I built a GenAI solution for **[use case]** where users ask questions and get **grounded answers** from internal documents. In production, the core flow is: ingest docs → chunk → embed → store in vector DB → retrieve top-k → generate answer with citations. I focused on reliability (guardrails + eval), cost (caching + model routing), and scalability (FastAPI + async + containerized deployment).

2) **Explain the transformer architecture.**

Transformer is a neural network built around **attention** instead of recurrence. It has embeddings + positional encoding, then stacked blocks with **multi-head self-attention** and **feed-forward networks**, plus residual connections and layer norm. Encoder models (BERT) read full context; decoder models (GPT) generate tokens autoregressively.

3) **How does the sentence pass through an encoder layer?**

The sentence is tokenized into token IDs, converted to vectors via embeddings, then positional information is added. In the encoder layer, self-attention mixes information across tokens to produce context-aware representations. Then a feed-forward network transforms each token representation; residual + layer norm stabilize training. Output is a sequence of contextual vectors (one per token).

4) **Explain self-attention.**

Self-attention lets each token “look at” other tokens to build context. We project token vectors into **Q, K, V**, compute attention weights via $\text{softmax}(QK^T/\sqrt{d})$, then take a weighted sum of V. Multi-head attention repeats this in parallel to capture different relationships (syntax, coreference, etc.).

5) **LoRA vs QLoRA vs PEFT.**

- **PEFT** is the umbrella: techniques to adapt models without updating all weights.
- **LoRA** adds small low-rank adapter matrices and trains only those (fast + low memory).
- **QLoRA** is LoRA + **quantized base model** (e.g., 4-bit) to fit large models on smaller GPUs while training adapters.

6) **Temperature, Top-k, Top-p, Max tokens.**

- **Temperature**: randomness (lower = more deterministic).
- **Top-k**: sample only from the k most likely tokens.
- **Top-p (nucleus)**: sample from the smallest set whose cumulative prob ≥ p.
- **Max tokens**: cap on generated output length (controls cost + latency).

7) **Prevent LLM from accessing PII.**

In production, I use a layered approach: PII detection/redaction before prompts (NER + regex + DLP tools), strict retrieval filters/ACLs, least-privilege data access, and logging with masking. Add output filters to block PII leakage, and use policy prompts + allowlists for tools. Also separate environments and ensure training/feedback pipelines never store raw PII.

8) **Hallucinations and prevention.**

Hallucinations happen because the model generates likely text, not verified truth. Mitigation: RAG/tool-calling for facts, strong instructions (“answer only from context”), citations, lower temperature, re-rankers, and post-generation validation (rules + consistency checks). Add “I don’t know” behavior and evaluate with faithfulness metrics.

9) **Incorporate feedback mechanisms.**

Collect user feedback (thumbs up/down + reason), capture traces (query → retrieved chunks → answer), and label failures. Use that to improve retrieval (chunking/metadata), prompts, and add regression tests. For larger improvements, create a supervised dataset and do prompt tuning / PEFT on a smaller model, plus continuous evaluation gates in CI.

10) **Components of RAG.**

- Ingestion: loaders (PDF/HTML/etc), cleaning, parsing
- Chunking + metadata
- Embeddings model
- Vector store/index (and optionally keyword index)
- Retriever (similarity + filters)
- Re-ranker (optional, improves precision)
- Prompt template (grounding + citations)
- LLM generation
- Guardrails (PII, safety, formatting)
- Observability + evaluation (latency, relevance, faithfulness)

11) **Indexing and vector DB.**

Indexing means organizing embeddings so similarity search is fast at scale. Vector DB stores vectors + metadata and supports nearest-neighbor search using structures like HNSW/IVF. At query time you embed the query, search the index, optionally filter by metadata, and return top matches.

12) **Chunking techniques.**

- Fixed-size (token/character)
- Recursive (split by headings → paragraphs → sentences)
- Semantic chunking (split on topic shifts)
- Sliding window/overlap (retain context)
- Structure-aware (tables, code blocks, sections)

13) **Improve RAG solutions.**

Better retrieval (metadata, hybrid search, re-ranking), better chunking, better embeddings, query rewriting, and caching. Add evaluation + feedback loops to find failure modes. Use multi-step retrieval for complex queries and enforce citations/grounding to reduce hallucinations.

14) **Create eval dataset + evaluation techniques.**

Create a QA set from real queries + curated answers with references to source passages. Include edge cases (ambiguous, long, conflicting docs). Evaluate at two levels: retrieval (precision/recall@k, MRR) and generation (answer relevance, faithfulness/groundedness, citation correctness), plus latency/cost. Use human review for high-stakes workflows.

15) **Agentic frameworks.**

Common options: LangChain agents, LangGraph, AutoGen, CrewAI, Semantic Kernel, custom orchestrators. Choice depends on tool ecosystem, state management, observability, and how much control you need over the loop.

16) **Implement LangGraph workflow.**

Define a state schema, then build a graph of nodes: “plan”, “retrieve”, “tool call”, “reflect”, “final”. Add conditional edges based on model outputs and validation signals. Enforce max-steps/timeouts, and log every node input/output for debugging.

17) **Components of a LangGraph solution.**

State (typed data), nodes (functions/LLM calls/tools), edges (control flow), routers/conditions, memory/checkpointing, tool registry, and observability (traces). In production you also add guardrails, retries, and cost/latency budgets.

18) **When to use agents vs tool call?**

Use **simple tool calls** when the workflow is known and deterministic (retrieve → answer). Use **agents** when tasks are multi-step, branching, or require planning across tools (e.g., “compare policies across docs, compute totals, then draft email”). If it can be expressed as a pipeline, prefer pipeline for reliability.

19) **Handle tool-use errors in an agent loop.**

Add structured error handling: retries with backoff, error-to-user mapping, and fallback tools. Validate tool inputs/outputs with schemas, and stop the loop on repeated failures (max attempts). Log traces so you can reproduce; also teach the agent to ask clarifying questions when inputs are missing.

20) **Prompting and enhancements.**

Use clear role + task + constraints + output format. Add examples (few-shot) only when needed. Keep prompts modular (system/base + task + RAG context). In production: version prompts, run A/B tests, and add eval gates to prevent regressions.

21) **Few-shot vs zero-shot prompting.**

- **Zero-shot**: just instructions (fast, cheap, works for many tasks).
- **Few-shot**: add examples to teach style/format and reduce errors.
Use few-shot when formatting is strict or the task is tricky; otherwise keep it simple.

22) **BERT architecture.**

BERT is an **encoder-only** transformer trained with masked language modeling (predict masked tokens) and next-sentence style objectives (varies by version). It outputs contextual embeddings for each token and is strong for understanding tasks (classification, NER, retrieval). Unlike GPT, it’s not primarily a generator.

23) **AWS/Azure pipeline for LLM deployment.**

Typical pipeline: code + prompts in repo → CI tests/evals → build Docker → push to registry → deploy API (FastAPI) on Kubernetes/managed container service → connect to LLM provider (Azure OpenAI/Bedrock) + vector store/search → observability (logs/traces/metrics) + secrets management + autoscaling. Add canary rollout and cost monitoring.

24) **MCP (Model Context Protocol).**

MCP is a standard way for models/agents to access external context and tools through a **server** interface. In practice: an MCP client (agent) discovers tools/resources exposed by MCP servers, calls them with structured inputs, receives structured outputs, and uses that as context for responses. It improves interoperability: one agent can work with many tool providers consistently.

## EY HR (1:00 PM, 1/19/2026) — Round 1 (Docs / Multimodal)

1. Give introduction; highlight projects in GenAI space.
2. Explain how you implemented the project (picked 1 project from introduction).
3. How do you extract tables from PDF? How to do extraction if there are multiple cells?
4. How do you validate that parsing information from a PDF having multiple tables has been done correctly and information from all tables has been parsed properly?
5. Talk about new GenAI technique for parsing (torchvision, vision.io).
6. How do you process table data into chunking step?
7. How do you process images into chunking step? Which OCR technique / advanced technique is best suited?
8. For long PDFs with long-context text, what is the most efficient way of chunking?
9. Explain different types of chunking.
10. How do you process audio data (e.g., board meeting recording)?
11. How to do prompt management so prompts can be reused / for context management?
12. How do you deploy AI solutions/models?
13. How do you handle memory leaks or memory management?
14. How do you use multi-agent MCP models? How to implement it?
15. How do you use ReAct agent with the help of AutoGen?
16. Explain different types of agents.
17. Talk about latest GenAI news with respect to reasoning models.
18. Explain architectural difference between GPT-o3 and GPT-5. Also between DeepSeek and GPT.
19. Talk about general top 3 challenges in GenAI solutions.

### Short answers (Round 1)

1) **Intro + projects.**

I’m a GenAI engineer focused on building production systems: RAG, agents, and multimodal document pipelines. Recently I shipped **[project]** end-to-end: ingestion → retrieval → LLM → API → monitoring, with measurable improvements in accuracy and latency.

2) **How you implemented the project.**

I start from the business KPI, then design architecture (RAG vs fine-tune), pick the right model, add guardrails, and build a clean API. I validate with offline evaluation and then monitor production usage and failure modes.

3) **Extract tables from PDF (multi-cell).**

I use a layered approach: try native extraction first (PDF text structure), then table parsers, and fall back to vision-based extraction for scanned PDFs. For multi-cell and merged cells, I rely on table structure detection (rows/columns/lines) and reconstruct the grid with coordinates.

4) **Validate parsing across multiple tables.**

I validate completeness and correctness: page coverage, table count checks, schema validation (expected columns), and consistency checks (totals, row counts). I also sample QA with visual overlays (bounding boxes) and maintain regression tests on a labeled PDF set.

5) **New GenAI technique for parsing.**

Modern pipelines use vision-language models for layout understanding: detect regions (title, table, figure), then extract structured content with OCR + layout models. The key is combining layout detection with structured reconstruction and confidence scoring.

6) **Process table data into chunking.**

I convert tables into a text form that preserves structure: header + row key/value statements, and store metadata like source page and table id. For retrieval, I chunk by logical groups (table sections) rather than fixed tokens.

7) **Process images into chunking + best OCR.**

For images, I run OCR + layout detection, then store extracted text with image/page metadata. Best choice depends: printed text → classical OCR; complex layouts/scans → OCR + layout model; screenshots/diagrams → VLM + captions.

8) **Efficient chunking for long PDFs.**

Use structure-aware chunking: split by headings/sections, keep overlap, and attach metadata (section title, page). Add summarization for very long sections, and use hierarchical retrieval (section → paragraph).

9) **Chunking types.**

Fixed-size, recursive, semantic, sliding window overlap, and structure-aware chunking.

10) **Process audio data (board meeting).**

Pipeline: speech-to-text with diarization (who spoke), then segment by topic/time, optionally summarize per segment, and store chunks with timestamps + speakers. Retrieval returns segments + timestamps; final answer can cite the exact time range.

11) **Prompt management / reusability.**

I store prompts as versioned templates with variables, use a prompt registry, and keep separate system/base prompts from task prompts. Add tests and evaluations per prompt version, and log prompt versions in production traces.

12) **Deploy AI solutions/models.**

I containerize the service, deploy via Kubernetes/managed containers, and integrate with managed LLM endpoints. I set up autoscaling, secrets, observability, and safe rollouts (canary) with evaluation gates.

13) **Memory leaks / memory management.**

I monitor RSS/heap growth, use profiling, and fix root causes (dangling references, large caches, unbounded queues). For ML/LLM apps: control batch sizes, stream responses, limit concurrency, and implement bounded caches with TTL.

14) **Multi-agent MCP models.**

I separate responsibilities: a coordinator agent routes tasks to specialist agents (retrieval, data QA, formatting). MCP standardizes tool access so each agent can call the same tool servers consistently, with shared tracing and policies.

15) **ReAct agent with AutoGen.**

I define tools and schemas, then run a loop: reason → call tool → observe → refine. AutoGen helps coordinate agent conversations and tool usage; in production I still add strict guards (max steps, validation, fallbacks).

16) **Types of agents.**

Single-agent tool user, planner-executor, multi-agent (specialists), and workflow/graph-based agents. In production, graph-based is usually more controllable.

17) **Latest GenAI news (how to answer).**

I talk about trends: stronger reasoning + tool use, multimodal models, smaller efficient models, better RAG (hybrid + re-ranking), and increased focus on safety/evaluation. I connect it to business value: accuracy, cost, governance.

18) **GPT-o3 vs GPT-5 / DeepSeek vs GPT (architecture difference).**

Public details are limited, so I explain differences at the system level: reasoning quality, latency/cost, context length, tool calling, and safety. I describe how I’d benchmark them on our tasks (eval set + cost/latency budgets) instead of guessing internals.

19) **Top 3 challenges in GenAI solutions.**

Reliability (hallucinations/grounding), data governance (PII/access control), and cost/latency at scale. The fix is strong retrieval + eval + guardrails, good observability, and careful model/prompt optimization.

## EY HR (1:00 PM, 1/19/2026) — Round 2 (Agentic / RAG)

1. Different agentic frameworks.
2. How to decide if an agentic approach is required vs a supervised workflow?
3. How to implement LangGraph-based agentic workflow?
4. How to evaluate results of an LLM model?
5. How to prevent LLM from accessing PII information?
6. LLM hallucinations and prevention techniques.
7. How to incorporate feedback mechanisms in LLM?
8. Cache memory.
9. Types of LLM and fine-tuning.
10. LLM guardrails.
11. Prompting and enhancements.
12. Chunking and its types.
13. Chunking with different kinds of files.
14. RAG and RAGAs.
15. Indexing.
16. Vector DB and knowledge base.
17. Disambiguate query.
18. LangChain/LangGraph.
19. Embeddings.
20. Encoder/Decoder/BERT.
21. Agentic RAG.
22. GPT/Llama and variants.
23. AWS/Azure pipeline for LLM deployment.

### Short answers (Round 2)

1) **Agentic frameworks.**

LangChain agents, LangGraph, AutoGen, CrewAI, Semantic Kernel, or custom graphs. I prefer LangGraph when I need deterministic control and observability.

2) **Agentic vs supervised workflow.**

If the steps are known and stable, use a workflow (more reliable). Use agents when the task requires planning, branching, or unknown steps. Always add budgets (steps/time/cost) for agents.

3) **Implement LangGraph.**

Define state, nodes, conditional edges, tool schemas, and checkpointing. Add validation nodes, retries, and stopping rules. Instrument traces for every node.

4) **Evaluate LLM results.**

Use offline eval sets + metrics (accuracy/relevance/faithfulness) and human review for critical flows. Track production metrics: refusal rate, user feedback, and error categories.

5) **Prevent PII.**

Redact before prompting, enforce document ACLs, mask logs, and validate outputs. Add policy prompts and output filters.

6) **Hallucination prevention.**

Grounding via RAG/tool calls, stronger prompts, lower temperature, citations, re-rankers, and post-checks.

7) **Feedback mechanisms.**

Collect feedback + traces, label failures, build regression tests, and iterate retrieval/prompt. For bigger wins: fine-tune via PEFT or train a re-ranker.

8) **Cache memory.**

Cache embeddings, retrieval results, and stable tool outputs with TTL. For chat memory, store summarized state and keep PII out. Make caches bounded to avoid memory bloat.

9) **Types of LLM and fine-tuning.**

Decoder-only (GPT/Llama), encoder-only (BERT), encoder-decoder (T5). Fine-tuning options: full fine-tune, PEFT/LoRA, instruction tuning, preference tuning. I choose based on data, budget, and update frequency.

10) **LLM guardrails.**

Input filters (PII), prompt policies, tool allowlists, output validation, and fallback behavior. Also add monitoring + red teaming.

11) **Prompting enhancements.**

Structured prompts, examples, strict output schemas, modular templates, and evaluation-driven iteration.

12) **Chunking types.**

Fixed, recursive, semantic, overlap windows, structure-aware.

13) **Chunking across file types.**

Text: headings/paragraphs; PDFs: layout-aware; tables: row/column representation; code: function/class blocks; audio: timestamped segments; images: OCR+captions.

14) **RAG and RAGAs.**

RAG = retrieve context + generate grounded answer. RAGAS is an evaluation framework that scores retrieval and generation quality (relevance, faithfulness, etc.).

15) **Indexing.**

Build fast ANN indexes, store metadata, and tune recall/latency. Consider hybrid indexes for keyword + vector.

16) **Vector DB and knowledge base.**

Vector DB stores embeddings; knowledge base is the curated corpus + metadata + access rules. Production systems combine both.

17) **Disambiguate query.**

Ask a clarifying question or run query rewriting to produce a clearer retrieval query. Use metadata filters (time, product, region) when available.

18) **LangChain/LangGraph.**

LangChain is a framework for chains/tools/agents; LangGraph adds graph-based control and stateful workflows for more reliable agents.

19) **Embeddings.**

Embeddings map text to vectors so similar meaning is close in space. They power semantic search and retrieval.

20) **Encoder/Decoder/BERT.**

Encoder reads full context (good for understanding). Decoder generates next token (good for generation). BERT is encoder-only.

21) **Agentic RAG.**

RAG + agent loop: plan retrieval, run multiple retrieval steps, call tools, validate citations, then answer. Useful for complex tasks, but requires budgets and validation.

22) **GPT/Llama variants.**

Compare by capability, cost, context length, tool calling, and safety. Choose based on task benchmarks, not hype.

23) **Cloud pipeline.**

CI/CD with eval gates, Docker, secure secrets, autoscaling, logs/traces, and cost monitoring.

## EY HR (1:00 PM, 1/19/2026) — Round 3 (GCP / Knowledge Graph)

1. Explain the project/work done in GenAI.
2. What is your understanding of GenAI and RAG?
3. How confident are you in building data pipelines?
4. Your experience in GCP; which GCP services have you worked on?
5. Have you done GenAI orchestration for any projects?
6. Your experience with knowledge graphs.
7. How do you create a graph for table data with meaningful information?
8. Have you done any agentic solutioning during research work?
9. Have you worked in multimodal in any of your use cases?
10. Which is the best multimodal model in the current scenario?
11. What is your expectation in GenAI work?

### Short answers (Round 3)

1) **Project/work in GenAI.**

I’ve built production GenAI systems using RAG + agents + APIs. I focus on grounding, evaluation, and scaling, not just demos.

2) **Understanding of GenAI and RAG.**

GenAI generates content; RAG grounds generation using retrieved enterprise context. It reduces hallucinations and improves freshness without changing model weights.

3) **Confidence in data pipelines.**

I’m comfortable building ingestion + transformation + orchestration with quality checks and monitoring. I design for idempotency, retries, and clear lineage.

4) **GCP experience (services).**

I frame this as: storage (GCS), compute (Cloud Run/GKE), data (BigQuery), orchestration (Composer), streaming (Pub/Sub), and security (IAM/Secrets). I map services to the project’s needs.

5) **GenAI orchestration.**

Yes: orchestration for ingestion jobs, embedding refresh, evaluation runs, and agent workflows. I use schedulers + event-driven triggers and keep everything observable.

6) **Knowledge graphs.**

KGs represent entities and relationships. They’re great for disambiguation, reasoning over structured relations, and improving retrieval with entity linking.

7) **Graph for table data.**

Extract entities from headers/rows, normalize to canonical IDs, then create nodes (Entity) and edges (Relation) from key/value and foreign-key style columns. Validate with constraints and domain rules.

8) **Agentic solutioning in research.**

I use agents for multi-step tasks (retrieve → compute → validate → summarize) with strict stopping rules and logging.

9) **Multimodal experience.**

I’ve worked with pipelines combining text + images (OCR/layout) and sometimes audio (ASR). The key is consistent metadata and evaluation per modality.

10) **Best multimodal model.**

I avoid “one best” and instead choose by benchmarks: accuracy on our docs, latency, cost, and deployment constraints. I can run a small bake-off with a labeled set.

11) **Expectation in GenAI work.**

I’m looking to build production-grade GenAI: measurable impact, strong evaluation, and responsible deployment (security/PII/governance).

## EY HR (1:00 PM, 1/19/2026) — Round 4 (Advanced RAG + APIs)

1. Advanced RAG pipeline and query correction.
2. How to debug the LLM flow in case of hallucination?
3. How to enhance retrieval quality?
4. How to add or create metadata of a document?
5. Experience on GCP and data engineering work.
6. How to create data pipeline in GCP.
7. LangGraph flow + agentic AI; fine-tuning of LLM and how its data is created.
8. Re-ranker.
9. How to do prompt engineering efficiently?
10. Audio/video data handling; working with APIs.
11. How to scale FastAPI?

### Short answers (Round 4)

1) **Advanced RAG + query correction.**

Use query rewriting (intent + entities), metadata filters, hybrid retrieval, re-ranking, and multi-hop retrieval for complex questions. Add spelling/term normalization and domain synonyms.

2) **Debug hallucination in LLM flow.**

Trace the full pipeline: retrieval results, prompt, model output. Check whether the answer is unsupported by context (faithfulness). Fix by improving retrieval, tightening prompt, and adding validation/citation enforcement.

3) **Enhance retrieval quality.**

Better chunking, better embeddings, hybrid search, re-rankers, metadata enrichment, and query rewriting. Measure improvements with retrieval metrics and grounded-answer metrics.

4) **Create metadata for documents.**

Auto-extract metadata (title, section, date, author, product) and add business tags (department, access level). Store it in the index so retrieval can filter and rank better.

5) **GCP + data engineering work.**

I talk about ingestion, transformations, scheduling, data quality, and governance, and how I’d integrate it with GenAI embedding refresh and eval jobs.

6) **Create data pipeline in GCP.**

Use GCS for landing, Dataflow/BigQuery for transforms, Composer/Workflows for orchestration, Cloud Run/GKE for services, and IAM/Secrets for security. Add monitoring and retries.

7) **LangGraph + fine-tuning data.**

For agentic graphs: log traces and outcomes, label good/bad tool use, then create instruction + preference datasets. For fine-tuning, I start with PEFT/LoRA and strict evaluation to avoid regressions.

8) **Re-ranker.**

A re-ranker takes top retrieved chunks and re-sorts them using a stronger cross-encoder / LLM-based scorer. It improves precision and reduces irrelevant context.

9) **Prompt engineering efficiently.**

Keep prompts modular, use structured outputs, reduce tokens, and iterate with evaluation. Maintain prompt versions and test suites.

10) **Audio/video handling with APIs.**

Audio: ASR + diarization + segmentation + timestamps. Video: extract audio + key frames, caption frames, then index both text and time ranges. Serve via API with references to timestamps.

11) **Scale FastAPI.**

Use async endpoints, connection pooling, background workers for heavy jobs, caching, and horizontal scaling behind a load balancer. Put rate limits, timeouts, and circuit breakers around LLM calls.

## EY HR Note

- These are questions asked in previous interviews.
- Additional prep topics:

  - GCP Data Engineering (GenAI Pipelines)
  - Agentic AI (LangGraph, ADK, AutoGen, CrewAI)
  - API creation & Docker build process

# HOW TO POSITION YOURSELF (VERY IMPORTANT)

Your mindset in answers:

You are solution-oriented, not academic

You design production-ready GenAI systems

You understand trade-offs, architecture, scaling, and risk

You talk in real-world examples

Use phrases like:

“In production…”

“From an architecture perspective…”

“When scaling this on Azure/AWS…”

“To make this reliable and cost-efficient…”



CORE ARCHITECTURE YOU MUST KNOW (AGENTIC + RAG)
🔹 High-level GenAI Architecture (Say This Confidently)

“A typical GenAI system I design includes:

An LLM (Azure OpenAI / OpenAI / Bedrock)

A RAG layer using vector DB (FAISS / Pinecone / Azure AI Search)

An Agentic orchestration layer (LangChain / custom agents)

APIs built in FastAPI

Observability, evaluation, and guardrails

Deployed using Docker + Kubernetes on cloud”

3️⃣ INTERVIEW QUESTIONS & MASTER-LEVEL ANSWERS
🔥 LLMs & GenAI Fundamentals
Q1. What is an LLM and how does GPT work?

Answer:

“LLMs are transformer-based models trained on large corpora using self-attention. GPT uses an autoregressive approach where each token is predicted based on prior context. The attention mechanism allows it to capture long-range dependencies efficiently, which is critical for reasoning and generation tasks.”

Q2. Difference between GPT-3.5 and GPT-4?

Answer:

“GPT-4 has better reasoning, instruction following, and reduced hallucinations. It performs better on complex multi-step tasks, coding, and contextual understanding. In enterprise systems, GPT-4 is preferred for agentic workflows, while GPT-3.5 is used for cost-sensitive use cases.”

🔥 Agentic AI (VERY IMPORTANT)
Q3. What is Agentic AI?

Answer (KEY ANSWER):

“Agentic AI refers to systems where LLMs can reason, plan, take actions, call tools, and iterate autonomously to achieve goals. Agents maintain state, decide next actions, and interact with APIs, databases, or other agents.”

Q4. How do agents work in LangChain?

Answer:

“LangChain agents use:

LLM as the reasoning engine

Tools (APIs, DBs, calculators)

Memory for context

A decision loop (Thought → Action → Observation)
This enables dynamic problem solving instead of static prompt-response.”

Q5. Difference between RAG and fine-tuning?

Answer:

“RAG retrieves external knowledge at inference time, ensuring up-to-date and explainable responses. Fine-tuning changes model weights and is expensive, less flexible, and harder to update. In enterprise systems, RAG is usually preferred.”

🔥 RAG (Retrieval Augmented Generation)
Q6. Explain RAG architecture.

Answer:

“Documents are chunked, embedded using models like text-embedding-ada-002, stored in a vector database. At query time, relevant chunks are retrieved via similarity search and injected into the LLM prompt for grounded responses.”

Q7. How do you reduce hallucinations?

Answer:

RAG with trusted data

Prompt grounding

Citation enforcement

Response validation

Temperature control

Post-generation validation rules

🔥 Azure OpenAI / Cloud
Q8. How do you use Azure OpenAI in production?

Answer:

“Azure OpenAI is used for enterprise-grade security, RBAC, private networking, and compliance. We deploy models behind APIs, use managed identity, log prompts/responses securely, and integrate with Azure AI Search for RAG.”

Q9. How do you scale GenAI systems?

Answer:

Async API calls

Caching embeddings

Load balancing

Model selection by complexity

Kubernetes auto-scaling

Cost monitoring

🔥 Python & API Development
Q10. How do you expose GenAI as an API?

Answer:

“Using FastAPI with async endpoints. The API handles authentication, request validation, calls LLM + retrieval layer, applies guardrails, and returns structured responses.”

Q11. How do you structure a GenAI codebase?

Answer:

/agents

/tools

/prompts

/retrievers

/services

/api
Clean separation improves maintainability.

🔥 NLP & Deep Learning
Q12. Explain transformers briefly.

Answer:

“Transformers rely on self-attention instead of recurrence, enabling parallelism and better long-context understanding.”

Q13. What NLP techniques are you familiar with?

Answer:

Tokenization

Embeddings

Attention

Named Entity Recognition

Text classification

Semantic search

🔥 GANs (Keep it Simple)
Q14. What are GANs?

Answer:

“GANs consist of a generator and discriminator in a competitive setup. They are mainly used for image, video, and data generation rather than LLM tasks.”

🔥 Chatbots & Rasa
Q15. Experience with Rasa?

Answer:

“Rasa is used for intent classification, entity extraction, and dialogue management. It integrates well with LLMs where Rasa handles flow control and LLM handles generative responses.”

🔥 ML Lifecycle
Q16. How do you retrain models?

Answer:

Monitor drift

Collect feedback

Retrain embeddings / classifiers

Validate performance

Redeploy via CI/CD

🔥 DevOps & Containers
Q17. How do you deploy GenAI apps?

Answer:

“Using Docker containers orchestrated by Kubernetes. CI/CD via Jenkins, infrastructure via Terraform or ARM templates, monitoring via ELK.”

🔥 Statistical Analysis
Q18. How do stats help in GenAI?

Answer:

“Used for evaluation metrics, A/B testing prompts, measuring accuracy, relevance, and confidence scoring.”

4️⃣ SYSTEM DESIGN QUESTION (YOU WILL GET THIS)
❓ “Design a GenAI chatbot for enterprise knowledge base”

Answer Structure:

Use Azure OpenAI GPT-4

RAG with Azure AI Search

LangChain agent orchestration

FastAPI backend

Authentication via Azure AD

Logging & monitoring

Docker + Kubernetes deployment

5️⃣ LAST-MINUTE POWER PHRASES (MEMORIZE)

Say these naturally:

“Agentic workflows”

“Grounded generation”

“Enterprise-grade GenAI”

“Contextual relevance”

“Production scalability”

“Cost-performance trade-offs”

“Observability and guardrails”

6️⃣ IF YOU WANT (VERY FAST)

I can:

🎯 Mock interview (rapid-fire)

🧠 Give you 10 must-memorize answers

🏗 Teach you Agentic AI in 30 minutes

🧪 Give coding + architecture questions

Just say:
👉 “Start mock interview” or “Give me quick revision”



LLM & GENAI CORE
Q1. What is temperature in LLMs?

Answer:

“Temperature controls randomness. Lower temperature gives deterministic, factual answers; higher temperature increases creativity. In production GenAI, I usually keep it between 0.2–0.5 for reliability.”

Q2. What are tokens and why do they matter?

Answer:

“Tokens are chunks of text processed by the model. They impact cost, latency, and context window. Optimizing token usage is critical for scaling GenAI systems.”

Q3. What is prompt engineering?

Answer:

“Prompt engineering is designing structured inputs that guide the LLM to produce accurate, grounded, and consistent outputs using instructions, examples, and constraints.”

Q4. Difference between system, user, and assistant prompts?

Answer:

System: Defines behavior and rules

User: Input query

Assistant: Model response
System prompts are critical for enforcing safety and tone.

2️⃣ AGENTIC AI (VERY HIGH FREQUENCY)
Q5. What problem does Agentic AI solve?

Answer:

“It enables multi-step reasoning, tool usage, decision-making, and autonomy — beyond single prompt-response interactions.”

Q6. How do agents decide next actions?

Answer:

“Using reasoning loops like Thought → Action → Observation. The LLM evaluates the current state and chooses the next tool or response.”

Q7. What is tool calling in LLMs?

Answer:

“Tool calling allows LLMs to invoke external APIs, databases, or functions to fetch real data instead of hallucinating.”

Q8. Stateless vs stateful agents?

Answer:

“Stateless agents handle single interactions. Stateful agents maintain memory and context across steps or sessions, essential for enterprise workflows.”

3️⃣ RAG (EXTREMELY COMMON)
Q9. Why is RAG important?

Answer:

“RAG grounds LLM responses using trusted enterprise data, reduces hallucinations, and avoids costly fine-tuning.”

Q10. How do you choose chunk size?

Answer:

“Typically 300–800 tokens. Too small loses context; too large reduces retrieval accuracy. I test and tune based on query patterns.”

Q11. What similarity search methods do you know?

Answer:

Cosine similarity

Dot product

Euclidean distance

Q12. How do you evaluate RAG performance?

Answer:

Retrieval precision

Answer relevance

Faithfulness to source

Latency and cost

4️⃣ HALLUCINATIONS & SAFETY (VERY COMMON)
Q13. Why do LLMs hallucinate?

Answer:

“Because they generate probabilistic text without real knowledge validation.”

Q14. How do you prevent hallucinations?

Answer:

RAG

Tool calling

Prompt constraints

Temperature control

Output validation

Q15. What are guardrails?

Answer:

“Guardrails are safety mechanisms that enforce policy, restrict output, and validate responses before returning them to users.”

5️⃣ CLOUD & DEPLOYMENT
Q16. Why use Azure OpenAI over OpenAI API?

Answer:

“Azure OpenAI provides enterprise security, private networking, compliance, RBAC, and integration with Azure services.”

Q17. How do you manage cost in GenAI?

Answer:

Model selection

Prompt optimization

Caching

Token limits

Batch processing

Q18. How do you deploy LLM apps in Kubernetes?

Answer:

“Containerize the app, configure autoscaling, manage secrets securely, and monitor performance and cost.”

6️⃣ PYTHON (VERY COMMON)
Q19. Why Python for GenAI?

Answer:

“Python has rich ML libraries, strong community support, and seamless integration with LLM frameworks like LangChain.”

Q20. How do you handle async LLM calls?

Answer:

“Using async/await with FastAPI to improve throughput and reduce latency.”

7️⃣ ML & DATA SCIENCE
Q21. Difference between supervised and unsupervised learning?

Answer:

“Supervised uses labeled data; unsupervised discovers patterns without labels.”

Q22. What is overfitting?

Answer:

“When a model performs well on training data but poorly on unseen data.”

Q23. How do you detect data drift?

Answer:

“By monitoring input distributions, embeddings similarity, and output quality over time.”

8️⃣ CHATBOTS (RASA / LLM)
Q24. How do Rasa and LLMs work together?

Answer:

“Rasa manages intents, entities, and dialogue flow, while LLMs generate dynamic responses for complex queries.”

Q25. Intent-based vs generative chatbots?

Answer:

“Intent-based bots are rule-driven; generative bots are flexible and context-aware.”

9️⃣ DEVOPS & MLOPS
Q26. What is MLOps?

Answer:

“MLOps automates the lifecycle of ML models from training to deployment and monitoring.”

Q27. CI/CD in GenAI?

Answer:

“Automates testing, deployment, prompt versioning, and rollback.”

10️⃣ SYSTEM DESIGN (VERY LIKELY)
Q28. Design a GenAI document Q&A system.

Answer Outline:

Ingest documents

Chunk + embed

Store in vector DB

Retrieve relevant chunks

Generate response using LLM

Apply guardrails

Deploy via API

11️⃣ BEHAVIORAL (DON’T IGNORE)
Q29. How do you explain GenAI to non-technical stakeholders?

Answer:

“I focus on business value, accuracy, risk, and ROI instead of model internals.”

Q30. How do you handle GenAI failures?

Answer:

“Fallback mechanisms, logging, human-in-the-loop, and continuous improvement.”