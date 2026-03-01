# Common Scenario-Based GenAI Interview Questions & Answers

---

## RAG-Based Scenarios

**Q1. Your RAG chatbot is giving correct documents but wrong final answers. How would you debug it?**

The retrieval is fine, so the bug is in the prompt or context assembly. The most common mistake is prompting the model to "be helpful" instead of "answer ONLY from the provided documents" — that alone causes a lot of hallucination. I'd also check whether the relevant text is getting buried deep in a long context. I'd log 10–20 failing examples, trace each stage, and 90% of the time the fix is in the prompt.

---

**Q2. Users say the system is hallucinating even though you are using RAG. What could be the reasons?**

A few common causes: the right document exists but didn't rank in top-k (fix: hybrid search + reranker), too many irrelevant chunks are competing with the correct one (fix: lower top-k), or the prompt allows the model to infer beyond the documents (fix: add "if the answer isn't in the docs, say I don't know"). Another one — the answer spans two chunks but only one was retrieved, so add chunk overlap.

---

**Q3. Retrieval quality is poor. How would you improve it?**

I'd fix it in layers. First, check chunking — naive fixed-size chunks are usually the root cause; semantic chunking with overlap works much better. Second, add BM25 alongside vector search — hybrid retrieval consistently beats pure vector search. Third, add a cross-encoder reranker on the top candidates. I'd measure Recall@k and MRR before and after each change to make sure I'm actually improving things.

---

**Q4. Your vector DB is slow with millions of documents. How do you optimize it?**

First check: are you using an ANN index like HNSW? Exact search at this scale is always slow. Second: apply metadata filters (tenant, date) before vector search to shrink the candidate set. Third: two-stage retrieval — fast approximate search for top-100, then precise reranking on just those. Monitor P95 search latency and cache hit rate as your key signals.

---

**Q5. How would you design a multi-hop RAG system?**

Multi-hop is for questions that need facts from multiple documents — you do iterative retrieval instead of one shot. Step 1: retrieve for the original question, extract an intermediate fact. Step 2: use that fact to form a follow-up query and retrieve again. Step 3: combine both to answer. I'd keep a structured scratchpad to track intermediate facts, and add a final verification step to check consistency across hops.

---

## Agentic AI Scenarios

**Q1. Your agent is stuck in a loop calling the same tool repeatedly. How do you fix it?**

Add hard guards: max tool calls per session, and block the same tool being called with the same arguments more than twice. The deeper fix is making the agent state-aware — include prior actions and outcomes in the prompt so it can see it's going in circles. Also add a no-progress detector: if last N outputs look similar, force a replan or hand off to a human.

---

**Q2. A tool returns incomplete or corrupted data. How should the agent handle it?**

Validate the tool response schema before it enters the reasoning loop — if required fields are missing, don't pass it to the model. For transient failures, retry once with backoff. For consistently bad data, fall back to a secondary source or tell the user "I couldn't get complete information." The agent should be explicit about uncertainty, not guess with bad data.

---

**Q3. The agent is making wrong decisions while choosing tools. How would you improve tool selection?**

Usually comes from vague or overlapping tool descriptions. Rewrite them to be specific — what the tool does, expected inputs, and when NOT to use it. Add few-shot examples in the system prompt. For high-stakes routing, put a lightweight intent classifier in front of the planner. Log wrong choices from production and use them to build an eval set.

---

**Q4. How do you prevent an autonomous agent from taking unsafe actions?**

Define upfront which actions require human approval before executing — anything that writes data, spends money, or is irreversible. Use least-privilege credentials so the agent can only access what it actually needs. Run dangerous actions in dry-run mode first. Keep an immutable audit log of every high-risk action. The goal is to fail safely, not to trust the model perfectly.

---

**Q5. How do you add memory to an agent system?**

Structure memory in three layers: short-term (current session), episodic (past tasks), and long-term (persistent facts and preferences). Be selective about what you write — store high-value facts, not full conversation dumps. Retrieve by relevance + recency. Scan for PII before persisting anything, and support user-initiated deletion.

---

## Production & Scalability Scenarios

**Q1. Your Gen AI app suddenly gets 10x traffic. What will you do?**

First stabilize: enable rate limiting, queue requests, switch to a degraded mode if needed (cheaper model, cached answers). Then scale the API and retrieval layers independently — the bottleneck is usually the LLM provider rate limit, which doesn't autoscale. Prioritize paid / SLA-critical traffic. After the incident, do a bottleneck RCA and update capacity estimates.

---

**Q2. Latency increased after adding RAG. How do you optimize?**

Measure each stage first — retrieval, reranking, context assembly, LLM call. Common fixes: lower top-k (retrieving 20 chunks and reranking is way slower than 5), cache repeated queries, and pass fewer chunks to the LLM. I optimize for P95 latency, not average, because if the slow tail is bad, average hides it.

---

**Q3. Cost per request is too high. How do you reduce it?**

Biggest wins: semantic caching (handles repeated traffic for free), model tiering (use small models for simple requests, expensive ones only when needed), and pruning bloated system prompts. Don't over-retrieve — top-5 chunks instead of top-20 cuts context tokens significantly. Set up cost tracking per feature and find the 10% of requests burning 60% of cost.

---

**Q4. How do you monitor hallucination in production?**

Use groundedness scoring — check each claim in the response against the retrieved docs automatically. Also track citation match rate and contradiction rate. Sample high-risk responses for daily human QA. The key thing: alerts should trigger a rollback review, not just light up a dashboard nobody checks.

---

**Q5. How would you design a multi-tenant Gen AI system?**

Strict isolation: each tenant gets their own vector index, data store, and scoped auth tokens. Config per tenant for prompt templates, safety rules, and model tier. Per-tenant rate limits and quotas to prevent noisy-neighbor problems. Segment latency, cost, and quality metrics by tenant so issues can be isolated quickly.

---

## Evaluation & Quality Scenarios

**Q1. How would you evaluate an LLM without ground truth answers?**

Use rubric-based human review — score each response on factuality, relevance, completeness, and safety. Pairwise comparisons ("which answer is better") are often more reliable than absolute scores. For production, track task-success proxies: did the user stop asking follow-ups, did the issue escalate to a human? Segment by intent — a single global average hides a lot.

---

**Q2. How do you test prompt changes before deploying?**

Run offline regression on a fixed dataset with expected behaviors, then mandatory safety regression (jailbreak + injection tests). Then canary rollout to 5% of traffic with automatic rollback if quality drops. Always version the prompt alongside the model — you need to be able to reproduce a production failure exactly.

---

**Q3. How do you create a benchmark dataset for a domain-specific chatbot?**

Source queries from production logs and support tickets, not from what developers think users ask. Balance easy/medium/hard cases and include adversarial inputs. Have domain experts write or validate the expected answers with a scoring rubric. Refresh monthly with new production failures — benchmarks go stale fast.

---

## Safety & Alignment Scenarios

**Q1. A user tries prompt injection in your RAG system. How do you prevent it?**

Treat retrieved content as data, never as instructions. Explicitly tell the model in the prompt that document content is untrusted and it should never follow instructions found within it. Scan documents at ingestion for injection patterns. Tool execution must go through a policy check — it can never be triggered directly by document text. Run adversarial tests to verify this holds after every prompt change.

---

**Q2. The model generates biased responses. What steps will you take?**

First, measure it specifically — same question about different groups and check if outputs differ unfairly. Then find the source: training data, retrieval, or prompt. Mitigate with fairness constraints in the prompt, filter biased retrieval results, and for high-impact domains like hiring or healthcare, add a human review board. Audit again after every major model or prompt update.

---

## Advanced Design Scenarios

**Q1. Design a financial advisor agent with tool access.**

Planner agent with specialized tools: market data, portfolio risk calculator, compliance rule checker, and an explainer for human-readable output. Every recommendation must cite retrieved evidence — no model guesses about asset performance. Regulated actions require explicit human approval. Full audit trail with inputs, tool calls, outputs, and which rule approved or rejected each step.

---

**Q2. Design a document summarization system for 10M PDFs.**

Distributed ingestion pipeline with OCR, parsing, and quality checks — flag low-confidence docs before summarization. Hierarchical summarization: chunks → section summaries → document summary. Always store source text alongside summaries for traceability. Use incremental updates (don't re-summarize unchanged docs) and model tiering for cost control.

---

**Q3. How would you build a multi-agent collaboration system?**

Specialized agents: planner, retriever, analyst, critic, synthesizer. All agents communicate via a standard message schema (task, evidence, confidence, next action) — no free-form text passing. An orchestrator manages flow, retries, and budgets. The critic agent is the most commonly skipped and the most commonly regretted — without it, errors compound unchecked.

---

**Q4. How would you add real-time learning to an agent?**

Don't update live model weights in real time — too risky. Instead, update components: retrain the retriever on failure cases, update prompt examples based on recurring user corrections. Use implicit signals like escalations and re-asks alongside explicit feedback. Always deploy behind a canary with rollback ready.

---

**Q5. How do you handle long-term memory in agentic systems?**

Structured schema: facts, preferences, decisions, constraints — stored separately from session history. Each memory item has source, confidence, and last-validated timestamp. New evidence that contradicts old memory marks it superseded, not overwrites blindly. Set TTLs on stale memories and support user deletion — GDPR right-to-erasure applies here too.

---

## Critical RAG Failure Scenarios

**Q1. Your RAG system works in testing but fails in production. Why?**

Usually distribution shift — real users write messier, more ambiguous queries than your test set. Also: stale index (data changes in prod but not in staging), infra differences (different timeouts or rate limits), and no production monitoring loop so degradation goes unnoticed. Fix: mirror production traffic in staging and run continuous eval on live workloads.

---

**Q2. Retrieved documents are correct, but model still hallucinates. What's wrong?**

Almost always a prompt problem. The relevant text is there but buried in noisy context — fix by reranking aggressively and keeping only top 3–4 chunks. Also check if the prompt says "be complete" or "be helpful" — that gives the model permission to fill gaps. Add a citation requirement: force the model to cite the exact passage it's drawing from.

---

**Q3. Your chunk size is 1000 tokens. Accuracy is low. What will you change?**

1000 tokens is usually too large — the embedding becomes a diluted average and relevant chunks rank lower than they should. Move to 300–500 tokens with 50–100 token overlap. Use structure-aware splitting (paragraphs, headings) instead of blind token cuts. Measure Recall@k before and after — don't guess.

---

**Q4. Users ask multi-part questions. RAG answers only partially. How to fix?**

Decompose the question into sub-questions, retrieve separately for each, and answer each part explicitly. Add instruction to the prompt requiring coverage of every sub-question. Use structured output with one section per sub-question — incompleteness becomes obvious and measurable.

---

**Q5. How would you design citation-based answering to reduce hallucination?**

Each retrieved chunk gets an ID. The model must cite chunk IDs for every claim. A post-processing step verifies the cited chunk actually supports the claim semantically. If no citation exists for a claim, output "insufficient evidence" rather than guessing. Track unsupported citation rate as your main quality signal.

---

**Q6. Your embedding model changed. How do you migrate safely?**

Never hard cut over — the old and new embedding spaces are incompatible. Run both indexes in parallel, re-embed the full corpus in batches, and shadow-evaluate by comparing both indexes on live queries. Cut over only when the new index meets or beats key metrics. Keep the old index for a couple of weeks as a rollback option.

---

## Agent System Failure Scenarios

**Q1. Agent chooses wrong tool repeatedly. How do you improve tool routing?**

Rewrite tool descriptions to be specific: purpose, expected inputs, and when NOT to use it — that last part is what's usually missing. Add few-shot examples of correct choices. For high-stakes routing, put a lightweight intent classifier before the LLM planner. Log and label wrong choices from production and use them as training signal.

---

**Q2. Tool execution is expensive. How do you reduce unnecessary calls?**

Before calling a tool, check if memory or cache can answer it. Set a confidence threshold — only call tools when the model can't confidently answer from existing context. Batch similar calls if the tool supports it. Give the planner a visible tool-call budget per request and it will respect it.

---

**Q3. Agent memory grows too large. How do you manage long-term memory?**

Write policy: only store things with lasting utility — preferences, key decisions, facts that will be needed again. Compress episodic memory into factual summaries at end of session. Set TTLs on low-importance memories so they expire. Periodically deduplicate semantically similar entries.

---

**Q4. Agent fails in multi-step reasoning tasks. How do you debug?**

Trace every step — don't just look at the final answer. Log planned action, tool inputs/outputs, and what decision was made at each hop. Find the first step that went wrong, not the last. Add a verifier after each reasoning hop to catch errors early before they compound through the chain.

---

**Q5. Agent output breaks JSON format required by downstream systems. Fix?**

Use structured output / JSON mode from the LLM provider — this forces valid JSON at generation time. If unavailable: validate before passing downstream, and if invalid, send a correction prompt with the expected schema. Run schema contract tests after every model or prompt update to catch regressions early.

---

**Q6. How do you prevent tool prompt injection attacks?**

Treat all tool outputs as untrusted data — never feed raw tool output back to the planner as instructions. Wrap tool results in a labeled container so the model knows it's data, not a command. Define an approved action list and block anything not on it regardless of what the tool says. Run adversarial tests with injection payloads in tool outputs regularly.

---

## Scalability & Production Issues

**Q1. Token usage suddenly increased 40%. How do you investigate?**

Diff what changed recently — prompt templates, model version, retrieval top-k, response format settings. Segment usage by endpoint and tenant to find where the spike is concentrated. Common culprits: increased top-k, bloated system prompt, or a changed stop condition causing longer outputs. Set hard token budgets per request and alert on drift.

---

**Q2. P95 latency increased. Where do you check first?**

Use distributed tracing to pinpoint which stage got slower — retrieval, LLM call, or post-processing. If it's the LLM: check provider throttling. If it's retrieval: check index size growth or cache hit rate drop. Mitigation while you investigate: load shedding on non-critical traffic and fallback to a faster model.

---

**Q3. How do you design high availability for LLM systems?**

Abstract the LLM call behind a provider interface so you can route to a secondary (Azure OpenAI, Anthropic, local model) when the primary fails. Stateless API servers + replicated vector DB + async queues for non-real-time work. Set degradation levels: full response → cached response → simplified response → graceful message. Track SLOs and error budgets.

---

**Q4. How do you handle API rate limits from LLM provider?**

Adaptive client-side throttling watching response headers. Priority queue so user-facing requests go before batch jobs. Exponential backoff with jitter on retries — never all servers retrying at the same second. Spread load across multiple API keys or a second provider for burst absorption.

---

**Q5. Your vector DB becomes a bottleneck. What next?**

If it's throughput: add read replicas and a caching layer. If it's individual query latency: tune ANN parameters (trade a small recall hit for big latency gain) and apply metadata pre-filters. Move retrieval to a dedicated autoscaling service. For large scale, shard by tenant or domain — hot partitions in memory, cold ones on disk.

---

## Cost Optimization Scenarios

**Q1. Your monthly LLM bill doubled. What steps do you take?**

First add cost attribution by feature, endpoint, and tenant — you can't fix what you can't see. Then find the top spenders: usually 10–20% of requests drive 60–70% of cost. Quick wins: semantic caching, model downtiering for simple requests, and auditing bloated system prompts. Set per-tenant quotas and do a weekly cost-quality review.

---

**Q2. How do you decide between fine-tuning vs RAG?**

RAG when knowledge changes frequently, you need citations, or the dataset is too large for context. Fine-tuning when you're teaching behavior, style, or format that doesn't change often. Win condition for fine-tuning: "model should act differently." Win condition for RAG: "model needs different knowledge." In practice the best setups use both: fine-tune for behavior + RAG for up-to-date facts.

---

**Q3. How do you reduce context size without losing accuracy?**

Fix retrieval first — top-3 reranked chunks beats top-10 raw every time. Compress chunks before sending: extract only sentences directly relevant to the query. Deduplicate near-identical chunks. Trim system prompt of anything not earning its tokens. Always measure faithfulness and completeness before and after — context reduction should improve quality, not hurt it.

---

## Security & Safety Scenarios

**Q1. User attempts prompt injection through document content. Prevent?**

Tell the model explicitly in the prompt that document content is untrusted data and to never follow instructions within it. Scan docs at ingestion for injection patterns and sanitize. Wrap retrieved content in a labeled container before passing to the model. Tool execution must go through a policy check — document text cannot trigger it directly. Run injection tests regularly in your eval suite.

---

**Q2. Model exposes sensitive data from training. What safeguards?**

Scrub PII, credentials, and secrets from training data before fine-tuning — non-negotiable. Run a DLP filter on every output before it reaches the user. RBAC at the API level for sensitive workflows. Probe the model periodically for memorization leakage and document data lineage for compliance.

---

**Q3. How do you handle PII in a RAG system?**

Detect and redact or tokenize PII at ingestion before it goes into the vector index. Filter PII-containing docs from retrieval results unless the user is authorized. Run a PII filter on model outputs too — the model might infer or hallucinate it even if it wasn't retrieved. Log every PII access with purpose metadata and support GDPR/CCPA deletion workflows.

---

**Q4. How do you ensure compliance in finance/healthcare domains?**

Encode regulatory rules as runtime guardrails that run before every response — not optional, not overridable. Mandate human sign-off for regulated decisions (investment recommendations, clinical suggestions). Retain full context for every response: retrieved docs, prompt, model version, output — you need to reconstruct decisions months later. Run a compliance-specific test suite before every release.

---

## Advanced Architecture Scenarios

**Q1. Design a multi-agent system for research automation.**

Specialized pipeline: Planner → Retriever → Analyst → Critic → Report Generator. All agents write to a shared evidence graph with source references — prevents duplicate work and makes the report traceable. Cap steps, tokens, and wall-clock time per task. The Critic is the agent teams most often skip and most often regret — without it, factual errors compound unchecked.

---

**Q2. How do you coordinate agents with different roles?**

Standard message schema between all agents: task, evidence, confidence, next_action — no free-form text. An orchestrator manages sequencing, retries, and conflict resolution. Each agent gets a shared global objective score, not just a local one — otherwise you get locally optimal but globally bad behavior. Deadlock prevention: timeout + forced replan if no progress.

---

**Q3. When would you avoid using an agent system?**

If the task is deterministic — use a rule, SQL query, or script instead. Much faster, cheaper, and more reliable. Avoid if you have sub-500ms latency requirements (agent planning loops add overhead). Avoid in high-compliance domains without strong guardrails. Principle: use the simplest architecture that solves the problem.

---

**Q4. How do you design fallback when LLM is down?**

Multi-tier fallback: secondary provider → smaller local model → rule-based/template responses. Circuit breaker: stop retrying the primary immediately on failure, don't let calls pile up. Queue non-real-time jobs to process after recovery. Tell users clearly when they're in degraded mode — hiding it erodes more trust than acknowledging it.

---

**Q5. How do you evaluate reasoning ability of an agent?**

Evaluate the process, not just the final answer — did it plan correctly, call the right tools, produce sound intermediate conclusions? Use multi-step task benchmarks measuring step-level success rate. Counterfactual tests: change one input fact and verify the conclusion changes appropriately — if it doesn't, the agent is guessing, not reasoning.

---

**Q6. How would you build self-correcting agents?**

Core pattern: draft → critique → revise. The critique must use external tools and rules for verification, not just self-judgment ("do I look right to myself" is unreliable). Store recurring failure patterns in the critique checklist for compound improvement. Hard limit: max 2–3 correction attempts, then escalate — don't spin in loops.

---

## Frequently Asked Interview Questions

**Q1. How do you debug hallucinations?**

Reproduce reliably with fixed settings. Do a claim-level audit — go sentence by sentence and check whether each claim is supported by retrieved docs. Fix order: retrieval quality → prompt constraints → citation requirement → fallback "I don't know." Add the failing examples to your eval suite so the same pattern gets caught in future deploys.

---

**Q2. How do you measure RAG retrieval quality?**

Three metrics: Recall@k (is the right doc in top-k?), MRR (where does it rank?), and answer support coverage (are final answers actually grounded in what was retrieved?). Build a hold-out eval set with hard negatives — similar-but-wrong docs that should not be retrieved. Measure after every change to retrieval config.

---

**Q3. How do you improve answer grounding?**

Simplest fix: update the prompt to "answer only from the provided documents, cite the source for each claim, and say I don't know if it's not there." Pass 3–4 focused, high-quality passages — not 10 full chunks. Add a post-generation verifier to check each claim maps to a retrieved passage. Surface citations in the UI so users can verify.

---

**Q4. How do you productionize a prototype RAG system?**

The gap is reliability, observability, and safety — not the AI logic. Harden every component with retries and fallbacks. Log the full trace per query. Set SLOs upfront (P95 latency, availability, hallucination rate). Add security: auth, tenant isolation, PII controls, audit logs. Version prompts + models + indexes together. Release via canary, never direct to prod.

---

**Q5. How do you test prompt changes safely?**

Offline eval first: run old and new prompt on a fixed test set and diff the results. Safety regression is mandatory — always run jailbreak and injection tests. Canary to 5–10% traffic with automated rollback if metrics drop. One rule: never ship a prompt change without an eval report saying it's better.

---

## Monitoring & Observability

**Response Accuracy** — Groundedness scoring on every response, split by intent and risk tier. Add daily human QA sampling for high-risk categories.

**Latency** — Track P95/P99 per pipeline stage. Alert on regressions after every deploy. P95, not average.

**Token Usage** — Break down by endpoint, tenant, and model. Alert on sudden spikes. Set hard per-request budgets.

**Escalation Rate** — Track handoff frequency and reason (low confidence, policy block, tool failure). Rising rate = early quality warning signal.

**Customer Satisfaction (CSAT)** — Thumbs up/down with reason tags. Correlate with groundedness and latency to diagnose root cause. Never look at CSAT in isolation.
