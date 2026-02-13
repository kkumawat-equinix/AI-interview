# Frequent GenAI Interview Q&A (Simple + Interview-Friendly)

Last updated: 14 Feb 2026

Goal: answers you can say **clearly** in interviews.

How to use this sheet:
- First say the **Say it like this (45–60s)** part.
- If they ask deeper, add the **Example** and **Details (optional)**.

## What interviewers usually want

- You understand the basics (Transformer, RAG, embeddings).
- You can design a real system (reliable, secure, monitored).
- You can explain trade-offs (cost vs quality, speed vs accuracy).
- You think about risk (hallucinations, PII, bias).

## Fundamentals (simple refresh)

### Discriminative vs Generative (simple)

**Say it like this (45s):**
Discriminative models classify or score input data (e.g., spam vs not spam). Generative models create new data, like text or images. In practice, both are used: discriminative for selecting or ranking, generative for producing content.

**Example:**
Spam classifier = discriminative. ChatGPT writing an email = generative.

**Details (optional):**
- Discriminative: classification, regression.
- Generative: creates text, images, audio; follows instructions.

### GANs vs VAEs vs Diffusion (no math)

**Say it like this (45s):**
GANs, VAEs, and Diffusion are all generative models:
- GANs: A generator tries to fool a discriminator. Outputs are sharp, but training is tricky.
- VAEs: Learn a compressed space and generate from it. Training is stable, but outputs may be blurry.
- Diffusion: Start with noise and gradually refine it into data. Slower, but high quality and controllable.

**Example:**
Modern image generators use diffusion for reliability and quality.

### What is a Transformer? What is attention?

**Say it like this (60s):**
A Transformer is the main architecture for LLMs. Its key idea is attention: the model looks at all tokens and decides which are most important for each prediction. This allows it to understand context over long texts. Transformers are faster than RNNs because they process tokens in parallel, enabling large-scale training.

**Example:**
In “The trophy doesn’t fit in the suitcase because it is too big,” attention helps the model link “it” to “trophy.”

## LLMs in practice (knobs + patterns)

### Temperature, top-k, top-p (sampling)

**Say it like this (45s):**
These settings control how creative the model is:
- Temperature: lower means more predictable, higher means more creative.
- Top-k/top-p: restrict the next token choices to avoid unlikely words.

**Example:**
For support answers, use low temperature. For creative writing, use higher values.

### Zero-shot vs Few-shot

**Say it like this (45s):**
Zero-shot means giving only instructions. Few-shot means giving instructions plus a few examples. Few-shot helps when you need a specific format or clarity.

**Example:**
If you want JSON output, show 1–2 examples in your prompt.

### RAG vs Fine-tuning (when to use which)

**Say it like this (60s):**
Use RAG when the model needs up-to-date or private data: retrieve relevant documents and give them to the model. Use fine-tuning when you need the model to change its behavior or style. Most companies start with RAG because it’s safer and easier to update.

**Example:**
Weekly policy updates → RAG. Consistent brand tone → fine-tune.

### How to evaluate LLM outputs (simple)

**Say it like this (60s):**
I check: (1) correctness (is it accurate?), (2) usefulness (does it solve the problem?), and (3) production metrics (speed, cost, errors). I use test cases, human review, and monitor real usage.

**Example:**
For a support bot: track resolution rate, escalations, and hallucinations.

## 20 practical interview questions (simple answers you can speak)

### 1) How would you design a GenAI customer support chatbot?

**Say it like this (60–90s):**
I’d use a RAG approach: (1) Gather and clean the knowledge base (FAQs, policies, tickets). (2) Chunk and embed the data, store in a vector database with metadata. (3) For each question, retrieve relevant chunks and give them to the LLM with clear instructions. (4) Add safety: if retrieval is weak, ask clarifying questions or escalate. (5) Measure results and improve.

**Example:**
For “Refund policy for EU customers,” filter docs by region=EU, then answer with the exact policy.

**Details (optional):**
- Guardrails: require citations, refuse if context is missing, avoid guessing.
- Metrics: deflection rate, CSAT, escalation, hallucination rate.

### 2) What is prompt engineering, and why does it matter?

**Say it like this (60s):**
Prompt engineering is writing clear instructions and context so the model gives consistent, useful answers. It matters because prompt wording can change results a lot. Good prompts reduce errors and cost.

**Example:**
Instead of “summarize this,” say “Summarize in 5 bullets, include dates/numbers, and say ‘Not provided’ if info is missing.”

**Details (optional):**
- Specify role, task, constraints, output format, and give examples.
- Ask for clarifying questions if input is unclear.

### 3) How do you handle data privacy/PII in GenAI apps?

**Say it like this (60s):**
Treat prompts and logs as sensitive. Minimize data sent to the model, detect and redact PII, and avoid storing raw user text. Secure secrets, encrypt data, and use strict access controls. Use private deployments if needed for compliance.

**Example:**
Replace “john.doe@email.com” with “[EMAIL]” before sending to the model, and keep the mapping internal.

**Details (optional):**
- Redact before inference and logging.
- Use tenant isolation, audit logs, and retention limits.

### 4) Explain RAG and when you would use it.

**Say it like this (60s):**
RAG combines search and LLMs. Store documents as embeddings, retrieve relevant pieces for each question, and give them to the LLM. Use RAG when data is private, large, or changes often, and you want answers based on real sources.

**Example:**
A finance assistant that must answer from internal policy PDFs uses RAG with citations.

**Details (optional):**
- Good chunking and metadata filters are key.
- Add re-ranking if retrieval is inconsistent.

### 5) How do you fine-tune an LLM for a domain task?

**Say it like this (60–90s):**
Fine-tuning means training the model on specific examples so it learns a pattern. Define the target behavior, collect high-quality data, fine-tune using efficient methods (like LoRA/PEFT), and evaluate on edge cases. Only fine-tune if prompting and RAG aren’t enough.

**Example:**
Customer support ticket categorization and reply templates that must follow a strict schema.

**Details (optional):**
- Keep training data clean and compliant; avoid secrets.
- Maintain regression tests to avoid breaking old behaviors.

### 6) How do you reduce hallucinations?

**Say it like this (60s):**
Reduce hallucinations by grounding answers with RAG, giving clear instructions (“answer only from provided context”), keeping temperature low, and adding validation. If context is missing, the system should say it doesn’t know or escalate.

**Example:**
If nothing relevant is retrieved, the bot says: “I don’t have that info. Which product/version are you using?”

**Details (optional):**
- Require citations or quotes.
- Add schema validation for structured outputs.

### 7) What are embeddings and how are they used?

**Say it like this (60s):**
Embeddings turn text into numbers that capture meaning. Similar texts have similar embeddings. This enables semantic search, so you can find relevant docs even if the words are different.

**Example:**
User asks “How do I reset my password?” and the doc says “Change account credentials”—embeddings match them.

**Details (optional):**
- Used in RAG, deduplication, clustering, recommendations.
- Pick an embedding model suited to your language/domain.

### 8) How do you manage context in chat applications?

**Say it like this (60s):**
Because models have context limits, keep only recent chat turns, summarize older ones, and store key facts (like user preferences) separately. This keeps answers consistent and reduces cost.

**Example:**
Store “user prefers concise answers” as a fact, not by repeating all chat history.

**Details (optional):**
- Use a facts memory, summaries, and a recent window.
- Treat user history as untrusted input (risk of prompt injection).

### 9) What do LangChain/LlamaIndex provide (at a high level)?

**Say it like this (45–60s):**
They are frameworks for building LLM apps quickly. They offer connectors (for PDFs/web), chunking and embedding pipelines, retrieval, prompt templates, tool calling, and agent workflows. Use them to speed up development, but keep core logic separate.

**Example:**
Load PDFs, chunk, embed, store, and query—all in a few lines of code.

### 10) How do you optimize token usage and cost?

**Say it like this (60s):**
Cost comes from tokens and latency. Reduce tokens by making prompts concise, retrieving only the best context, summarizing long content, and limiting output. Cache repeated work and use smaller models for simple tasks.

**Example:**
Use a small model for intent classification; call a larger model only when needed.

### 11) How do you evaluate LLM quality in production?

**Say it like this (60–90s):**
Start with a test set based on real user questions and expected answers. Evaluate quality (correct, grounded, helpful) and engineering metrics (latency, cost, errors). In production, monitor outcomes, collect feedback, and re-run evaluations after changes.

**Example:**
When the knowledge base or model changes, run regression tests and compare scores.

### 12) What is a vector database and why use one?

**Say it like this (45–60s):**
A vector database stores embeddings and lets you quickly find the most similar text. It’s used in RAG to retrieve relevant chunks fast, even at large scale.

**Example:**
Search across 1 million internal paragraphs in milliseconds.

**Details (optional):**
- Metadata filtering prevents cross-tenant leakage.
- Re-ranking can improve accuracy.

### 13) Prompting vs fine-tuning—how do you choose?

**Say it like this (60s):**
First, improve prompts. If knowledge is missing, add RAG. Fine-tune only if behavior still isn’t consistent. Fine-tuning is powerful but more costly and needs maintenance.

**Example:**
Need latest docs → RAG. Need strict output format → fine-tune.

### 14) What is model drift in GenAI systems?

**Say it like this (60s):**
Model drift means the system’s quality changes over time, often due to changes in data, user questions, or model updates. Detect drift by monitoring real traffic, versioning prompts and indexes, and running regular evaluations.

**Example:**
After a policy update, old answers may be wrong unless the RAG index is refreshed.

### 15) Zero-shot vs few-shot: when does few-shot help most?

**Say it like this (45–60s):**
Few-shot helps most when the model needs to follow a specific pattern, strict format, or tricky rules. One good example is often better than a long explanation.

**Example:**
Show an example of “good vs bad” classification for moderation.

### 16) How do you choose chunk size and overlap in RAG?

**Say it like this (60s):**
Chunking means splitting documents for retrieval. Choose chunk sizes that keep meaning together (often 200–800 tokens), add some overlap to avoid cutting sentences, and validate by testing retrieval quality.

**Example:**
For policy docs, chunk by headings so each chunk is a full rule.

**Details (optional):**
- Too small: misses context. Too large: wastes tokens, adds irrelevant info.

### 17) How do you secure API keys and secrets?

**Say it like this (45–60s):**
Never hardcode secrets. Use a secret manager, least privilege, rotate keys, and monitor usage. Separate keys by environment and restrict access.

**Example:**
Store keys in AWS Secrets Manager, inject at runtime, rotate monthly.

### 18) What’s the role of caching in GenAI pipelines?

**Say it like this (45–60s):**
Caching saves time and money. Cache embeddings, retrieval results, and sometimes answers for repeated questions. Make sure caches expire when documents change.

**Example:**
Cache retrieval results for “reset password” since it’s a common question.

### 19) How do you control tone and writing style reliably?

**Say it like this (60s):**
Give clear instructions for tone, length, audience, and format. Add examples and a checklist (e.g., “no jargon, 6th-grade reading level”). Validate outputs. Fine-tune only if you need perfect consistency.

**Example:**
“Write in a polite, professional tone. Max 6 bullets. No acronyms unless explained.”

### 20) How do you integrate GenAI into existing workflows?

**Say it like this (60–90s):**
Integrate GenAI like any service: put it behind an API, connect to systems (CRM, ticketing, docs), and add controls to prevent unsafe actions. Add approvals for high-impact steps, logging, and dashboards for cost and quality. Measure ROI with clear metrics.

**Example:**
Draft a ticket reply automatically, but require agent approval before sending.

**Details (optional):**
- Observability: trace requests, version prompts/models, avoid logging PII.
- Metrics: time saved, resolution rate, error reduction.

## Quick daily drill (5 minutes)

- Pick 3 questions from the 20 and answer each in 45 seconds.
- For each answer, add 1 production concern (latency/cost/privacy) and 1 mitigation.
