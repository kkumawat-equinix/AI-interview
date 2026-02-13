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

**Say it like this (45s)**: Discriminative models are like “judge models” — they take an input and pick a label/score (spam vs not spam). Generative models are like “writer models” — they learn to produce new content (text, images). In real LLM apps we often use both: a retriever/ranker to find the right info, and a generator to write the final answer.

**Example**: Spam classifier = discriminative. ChatGPT writing an email = generative.

**Details (optional)**
- Discriminative: classification/regression.
- Generative: create text/images/audio; can follow instructions.

### GANs vs VAEs vs Diffusion (no math)

**Say it like this (45s)**: All three are ways to generate new data.
- GANs are like a forger vs detective game: generator tries to fool a discriminator. Often sharp outputs, but training can be unstable.
- VAEs learn a “compressed space” and generate from it. Training is stable, but outputs can be softer.
- Diffusion starts from noise and slowly cleans it into an image/text representation. It’s slower, but quality and control are strong.

**Example**: Modern image generators often use diffusion because it’s reliable and high quality.

### What is a Transformer? What is attention?

**Say it like this (60s)**: A Transformer is the architecture behind most LLMs. The key idea is **attention**: for each word/token, the model looks at other tokens and decides what matters most. This lets it connect information across a long text. Compared to RNNs, Transformers train much faster because they can process tokens in parallel on GPUs, which made scaling possible.

**Example**: In “The trophy doesn’t fit in the suitcase because it is too big”, attention helps the model link “it” → “trophy”.

## LLMs in practice (knobs + patterns)

### Temperature, top-k, top-p (sampling)

**Say it like this (45s)**: These are controls for how “creative” the model is.
- Temperature: lower = more consistent and conservative; higher = more varied.
- Top-k / top-p: limit the choices for the next token so the model doesn’t pick weird low-probability words.

**Example**: For support answers or policy text, keep temperature low (more reliable). For marketing copy, raise it a bit.

### Zero-shot vs Few-shot

**Say it like this (45s)**: Zero-shot is “just instructions”. Few-shot is “instructions + a couple of examples”. Few-shot helps a lot when you need a strict format or the task is easy to misunderstand.

**Example**: If you want JSON output with specific fields, show 1–2 examples.

### RAG vs Fine-tuning (when to use which)

**Say it like this (60s)**: If the problem is “the model doesn’t know my company’s data”, use **RAG**: retrieve relevant documents and give them to the model at answer time. If the problem is “the model’s behavior is wrong” (tone, style, classification, consistent tool usage), consider **fine-tuning**. Most enterprise apps start with RAG because it’s safer and updates quickly.

**Example**: New HR policy updates weekly → RAG. Brand tone for marketing emails → fine-tune (maybe).

### How to evaluate LLM outputs (simple)

**Say it like this (60s)**: I evaluate on 3 buckets: (1) correctness/grounding (is it true and supported?), (2) usefulness (does it solve the task?), and (3) production metrics (latency, cost, failure rate). I use offline test cases + human review, and then online monitoring with feedback.

**Example**: For a support bot: measure resolution rate, escalation rate, and hallucination incidents.

## 20 practical interview questions (simple answers you can speak)

### 1) How would you design a GenAI customer support chatbot?

**Say it like this (60–90s)**: I’d build it RAG-first. Step 1: collect and clean the knowledge base (FAQs, policies, past tickets). Step 2: chunk it, create embeddings, and store in a vector DB with metadata like product and region. Step 3: on each user question, retrieve the most relevant chunks and pass them to the LLM with instructions to answer only from that context. Step 4: add safety: if retrieval is weak, ask a clarifying question or escalate to a human. Step 5: measure outcomes and iterate.

**Example**: “Refund policy for EU customers” → filter docs by region=EU before retrieval, then answer with the exact policy text.

**Details (optional)**
- Guardrails: citations, refusal when context missing, “don’t guess”.
- Product metrics: deflection rate, CSAT, escalation rate, hallucination rate.

### 2) What is prompt engineering, and why does it matter?

**Say it like this (60s)**: Prompt engineering means writing instructions and context so the model behaves consistently. It matters because the same model can perform very differently depending on how you ask. Good prompts reduce hallucinations, improve formatting, and lower cost because you avoid retries.

**Example**: Instead of “summarize this”, use “Summarize in 5 bullets, include dates/numbers, and if info is missing say ‘Not provided’.”

**Details (optional)**
- Clear role + task + constraints + output format + 1–2 examples.
- Add “ask clarifying questions” for ambiguous inputs.

### 3) How do you handle data privacy/PII in GenAI apps?

**Say it like this (60s)**: I treat prompts and logs like sensitive data. I minimize what I send to the model, detect and redact PII, and avoid storing raw user text unless necessary. I also secure secrets, encrypt data, and use strict access control and retention. If compliance requires it, I choose a private deployment instead of a public API.

**Example**: Replace “john.doe@email.com” with “[EMAIL]” before sending to the model, and keep the mapping only inside your system.

**Details (optional)**
- Redaction before inference and before logging.
- Tenant isolation + audit logs + retention limits.

### 4) Explain RAG and when you would use it.

**Say it like this (60s)**: RAG is “search + LLM”. First you store your documents as embeddings. When a user asks a question, you retrieve the most relevant pieces and give them to the LLM, so it answers based on your actual data. Use RAG when your data is private, large, or changes often, and when you want answers grounded in sources.

**Example**: A finance assistant that must answer from internal policy PDFs → RAG with citations.

**Details (optional)**
- Good chunking and metadata filters often matter more than model choice.
- Add re-ranking if retrieval quality is inconsistent.

### 5) How do you fine-tune an LLM for a domain task?

**Say it like this (60–90s)**: Fine-tuning is when you teach the model a repeated pattern so it behaves consistently. I start by defining the exact target behavior and collecting high-quality examples. Then I fine-tune using efficient methods like LoRA/PEFT, evaluate on a holdout set and tricky edge cases, and finally deploy with versioning and monitoring. I only fine-tune when prompting and RAG can’t reach stable quality.

**Example**: Customer support “ticket categorization + reply template” where the output format must always match a strict schema.

**Details (optional)**
- Keep training data clean and compliant; don’t include secrets.
- Maintain regression tests so you don’t break prior behaviors.

### 6) How do you reduce hallucinations?

**Say it like this (60s)**: I reduce hallucinations by not asking the model to “guess”. I ground answers with RAG, set clear instructions like “answer only from provided context”, keep temperature low for factual tasks, and add validation checks. If the system can’t find strong context, it should say it doesn’t know and either ask a question or escalate.

**Example**: If retrieval returns nothing relevant, the bot responds: “I don’t have that info in our docs. Which product/version are you using?”

**Details (optional)**
- Require citations/quotes.
- Add schema validation for structured outputs.

### 7) What are embeddings and how are they used?

**Say it like this (60s)**: Embeddings turn text into numbers that capture meaning. If two texts mean similar things, their embeddings are close together. That lets you do semantic search: find relevant docs even when the words don’t match exactly.

**Example**: User asks “How do I reset my password?” and your doc says “Change account credentials” — embeddings still match them.

**Details (optional)**
- Used in RAG retrieval, deduplication, clustering, recommendations.
- Choose an embedding model that matches your language/domain.

### 8) How do you manage context in chat applications?

**Say it like this (60s)**: Because models have context limits, I don’t keep the entire chat forever. I keep the most recent turns, summarize older turns, and store important facts (like user preferences) separately so I can retrieve them when needed. This keeps answers consistent and reduces cost.

**Example**: Store “user prefers concise answers” as a fact, rather than repeating 50 turns of chat.

**Details (optional)**
- Use a “facts memory” store + summary + recent window.
- Treat user history as untrusted input (prompt injection risk).

### 9) What do LangChain/LlamaIndex provide (at a high level)?

**Say it like this (45–60s)**: They’re frameworks that help you build LLM apps faster. They provide connectors (read PDFs/web pages), chunking + embedding pipelines, retrieval, prompt templates, tool calling, and agent workflows. I use them to move faster, but I keep my core logic and tests separate.

**Example**: Load a folder of PDFs → chunk → embed → store → query in a few lines.

### 10) How do you optimize token usage and cost?

**Say it like this (60s)**: Cost is mainly tokens and latency. I reduce tokens by keeping prompts tight, retrieving only the best context, summarizing long content, and setting output limits. I also cache repeated work (embeddings and retrieval results) and route simple tasks to smaller/cheaper models.

**Example**: Use a small model to classify intent; call a larger model only when needed.

### 11) How do you evaluate LLM quality in production?

**Say it like this (60–90s)**: I start with a realistic test set based on real user questions and expected answers. Then I evaluate across quality (correct, grounded, helpful) and engineering metrics (latency, cost, errors). In production, I monitor outcomes, collect user feedback, and regularly re-run the eval suite when prompts, models, or the index changes.

**Example**: Every time the KB updates or the model version changes, run a regression eval and compare scores.

### 12) What is a vector database and why use one?

**Say it like this (45–60s)**: A vector DB stores embeddings and lets you quickly find “most similar” text. It’s used in RAG to retrieve relevant chunks fast, even at large scale.

**Example**: Search across 1M internal paragraphs in milliseconds.

**Details (optional)**
- Metadata filtering prevents cross-tenant leakage.
- Re-ranking can improve accuracy.

### 13) Prompting vs fine-tuning—how do you choose?

**Say it like this (60s)**: I use a simple decision order: (1) prompt improvements, (2) add RAG if knowledge is missing, (3) fine-tune only if the behavior still isn’t consistent enough. Fine-tuning is powerful but costs more and needs maintenance.

**Example**: Need latest product docs → RAG. Need the model to always return a strict structured template → fine-tune.

### 14) What is model drift in GenAI systems?

**Say it like this (60s)**: Drift means the system’s quality changes over time. It can happen because your documents change, user questions change, or you upgraded the model/provider. I detect drift by monitoring real traffic, keeping versioned prompts/indexes, and running the same evaluation set regularly.

**Example**: After a policy update, old answers become wrong unless the RAG index is refreshed.

### 15) Zero-shot vs few-shot: when does few-shot help most?

**Say it like this (45–60s)**: Few-shot helps most when the model needs to follow a specific pattern: strict output format, tricky label rules, or a consistent tone. One good example often beats a long explanation.

**Example**: Show one example of “good vs bad” classification for moderation.

### 16) How do you choose chunk size and overlap in RAG?

**Say it like this (60s)**: Chunking is splitting documents into pieces for retrieval. I choose chunk sizes that keep meaning together (often 200–800 tokens depending on docs), add a little overlap so sentences don’t get cut, and then validate by checking retrieval quality on real questions.

**Example**: For policy docs, chunk by headings/sections so each chunk is a complete rule.

**Details (optional)**
- Too small: retrieval misses full context.
- Too large: wastes tokens and pulls irrelevant content.

### 17) How do you secure API keys and secrets?

**Say it like this (45–60s)**: Never hardcode secrets. Use a secret manager, least privilege, rotation, and monitoring. Keep separate keys for environments and lock down who/what can use them.

**Example**: Store keys in AWS Secrets Manager (or similar), inject at runtime, rotate monthly.

### 18) What’s the role of caching in GenAI pipelines?

**Say it like this (45–60s)**: Caching saves money and time. You can cache embeddings, retrieval results, and sometimes final answers for repeated questions. The key is correct invalidation—if docs change, caches must expire.

**Example**: Cache retrieval results for “reset password” because it’s asked thousands of times.

### 19) How do you control tone and writing style reliably?

**Say it like this (60s)**: Start with clear instructions: tone, length, audience, and format. Add 1–2 examples and a checklist (e.g., “no jargon, 6th-grade reading level”). Then validate outputs. Fine-tune only if you need near-perfect consistency at scale.

**Example**: “Write in a polite, professional tone. Max 6 bullets. No acronyms unless explained.”

### 20) How do you integrate GenAI into existing workflows?

**Say it like this (60–90s)**: I integrate GenAI like any other service: put it behind an API, connect it to systems like CRM/ticketing/docs, and add controls so it can’t take unsafe actions. I add approvals for high-impact steps, logging/auditing, and dashboards for cost and quality. Finally, I measure ROI with clear metrics.

**Example**: Draft a ticket reply automatically, but require an agent to approve before sending.

**Details (optional)**
- Observability: trace requests end-to-end, version prompts/models, avoid logging PII.
- Business metrics: time saved, resolution rate, error reduction.

## Quick daily drill (5 minutes)

- Pick 3 questions from the 20 and answer each in 45 seconds.
- For each answer, add 1 production concern (latency/cost/privacy) and 1 mitigation.
