# 🚀 Agentic AI Developer + GenAI Developer Interview Questions and Answers
## 🧭 SECTION A — Core Agentic AI (Agentic Intelligence / Autonomous Systems)
### 1. What is an AI agent?
An autonomous system that can **perceive**, **reason**, **plan**, **act**, and **self-improve** using tools, memory, and feedback loops.
### 2. What are the main components of an agentic system?
1. LLM
2. Planner
3. Tools / functions
4. Memory
5. Control loop
6. Safety guardrails
7. Environment interface
### 3. What is the “Agent Loop”?
A repeating cycle:
**Thought → Action → Observation → Reflection → Next Action**.
Used in ReAct, LangGraph, AutoGPT.
### 4. What is the ReAct framework?
*Reason + Act* pattern allowing LLMs to explain thought steps + call tools.
Reduces hallucination and improves reasoning.
### 5. What is a Tool/Function Call?
LLM outputs structured JSON that triggers external actions (APIs, search, SQL, etc.).
### 6. Why do we need external tools in agents?
LLMs cannot:
* know real-time data
* perform precise math
* access databases
* run code
Tools solve these gaps.
### 7. What is LangChain?
A framework for building LLM workflows with:
* tools
* agents
* retrieval
* chains
* memory
* prompt templates
### 8. What is LangGraph?
A graph-based agent framework for:
* deterministic execution
* stateful workflows
* parallel branching
* safe multi-agent orchestration
* retry logic & checkpoints
### 9. What is CrewAI?
A multi-agent orchestrator where agents collaborate with roles (Researcher, Developer, Reviewer).
### 10. What is LlamaIndex?
A modular framework focused on **RAG**, retrieval, data connectors, and agent capabilities.
### 11. Difference between Chatbot vs AI Agent
Chatbot: conversational → no autonomy
Agent: plans, executes tasks, uses tools, self-corrects.
### 12. What is a Planner Agent?
An agent responsible for breaking a goal into steps and guiding execution.
### 13. What is an Executor Agent?
Executes steps created by the planner using tools/APIs.
### 14. What is a Supervisor Agent?
Oversees multiple agents, ensures correctness, and resolves conflicts.
### 15. What is Memory in Agents?
Persistent or temporary storage of:
* conversations
* past actions
* vector-based knowledge
* state checkpoints
### 16. What types of memory exist?
* **Short-term:** Within current task
* **Long-term:** Vector DB (FAISS/Pinecone)
* **Episodic:** Past session states
* **Semantic:** Stored documents & embeddings
* **Procedural:** How to perform tasks
### 17. How do agents use vector databases?
To store and retrieve knowledge using embeddings.
### 18. What is reflection in agents?
Self-evaluation step to improve reasoning or correct mistakes.
### 19. What is scaffolding in agents?
Using tools + intermediate steps to help an LLM solve harder tasks.
### 20. What is the biggest challenge in agent systems?
Stability → preventing infinite loops, hallucinations, and wrong actions.
### 21. How to prevent infinite loops in agents?
* Max iterations
* Confidence thresholds
* Guardrail rules
* State logging
* Timeouts
### 22. What is a deterministic agent workflow?
An agent that always follows the same graph execution, regardless of LLM randomness.
### 23. What is a self-correcting agent?
Uses feedback loops & validation models to fix outputs.
### 24. What is an autonomous agent?
Agent that can plan & execute tasks without human intervention.
### 25. What is a multi-agent system?
Multiple specialized agents collaborating on tasks (Researcher, Coder, Tester...).
### 26. How to enable multi-agent collaboration?
Assign roles → define tasks → create communication channel → supervise.
### 27. What is agent delegation?
One agent assigns a subtask to another agent.
### 28. What is a skill in agent systems?
A reusable function/tool (e.g., web search, code execution, email sending).
### 29. What is an agent environment?
The external world the agent interacts with: web, APIs, file systems, IDEs.
### 30. What are Agent Guardrails?
Safety controls that restrict risky actions (e.g., deleting files, running harmful code).
### 31. What is state management?
Tracking progress, memory, steps, tool outputs.
### 32. What is a retry policy?
How the agent handles failed tool calls or wrong outputs.
### 33. What is agent reproducibility?
Ensuring results do not vary unexpectedly — LangGraph solves this.
### 34. What is a Code Execution Tool?
Allows agents to run Python/JS code safely in a sandbox.
### 35. What is the difference between Tool and Action?
Tool = what agent can use
Action = what agent decides to do
### 36. What is an observation in agent loop?
The output from executing an action.
### 37. What is episodic memory used for?
Learning from past tasks.
### 38. What is LLM-driven planning?
LLMs generating task plans.
### 39. What is Hierarchical Agent Planning?
Top-level planner → sub-goal → sub-tasks → agents.
### 40. What is Human-in-the-loop (HITL)?
Human approves critical decisions.
### 41. What is the architecture of a coding agent (like Devin-style)?
* Code execution
* File-system access
* Planning agent
* Review agent
* Debugger agent
* Search tools
### 42. What is Agent Simulation?
Running agents in virtual environments (e.g., Minecraft Voyager).
### 43. What is an "inner monologue" in agents?
Hidden chain-of-thought reasoning.
### 44. What is "outer monologue"?
Visible chain-of-thought that user sees.
### 45. What is a context window in agent design?
Max tokens the model can recall at once.
### 46. What is a cold start for agents?
Initial state with no memory → slower reasoning.
### 47. What is a warm start?
Pre-loaded memory or embeddings → faster and accurate execution.
### 48. What is grounding?
Making LLM answers reflect real verified facts (using RAG/tools).
### 49. What is task decomposition?
Breaking big task into small actionable units.
### 50. How do you evaluate an agent?
* Task success rate
* Tool efficiency
* Latency
* Cost
* Safety violations
* Reproducibility
## 🧠 SECTION B — LLM Fundamentals
### 51. What is a Large Language Model?
Transformer-based neural model that predicts next token.
### 52. What is a token?
Smallest unit used for language processing.
### 53. What is a transformer?
Architecture using self-attention for parallel sequence processing.
### 54. What is self-attention?
Mechanism to understand relationships between words.
### 55. What are Q, K, V?
Query, Key, Value vectors used in attention.
### 56. What is multi-head attention?
Multiple attention calculations for better representation.
### 57. What is positional encoding?
Adds sequence order info.
### 58. Why are transformers faster than RNNs?
Parallelism + constant path length.
### 59. What is an embedding?
Numerical vector representation of text semantics.
### 60. Types of embeddings?
* Word
* Sentence
* Document
* Image
* Multimodal
### 61. What is masked language modeling?
Predicting masked words (BERT).
### 62. What is next-token prediction?
Predicting next word (GPT models).
### 63. What is perplexity?
Measure of model surprise → lower = better.
### 64. What is temperature?
Controls randomness.
### 65. What is Top-K sampling?
Sampling from top K probable tokens.
### 66. What is Top-p sampling?
Sampling from tokens whose cumulative probability = p.
### 67. What is an LLM checkpoint?
Saved weights for restoration.
### 68. What is quantization?
Reducing precision to speed up inference.
### 69. What is a LoRA adapter?
Small matrices added to train only 1–2% parameters.
### 70. What is QLoRA?
Quantized LoRA → trains LLMs on 4-bit weights.
### 71. What is SFT (Supervised Fine-Tuning)?
Training model on input → output pairs.
### 72. What is instruction tuning?
Training model on tasks described with natural language.
### 73. What is RLHF?
Reinforcement learning with human preference feedback.
### 74. What is DPO?
Direct Preference Optimization — alternative to RLHF.
### 75. What causes hallucinations?
* No grounding
* Ambiguous prompts
* Missing context
* Low quality training data
* Overconfidence
### 76. How to reduce hallucinations?
* RAG
* Low temperature
* Tools
* Structured prompts
* Guardrails
### 77. What is context length?
Max tokens LLM can handle in a single input.
### 78. What is an instruction-following model?
LLM tuned to follow natural language instructions.
### 79. What is a multi-modal model?
Supports image/video/audio + text.
### 80. What are safety filters?
Rules preventing harmful output.
### 81. What is grounding with retrieval?
Injecting factual data at runtime.
### 82. What is chain-of-thought prompting?
Asking LLM to show reasoning steps.
### 83. What is tool augmentation?
Expanding LLMs with external tools.
### 84. What is long-context attention?
Sparse attention that handles massive document lengths.
### 85. What are KV caches?
Stores attention states for faster inference.
### 86. What is Mixture of Experts (MoE)?
Selectively activates experts → faster/larger models.
### 87. What is retrieval-augmented fine-tuning?
Combining RAG + fine-tuning.
### 88. What is a synthetic dataset?
LLM-generated dataset for training/fine-tuning.
### 89. What is domain adaptation?
Adapting LLMs for a specific knowledge domain.
### 90. What is catastrophic forgetting?
Model loses previously learned knowledge.
## 📚 SECTION C — RAG (Retrieval-Augmented Generation)
### 91. What is RAG?
Pipeline combining vector search + LLM generation.
### 92. Why use RAG?
* Reduces hallucinations
* Real-time knowledge updates
* Cheaper than fine-tuning
### 93. What is chunking?
Breaking documents into segments for embeddings.
### 94. What is optimal chunk size?
Depends on task, usually:
* 200–500 tokens for Q&A
* 800–1800 tokens for reasoning
### 95. What is overlap size?
Percentage of content repeated in next chunk to preserve context.
### 96. What is a vector database?
Stores embeddings for similarity search.
### 97. Popular vector DBs?
* Pinecone
* FAISS
* Chroma
* Weaviate
* Milvus
* RedisVector
### 98. What is cosine similarity?
Measures angle between vectors → semantic closeness.
### 99. What is ANN (Approximate Nearest Neighbor)?
Fast but approximate vector search.
### 100. What is metadata filtering?
Filter search results with tags (doc_type=invoice).
### 101. What is hybrid search?
Combining keyword + vector search.
### 102. What is cross-encoder reranking?
LLM-based reranker improves retrieval accuracy.
### 103. What is retrieval-depth?
Top K candidates returned (commonly K=3–10).
### 104. What is hallucination-free RAG?
RAG with verification layer and grounding checks.
### 105. What is a reranker?
Reorders retrieved docs based on relevance.
### 106. What is reference extraction?
LLM extracts supporting evidence from retrieved documents.
### 107. What is multi-vector indexing?
Store multiple embeddings per chunk (paragraphs, sentences).
### 108. What is late fusion in RAG?
Combine multiple retrieval sources at answer time.
### 109. What is early fusion?
Combine documents before embedding.
### 110. What is a knowledge graph RAG?
Retrieval based on nodes/edges (graphs) + embeddings.
### 111. What is a memory RAG?
Store dynamic user sessions + embeddings.
### 112. What is a multi-hop RAG?
RAG that retrieves info across multiple documents.
### 113. What is hallucination detection?
Agent checks if output cites provided context.
### 114. What is a RAG cache?
Caching retrieval results to reduce cost.
### 115. What impacts RAG quality most?
Chunking → embeddings → retrieval strategy → prompting.
### 116. What is a retrieval score?
Similarity-based confidence.
### 117. What is document ranking?
Sorting retrieved chunks by relevance.
### 118. What is a context window overflow?
Input exceeds LLM max tokens → truncated context.
### 119. How to deal with large documents?
Hierarchical chunking → embeddings → retrieval.
### 120. How to evaluate RAG?
* Ground truth matching
* Context precision
* Answer correctness
* Hallucination rate
## 🛠 SECTION D — Tools, Frameworks, and Ecosystems
### 121. What is LangChain?
Framework for chaining LLM tools + agents.
### 122. What is a Chain?
Sequential execution of steps.
### 123. Types of Chains?
* LLMChain
* RetrievalChain
* SequentialChain
* RouterChain
* ToolChain
### 124. What is an AgentExecutor?
Runs agent loops with tools.
### 125. What is an LCEL?
LangChain Expression Language — declarative pipeline building.
### 126. What is LangGraph used for?
Deterministic agent graphs + state machines.
### 127. What is a Node in LangGraph?
A functional step in graph.
### 128. What is a Branch?
Conditional path based on agent output.
### 129. What is a Checkpoint?
Save agent state for resumption.
### 130. What is LlamaIndex used for?
Data connectors + RAG pipelines.
### 131. What are Observers?
Hooks that track agent progress.
### 132. What is OpenAI Assistants API?
Agentic API for tools, memory, file integration.
### 133. What is Google Gemini API?
Multi-modal LLM with tool calling.
### 134. What is AWS Bedrock?
Managed LLM service (Anthropic, Amazon, Meta, Mistral).
### 135. What is Azure OpenAI?
Enterprise-grade version of OpenAI APIs.
### 136. What is HuggingFace Transformers?
Library for model loading, fine-tuning, pipelines.
### 137. What is vLLM?
Fast inference engine for LLMs.
### 138. What is Ollama?
Local LLM runner.
### 139. What is LM Studio?
Desktop LLM interface/tool.
### 140. What is Ray Serve?
Distributed model serving.
### 141. What is Databricks Mosaic AI?
Enterprise GenAI platform.
### 142. What is Guardrails AI?
Framework for validating LLM outputs.
### 143. What is Rebuff?
Tool for jailbreak detection.
### 144. What is TruLens?
LLM evaluation framework.
### 145. What is LiteLLM?
Unified LLM API gateway.
### 146. What is OpenAI Function calling?
Structured JSON outputs → tools.
### 147. What is OpenAI Realtime API?
Streaming, agent-like interactions.
### 148. What is WebNavigator?
Agent tool for browsing the internet.
### 149. What is Voice Mode?
Real-time speech and multimodal agents.
### 150. What is an Embedding Model?
Model generating dense vectors (e.g., text-embedding-3-large).
## ⚙️ SECTION E — Agentic System Design
### 151. Design a RAG-powered customer support agent.
Components:
* Query → embed → retrieve → answer using context → verify → respond.
### 152. Design a Coding Agent (Devin-style).
Needs:
* Code interpreter
* FS tooling
* Browser
* Planner agent
* Unit testing agent
* Debugger
* Git integration
### 153. How to design a multi-agent research system?
Researcher → summarizer → validator → writer → reviewer.
### 154. How to design a safe browsing agent?
* Web search tool
* Scraper
* Safety filters
* Content parser
* URL validator
### 155. How to design an agent for business automation?
* Email read/write
* Database connector
* Scheduling tool
* Planner
* Logging
### 156. How to design a workflow runner agent?
* Task queue
* Retry logic
* Async tools
* State machine
### 157. How to build a self-correcting agent?
Agent → evaluator → corrector → final.
### 158. What’s the best architecture for enterprise agents?
Modular: planning, tools, memory, verification, guardrails.
### 159. How to design a safe code-execution agent?
Sandboxing:
* timeouts
* resource limits
* no FS write
* container isolation
### 160. How to scale agents?
* Worker pools
* Batched tool calls
* Caching
* KV caches
* Sharded vector DB
### 161. What’s the difference between stateless & stateful agents?
Stateful retains memory; stateless restarts from scratch.
### 162. How to design a decision-making agent?
Use planner → evaluate options → choose best using scoring rules.
### 163. How do you evaluate agent correctness?
Outcome-based metrics + execution trace replay.
### 164. How to log agent steps?
Track prompts, actions, thoughts, tool calls.
### 165. What is a safety sandbox?
Isolated environment preventing harmful actions.
### 166. How do you design a multi-step planner?
Use chain-of-thought + scratchpad memory.
### 167. How to ensure agents don’t misuse tools?
Role-based access + permission checks.
### 168. How to build a financial analysis agent?
Tools: stock APIs, news search, SQL DB, PDF reader.
### 169. How to design a QA agent for documents?
RAG + summarizer + cross-checker.
### 170. How to design a data extraction agent?
OCR → chunk → embed → extract → validate JSON.
### 171. How to add vision capabilities to agents?
Use multimodal models like GPT-Vision or Gemini Vision.
### 172. How to build an enterprise search agent?
Index PDFs/Docs/Emails → RAG → ranking → verification.
### 173. How to build a meeting assistant?
Speech-to-text → summarizer → action-items → calendar tool.
### 174. How to design workflow guardrails?
* Format checks
* Regex validators
* Structured outputs
* Fact-checking layer
### 175. Best way to combine agents + APIs?
Use function calling API.
### 176. Why use agent graphs instead of loops?
Graph = deterministic, testable, debuggable.
### 177. How to add memory to multi-agent systems?
Shared vector DB.
### 178. How to implement reason-checker?
Use a second LLM to verify steps.
### 179. How to build a planning agent?
Few-shot examples + task decomposition rules.
### 180. How to reduce cost in agent systems?
* Caching
* Smaller models
* LoRA adapters
* KV reuse
* Reranking before LLM
## 🧪 SECTION F — Coding, Debugging & Practical Questions
### 181. Write Python code to compute cosine similarity.
```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([v1], [v2])
```
### 182. Write a FAISS search example.
```python
import faiss
import numpy as np
index = faiss.IndexFlatL2(768)
index.add(embeddings)
D, I = index.search(query, k=5)
```
### 183. How to implement a simple RAG in Python?
* embed docs
* store in vector DB
* retrieve
* pass to LLM
### 184. How to call OpenAI function calling?
```json
{
 "name": "get_weather",
 "arguments": {"city": "Delhi"}
}
```
### 185. Write code to build a LangChain agent.
```python
from langchain.agents import initialize_agent
agent = initialize_agent(tools, llm, agent="zero-shot-react")
```
### 186. How to retrieve documents in LlamaIndex?
```python
query_engine = index.as_query_engine()
result = query_engine.query("What is X?")
```
### 187. How to run vLLM inference?
```python
from vllm import LLM
llm = LLM("model")
llm.generate("Hello")
```
### 188. Write a prompt template example.
```python
template = "Summarize: {text}"
```
### 189. How to create a PDF agent?
Use PyMuPDF to extract text → RAG → LLM.
### 190. Write retry logic for agents.
```python
for _ in range(3):
    try: do_action()
    except: continue
```
### 191. Validate JSON output from LLM.
Use `jsonschema`.
### 192. Write Python code to chunk text.
```python
text[i:i+500]
```
### 193. How to detect hallucinations?
Check if answer references retrieved context.
### 194. What is a backoff strategy?
Retry with increasing wait time.
### 195. Write Python code to run embeddings.
```python
client.embeddings.create(model="text-embedding-3-small", input=text)
```
### 196. Detect if agent is stuck in loops.
Track repeating states.
### 197. What is log-probability?
Numerical confidence of model tokens.
### 198. What is a scratchpad?
Memory of intermediate reasoning steps.
### 199. Implement a JSON guardrail.
Regex or JSON schema validation.
### 200. Deploy an agent on FastAPI?
FastAPI endpoint → agent.run → return output.
