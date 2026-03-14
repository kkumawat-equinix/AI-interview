🔥 ML Engineer Interview Questions (GenAI + LLM Focused)
1️⃣ LLM Fine-Tuning (MOST IMPORTANT)

What is the difference between pre-training, fine-tuning, and instruction tuning?

When would you avoid fine-tuning and use prompt engineering or RAG instead?

Explain PEFT. Why is it preferred over full fine-tuning?

How does LoRA work internally? Which layers do you apply it to and why?

What is QLoRA and how does it reduce GPU memory usage?

What are the risks of fine-tuning on synthetic (LLM-generated) data?

How do you prevent catastrophic forgetting during fine-tuning?

What hyperparameters matter most during LLM fine-tuning?

How do you evaluate whether fine-tuning actually helped?

How would you fine-tune an LLM for tool calling or agent behavior?

👉 If they ask follow-ups here, you’re doing well.

2️⃣ Training Mechanics & Optimization

Why does training require more GPU memory than inference?

What is gradient accumulation and when do you use it?

Explain mixed precision training (fp16 vs bf16).

What is gradient checkpointing and its trade-offs?

How do batch size and sequence length affect memory?

What optimizer do you usually use for LLM fine-tuning and why?

How do learning rate schedulers help in fine-tuning?

What causes training instability in LLMs?

How do you detect overfitting in generative models?

What is loss masking and why is it important for instruction tuning?

3️⃣ GPU / CPU / Memory (VERY COMMON)

Break down GPU memory usage during training.

Why can’t CPU RAM compensate for insufficient GPU VRAM?

How does quantization affect inference vs training?

What is KV-cache and how does it improve inference speed?

How do you decide model size based on available GPUs?

What are common causes of GPU under-utilization?

How do you train a 13B model on a single 24GB GPU?

What happens if sequence length doubles?

How does multi-GPU training work (data vs model parallelism)?

What monitoring metrics do you track for GPU health in production?

4️⃣ Prompt Engineering vs Fine-Tuning

What are Chain-of-Thought and Few-Shot prompting?

Why does Chain-of-Thought sometimes degrade performance?

How do prompts interact with fine-tuned models?

Can bad prompts hurt a fine-tuned model?

How do you design prompts for tool usage?

How do you reduce hallucinations using prompts?

When do prompts stop scaling and fine-tuning becomes necessary?

5️⃣ Agentic AI & Multi-Agent Systems (JD-Specific)

What is an agentic AI system?

What components does an agent typically have?

How does memory work in agentic systems?

What’s the difference between short-term and long-term memory?

How do agents decide which tool to call?

How do you evaluate an agent’s performance?

What failure modes do agentic systems have?

How do you prevent infinite loops in agents?

How do multi-agent systems coordinate tasks?

What’s the difference between RAG and agentic workflows?

6️⃣ Deployment, Scaling & Production

What challenges arise when deploying LLMs in production?

How do you reduce inference latency?

What is batching and why does it improve throughput?

How do you handle model versioning?

How do you roll out a new fine-tuned model safely?

What metrics do you monitor in production?

How do you detect data drift for LLMs?

What is A/B testing for models?

How do you handle hallucinations in production?

What’s your strategy for continuous retraining?

7️⃣ NLP Fundamentals (They WILL Test Basics)

What is tokenization and why does it matter?

What is attention and why is it better than RNNs?

What is positional encoding?

Explain self-attention step by step.

What causes long-context degradation?

Difference between encoder-only and decoder-only models?

How does BERT differ from GPT?

What are common NLP evaluation metrics and their limitations?

8️⃣ Scenario-Based Questions (MOST IMPORTANT)

Model accuracy dropped after deployment. What do you check first?

Fine-tuning made performance worse. Why?

Users report hallucinations. How do you debug?

GPU cost is too high. What optimizations do you apply?

New domain data arrives weekly. How do you adapt the model?

Latency SLA is 200ms. How do you design the system?

Agent is making wrong tool calls. How do you fix it?

Stakeholders want explainability. What do you provide?

How do you justify fine-tuning cost to business?

What would you improve if given 6 months?