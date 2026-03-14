🔥 ML Engineer Interview Questions (GenAI + LLM Focused)
1️⃣ LLM Fine-Tuning (MOST IMPORTANT)

Q1. What is the difference between pre-training, fine-tuning, and instruction tuning?





Answer: Pre-training teaches a model general language patterns using large datasets. Fine-tuning adapts the model to a specific task or domain using smaller, targeted data. Instruction tuning further refines the model to follow explicit instructions or prompts, improving usability for task-based queries.

Q2. When would you avoid fine-tuning and use prompt engineering or RAG instead?



Answer: Avoid fine-tuning when you need quick adaptation, have limited data, or want to reduce costs. Use prompt engineering for flexible task changes, and Retrieval-Augmented Generation (RAG) when you need real-time access to external knowledge without retraining the model.

Q3. Explain PEFT. Why is it preferred over full fine-tuning?



Answer: Parameter-Efficient Fine-Tuning (PEFT) updates only a small subset of model parameters, reducing compute and memory requirements. It is preferred because it’s faster, cheaper, and minimizes risks of overfitting or catastrophic forgetting.

Q4. How does LoRA work internally? Which layers do you apply it to and why?



Answer: LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into specific layers (usually attention and feed-forward layers) of the model. This allows efficient adaptation with minimal parameter updates, saving memory and computation.

Q5. What is QLoRA and how does it reduce GPU memory usage?



Answer: QLoRA quantizes model weights to lower precision (e.g., 4-bit), drastically reducing memory usage during training. It enables fine-tuning large models on limited hardware without significant loss in accuracy.

Q6. What are the risks of fine-tuning on synthetic (LLM-generated) data?



Answer: Risks include reinforcing model biases, amplifying hallucinations, and reducing generalization. Synthetic data may lack real-world diversity, leading to overfitting or unreliable outputs.

Q7. How do you prevent catastrophic forgetting during fine-tuning?



Answer: Use regularization, mix original and new data, and apply techniques like replay or gradual unfreezing. Monitoring performance on original tasks helps detect and mitigate forgetting.

Q8. What hyperparameters matter most during LLM fine-tuning?



Answer: Learning rate, batch size, number of epochs, and weight decay are critical. Proper tuning prevents overfitting and ensures stable convergence.

Q9. How do you evaluate whether fine-tuning actually helped?



Answer: Compare task-specific metrics (accuracy, F1, etc.) before and after fine-tuning. Use validation sets and real-world scenarios to assess improvements and check for unintended side effects.

Q10. How would you fine-tune an LLM for tool calling or agent behavior?



Answer: Collect examples of desired tool calls or agent actions, fine-tune the model on these, and validate using task-specific benchmarks. Reinforcement learning or supervised fine-tuning can be used to optimize agent performance.

👉 If they ask follow-ups here, you’re doing well.

2️⃣ Training Mechanics & Optimization

Q11. Why does training require more GPU memory than inference?



Answer: Training requires storing activations, gradients, and optimizer states for backpropagation, which significantly increases memory usage compared to inference, where only forward activations are needed.

Q12. What is gradient accumulation and when do you use it?



Answer: Gradient accumulation sums gradients over multiple mini-batches before updating weights. It is used when GPU memory is limited, allowing effective larger batch sizes without increasing memory requirements.

Q13. Explain mixed precision training (fp16 vs bf16).
Answer: Mixed precision training uses lower-precision formats (fp16 or bf16) for computations, reducing memory usage and speeding up training. fp16 is widely supported, while bf16 offers better numerical stability on newer hardware.

Q14. What is gradient checkpointing and its trade-offs?



Answer: Gradient checkpointing saves memory by storing fewer activations during forward pass and recomputing them during backward pass. The trade-off is increased computation time for reduced memory usage.

Q15. How do batch size and sequence length affect memory?



Answer: Larger batch sizes and longer sequence lengths increase memory usage because more data and activations must be stored during training. Adjusting these parameters helps balance memory and performance.

Q16. What optimizer do you usually use for LLM fine-tuning and why?



Answer: Adam or AdamW are commonly used for LLM fine-tuning due to their adaptive learning rates and good convergence properties. AdamW also decouples weight decay for better regularization.

Q17. How do learning rate schedulers help in fine-tuning?



Answer: Learning rate schedulers adjust the learning rate during training, preventing overshooting and improving convergence. Common schedulers include cosine annealing and step decay.

Q18. What causes training instability in LLMs?



Answer: Instability can be caused by high learning rates, poor initialization, exploding/vanishing gradients, or inappropriate batch sizes. Careful hyperparameter tuning and normalization help mitigate these issues.

Q19. How do you detect overfitting in generative models?



Answer: Overfitting is detected by monitoring validation loss and performance metrics. If the model performs well on training data but poorly on unseen data, it is likely overfitting.

Q20. What is loss masking and why is it important for instruction tuning?



Answer: Loss masking ignores irrelevant tokens during loss calculation, focusing training on meaningful outputs. It is crucial for instruction tuning to ensure the model learns to follow instructions accurately.

3️⃣ GPU / CPU / Memory (VERY COMMON)

Q21. Break down GPU memory usage during training.
Answer: GPU memory is used for model weights, activations, gradients, optimizer states, and temporary buffers. Activations and gradients consume the most memory during training.

Q22. Why can’t CPU RAM compensate for insufficient GPU VRAM?



Answer: GPU VRAM is required for fast parallel computations. CPU RAM is slower and cannot directly accelerate deep learning operations, so it cannot substitute for VRAM during training.

Q23. How does quantization affect inference vs training?



Answer: Quantization reduces model size and speeds up inference by using lower precision weights. During training, quantization is less common because it can impact gradient calculations and model accuracy.

Q24. What is KV-cache and how does it improve inference speed?



Answer: KV-cache stores key and value tensors from previous transformer layers, allowing reuse in autoregressive generation. This reduces redundant computation and speeds up inference.

Q25. How do you decide model size based on available GPUs?



Answer: Model size is chosen based on GPU VRAM, batch size, and sequence length. Profiling and memory estimation tools help ensure the model fits and runs efficiently.

Q26. What are common causes of GPU under-utilization?



Answer: Causes include data loading bottlenecks, small batch sizes, inefficient code, and poor parallelism. Optimizing input pipelines and increasing batch size can improve utilization.

Q27. How do you train a 13B model on a single 24GB GPU?



Answer: Use techniques like gradient checkpointing, mixed precision, and low-rank adaptation (LoRA/QLoRA) to reduce memory usage. Smaller batch sizes and sequence lengths also help.

Q28. What happens if sequence length doubles?



Answer: Memory and computation requirements increase quadratically, as transformer models scale with sequence length. This can lead to out-of-memory errors or slower training.

Q29. How does multi-GPU training work (data vs model parallelism)?



Answer: Data parallelism splits batches across GPUs, each processing a copy of the model. Model parallelism splits the model itself across GPUs. Both approaches enable training larger models or faster throughput.

Q30. What monitoring metrics do you track for GPU health in production?



Answer: Track GPU utilization, memory usage, temperature, power consumption, and error rates. Monitoring ensures stable operation and helps detect hardware issues early.

4️⃣ Prompt Engineering vs Fine-Tuning

Q31. What are Chain-of-Thought and Few-Shot prompting?



Answer: Chain-of-Thought prompting guides the model to reason step-by-step, improving complex problem solving. Few-Shot prompting provides a few examples in the prompt to help the model generalize to new tasks.

Q32. Why does Chain-of-Thought sometimes degrade performance?



Answer: It can introduce unnecessary complexity or distract the model if the task is simple, leading to worse results compared to direct answers.

Q33. How do prompts interact with fine-tuned models?



Answer: Prompts can steer fine-tuned models toward desired behaviors, but the model’s responses depend on both the prompt quality and the fine-tuning data.

Q34. Can bad prompts hurt a fine-tuned model?



Answer: Yes, poorly designed prompts can cause confusion, increase hallucinations, or override fine-tuned behaviors, reducing reliability.

Q35. How do you design prompts for tool usage?



Answer: Use clear instructions, specify tool names and expected outputs, and provide examples. Consistency and explicitness improve tool call accuracy.

Q36. How do you reduce hallucinations using prompts?



Answer: Use precise, fact-based prompts, include context or references, and discourage speculation. Reinforce correct behavior with examples.

Q37. When do prompts stop scaling and fine-tuning becomes necessary?



Answer: When prompt complexity grows, performance plateaus, or tasks require domain adaptation, fine-tuning is needed for consistent results.

5️⃣ Agentic AI & Multi-Agent Systems (JD-Specific)

Q38. What is an agentic AI system?



Answer: An agentic AI system is a model or set of models that autonomously plan, act, and interact with tools or environments to achieve goals.

Q39. What components does an agent typically have?



Answer: Agents usually have a planner, memory, tool interface, reasoning module, and a feedback loop for learning and adaptation.

Q40. How does memory work in agentic systems?



Answer: Memory stores past actions, observations, and context, enabling agents to learn, recall, and make informed decisions over time.

Q41. What’s the difference between short-term and long-term memory?



Answer: Short-term memory stores recent context and actions, while long-term memory retains historical data and knowledge for future reference and learning.

Q42. How do agents decide which tool to call?



Answer: Agents use context, task requirements, and learned policies to select the most appropriate tool for the current situation.

Q43. How do you evaluate an agent’s performance?



Answer: Evaluate using task success rate, efficiency, adaptability, and user satisfaction. Benchmarking and real-world testing are essential.

Q44. What failure modes do agentic systems have?



Answer: Common failures include infinite loops, incorrect tool calls, memory loss, and inability to adapt to new tasks or environments.

Q45. How do you prevent infinite loops in agents?



Answer: Implement loop detection, set execution limits, and use feedback mechanisms to break cycles and ensure progress.

Q46. How do multi-agent systems coordinate tasks?



Answer: Coordination is achieved through communication protocols, shared memory, and task allocation strategies to avoid conflicts and maximize efficiency.

Q47. What’s the difference between RAG and agentic workflows?



Answer: RAG (Retrieval-Augmented Generation) uses external knowledge retrieval to enhance responses, while agentic workflows involve autonomous planning, tool use, and multi-step reasoning.

6️⃣ Deployment, Scaling & Production

Q48. What challenges arise when deploying LLMs in production?



Answer: Challenges include scaling, latency, cost, security, monitoring, and handling data drift or model degradation.

Q49. How do you reduce inference latency?



Answer: Optimize model size, use quantization, batch requests, and deploy on high-performance hardware. Caching and efficient pipelines also help.

Q50. What is batching and why does it improve throughput?



Answer: Batching groups multiple requests for simultaneous processing, maximizing hardware utilization and increasing throughput.

Q51. How do you handle model versioning?



Answer: Use semantic versioning, maintain clear documentation, and track changes. Store models in a registry and automate deployment to ensure consistency.

Q52. How do you roll out a new fine-tuned model safely?



Answer: Use staged rollout, A/B testing, and monitor performance. Roll back if issues arise and communicate changes to stakeholders.

Q53. What metrics do you monitor in production?



Answer: Monitor latency, throughput, error rates, accuracy, user feedback, and resource utilization to ensure reliable operation.

Q54. How do you detect data drift for LLMs?



Answer: Track input distribution, monitor prediction changes, and use statistical tests. Retrain or update models when drift is detected.

Q55. What is A/B testing for models?



Answer: A/B testing compares two model versions on live data to measure performance differences and select the best option.

Q56. How do you handle hallucinations in production?



Answer: Use post-processing filters, monitor outputs, retrain with better data, and provide user feedback mechanisms to catch and correct errors.

Q57. What’s your strategy for continuous retraining?



Answer: Regularly collect new data, monitor performance, retrain models, and automate the pipeline for seamless updates.

7️⃣ NLP Fundamentals (They WILL Test Basics)

Q58. What is tokenization and why does it matter?



Answer: Tokenization splits text into smaller units (tokens) for processing. It’s essential for model input and affects accuracy and efficiency.

Q59. What is attention and why is it better than RNNs?



Answer: Attention allows models to focus on relevant parts of input, improving context handling. It’s more parallelizable and effective than RNNs for long sequences.

Q60. What is positional encoding?



Answer: Positional encoding adds information about token order to embeddings, enabling transformers to understand sequence structure.

Q61. Explain self-attention step by step.
Answer: Self-attention computes a weighted sum of all tokens in a sequence for each token, allowing the model to focus on relevant context. Steps: calculate query, key, value vectors; compute attention scores; apply softmax; aggregate values.

Q62. What causes long-context degradation?



Answer: As context length increases, attention scores dilute, and memory limitations cause loss of relevant information, reducing model performance.

Q63. Difference between encoder-only and decoder-only models?



Answer: Encoder-only models (e.g., BERT) process input for understanding tasks. Decoder-only models (e.g., GPT) generate output sequentially for text generation tasks.

Q64. How does BERT differ from GPT?



Answer: BERT is bidirectional and used for understanding tasks, while GPT is unidirectional and optimized for text generation.

Q65. What are common NLP evaluation metrics and their limitations?



Answer: Metrics include accuracy, F1, BLEU, ROUGE. Limitations: may not capture semantic meaning, context, or user satisfaction.

8️⃣ Scenario-Based Questions (MOST IMPORTANT)

Q66. Model accuracy dropped after deployment. What do you check first?



Answer: Check data distribution, input quality, model drift, and system changes. Validate with test cases and logs.

Q67. Fine-tuning made performance worse. Why?



Answer: Possible causes: poor data quality, overfitting, wrong hyperparameters, or loss of generalization. Review training process and data.

Q68. Users report hallucinations. How do you debug?



Answer: Analyze prompts, review training data, monitor outputs, and retrain with factual data. Add post-processing filters.

Q69. GPU cost is too high. What optimizations do you apply?



Answer: Use quantization, mixed precision, batch processing, and optimize code. Reduce model size or deploy on efficient hardware.

Q70. New domain data arrives weekly. How do you adapt the model?



Answer: Set up incremental retraining, monitor performance, and automate data integration for continuous improvement.

Q71. Latency SLA is 200ms. How do you design the system?



Answer: Use optimized models, batch processing, caching, and deploy on fast hardware. Monitor latency and scale infrastructure to meet SLA.

Q72. Agent is making wrong tool calls. How do you fix it?



Answer: Review training data, improve prompt clarity, retrain with correct examples, and add validation checks for tool selection.

Q73. Stakeholders want explainability. What do you provide?



Answer: Offer model decision logs, feature importance, visualizations, and clear documentation. Use interpretable models or post-hoc analysis tools.

Q74. How do you justify fine-tuning cost to business?



Answer: Show ROI with improved accuracy, efficiency, user satisfaction, and competitive advantage. Quantify benefits and align with business goals.

Q75. What would you improve if given 6 months?



Answer: Enhance data quality, optimize models, automate retraining, improve monitoring, and expand use cases for greater impact.
