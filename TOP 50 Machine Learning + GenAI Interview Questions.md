⚙️ 1. Core Machine Learning (Concepts)

1️⃣ What is overfitting?
When a model performs well on training data but poorly on unseen data.
✅ Fix: regularization, dropout, early stopping, more data.

2️⃣ What is underfitting?
When a model is too simple to learn underlying patterns.
✅ Fix: use a more complex model or better features.

3️⃣ What is bias–variance tradeoff?
Balancing model simplicity (bias) and complexity (variance) to achieve generalization.

4️⃣ What is cross-validation?
Splitting data into multiple folds to test model performance across subsets.

5️⃣ What are hyperparameters?
External settings controlling model training (e.g., learning rate, tree depth).

6️⃣ What is regularization?
Adding a penalty term (L1/L2) to reduce overfitting.

7️⃣ Difference between bagging and boosting?
Bagging trains models in parallel (e.g., Random Forest);
Boosting trains sequentially, focusing on previous errors (e.g., XGBoost).

8️⃣ What’s the difference between precision and recall?
Precision: correctness of positives.
Recall: coverage of actual positives.

9️⃣ What is ROC-AUC?
Area under ROC curve — measures the tradeoff between TPR and FPR.

🔟 What is data leakage?
When test information leaks into training — causes unrealistic performance.

📊 2. Algorithms & Models

11️⃣ How does logistic regression work?
Applies a sigmoid function to map linear combination of features to probabilities.

12️⃣ What is a decision tree?
Splits data using feature thresholds to minimize impurity (Gini/Entropy).

13️⃣ What is gradient boosting?
Sequentially builds trees on residual errors from previous ones.

14️⃣ What is KNN?
Classifies points based on nearest k data points in feature space.

15️⃣ What is PCA and why is it used?
Dimensionality reduction technique that finds orthogonal components maximizing variance.

🧠 3. Deep Learning

16️⃣ What is a neural network?
A series of layers that transform input data using learnable weights and nonlinear activations.

17️⃣ What is an activation function?
Adds non-linearity (ReLU, sigmoid, tanh).

18️⃣ What is dropout?
Randomly deactivates neurons during training to prevent overfitting.

19️⃣ What is batch normalization?
Normalizes activations between layers to stabilize training.

20️⃣ What is gradient descent?
An optimization algorithm that updates weights by minimizing loss function.

🧩 4. Natural Language Processing (NLP)

21️⃣ What is tokenization?
Breaking text into words, subwords, or tokens (BPE, WordPiece).

22️⃣ What is word embedding?
Vector representation of words (e.g., Word2Vec, GloVe, BERT embeddings).

23️⃣ What is attention mechanism?
Computes weighted importance of input tokens in context (core of Transformers).

24️⃣ What is BERT?
Bidirectional Transformer trained on masked language modeling and next sentence prediction.

25️⃣ What is GPT architecture?
Decoder-only Transformer trained with causal language modeling (predict next token).

🤖 5. Generative AI / LLMs

26️⃣ What is fine-tuning?
Training a pre-trained model on domain-specific data for customization.

27️⃣ What is LoRA (Low-Rank Adaptation)?
Trains small adapter matrices instead of full model weights — efficient fine-tuning.

28️⃣ What is PEFT (Parameter-Efficient Fine-Tuning)?
Umbrella term for lightweight methods like LoRA, Prefix, and Adapter tuning.

29️⃣ What is RAG (Retrieval-Augmented Generation)?
Combines a retriever (vector DB search) with an LLM to generate fact-based responses.

30️⃣ What are embeddings used for?
Convert text/images into dense vectors for semantic search or similarity.

31️⃣ What is prompt engineering?
Crafting structured prompts to guide model output effectively.

32️⃣ What is temperature in text generation?
Controls randomness — low = focused, high = creative.

33️⃣ What are top-k and top-p sampling?
Top-k limits to k highest probabilities; top-p samples until cumulative probability ≥ p.

34️⃣ What causes hallucinations in LLMs?
When models generate plausible but false information due to lack of grounding or poor context.

35️⃣ How do you reduce hallucinations?
Use RAG, better prompts, fact-checking, and retrieval-based grounding.

⚡ 6. Model Evaluation & Metrics

36️⃣ What is confusion matrix?
A 2×2 matrix showing TP, FP, FN, TN for classification.

37️⃣ What is F1-score?
Harmonic mean of precision and recall.

38️⃣ What are common regression metrics?
MSE, RMSE, MAE, R².

39️⃣ What is log loss?
Measures how uncertain predictions are compared to actual labels.

40️⃣ How do you handle imbalanced data?
SMOTE, undersampling, class weights, or F1-focused metrics.

🔧 7. MLOps / Deployment

41️⃣ What is MLOps?
End-to-end lifecycle management of ML — training → deployment → monitoring.

42️⃣ How do you deploy an ML model?
Serve via Flask/FastAPI, containerize (Docker), orchestrate (Kubernetes).

43️⃣ What is model drift?
When model performance degrades over time due to data distribution change.

44️⃣ How do you detect drift?
Monitor metrics, input distribution, concept drift tests.

45️⃣ What is model versioning?
Tracking and managing model changes with tools like MLflow or DVC.

🧮 8. Practical Engineering & System Design

46️⃣ How would you build a recommendation system?
Collaborative filtering, embeddings, or hybrid model; deploy via API and retrain periodically.

47️⃣ How do you build a chatbot using LLMs?
Use retrieval (FAISS/Pinecone) + LLM (GPT/LLaMA) + prompt templates (LangChain).

48️⃣ How do you optimize model inference speed?
Quantization, batching, caching, distillation, or GPU acceleration.

49️⃣ How do you monitor production ML systems?
Track prediction accuracy, latency, drift, and user feedback.

50️⃣ How do you ensure explainability in ML?
Use SHAP, LIME, attention visualization, and model interpretability tools.