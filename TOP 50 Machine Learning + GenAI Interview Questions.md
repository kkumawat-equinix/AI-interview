⚙️ 1. Core Machine Learning (Concepts)

1️⃣ What is overfitting?

**Overfitting** is when a model learns the training data too well, including noise, and performs poorly on new, unseen data.

**How to fix:** Use regularization, dropout, early stopping, or add more data.

2️⃣ What is underfitting?

**Underfitting** is when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test sets.

**How to fix:** Use a more complex model or better features.

3️⃣ What is bias–variance tradeoff?

The **bias–variance tradeoff** is the balance between bias (error from overly simple models) and variance (error from overly complex models). The goal is to achieve good generalization to new data.

4️⃣ What is cross-validation?

**Cross-validation** splits data into multiple folds to train and test the model on different subsets. This helps assess model performance and avoid overfitting.

5️⃣ What are hyperparameters?

**Hyperparameters** are external settings that control the training process, such as learning rate, batch size, or tree depth. They are set before training and not learned from data.

6️⃣ What is regularization?

**Regularization** adds a penalty to the loss function (like L1 or L2) to discourage complex models and reduce overfitting.

7️⃣ Difference between bagging and boosting?

**Bagging** trains multiple models in parallel on random subsets and averages their results (e.g., Random Forest).

**Boosting** trains models sequentially, each focusing on correcting the errors of the previous one (e.g., XGBoost).

8️⃣ What’s the difference between precision and recall?

**Precision** is the proportion of true positives among all predicted positives.

**Recall** is the proportion of true positives among all actual positives.

9️⃣ What is ROC-AUC?

**ROC-AUC** is the area under the Receiver Operating Characteristic curve. It measures a model’s ability to distinguish between classes, balancing true positive and false positive rates.

🔟 What is data leakage?

**Data leakage** happens when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates.

📊 2. Algorithms & Models

11️⃣ How does logistic regression work?

**Logistic regression** uses a linear combination of features and applies a sigmoid function to output probabilities for binary classification.

A decision tree splits data into branches using feature thresholds to minimize impurity (like Gini or Entropy), making decisions at each node.
12️⃣ What is a decision tree?

A **decision tree** splits data into branches using feature thresholds to minimize impurity (like Gini or Entropy), making decisions at each node.
A decision tree splits data into branches using feature thresholds to minimize impurity (like Gini or Entropy), making decisions at each node.

13️⃣ What is gradient boosting?

**Gradient boosting** builds models sequentially, where each new model tries to correct the errors of the previous ones, often using decision trees.

14️⃣ What is KNN?

**K-Nearest Neighbors (KNN)** classifies a data point based on the majority label among its k closest neighbors in the feature space.

15️⃣ What is PCA and why is it used?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that finds new orthogonal axes (principal components) capturing the most variance in the data.

🧠 3. Deep Learning

16️⃣ What is a neural network?

A **neural network** is a series of connected layers that transform input data using learnable weights and nonlinear activation functions to model complex patterns.

17️⃣ What is an activation function?

An **activation function** introduces non-linearity into a neural network, allowing it to learn complex relationships (e.g., ReLU, sigmoid, tanh).

18️⃣ What is dropout?

**Dropout** randomly deactivates a fraction of neurons during training to prevent overfitting and improve generalization.

19️⃣ What is batch normalization?

**Batch normalization** normalizes the activations of each layer to have zero mean and unit variance, which stabilizes and speeds up training.

20️⃣ What is gradient descent?

**Gradient descent** is an optimization algorithm that updates model weights to minimize the loss function by moving in the direction of steepest descent.

🧩 4. Natural Language Processing (NLP)

21️⃣ What is tokenization?

**Tokenization** is splitting text into smaller units like words, subwords, or tokens (e.g., BPE, WordPiece) for processing by NLP models.

22️⃣ What is word embedding?

**Word embeddings** are dense vector representations of words that capture their meaning and relationships (e.g., Word2Vec, GloVe, BERT embeddings).

23️⃣ What is attention mechanism?

The **attention mechanism** computes the importance of each input token in context, allowing models to focus on relevant parts of the input (core to Transformers).

24️⃣ What is BERT?

**BERT** is a bidirectional Transformer model trained on masked language modeling and next sentence prediction, enabling deep understanding of context in text.

25️⃣ What is GPT architecture?

**GPT** is a decoder-only Transformer trained with causal language modeling, predicting the next token in a sequence.

🤖 5. Generative AI / LLMs

26️⃣ What is fine-tuning?

**Fine-tuning** is training a pre-trained model further on domain-specific data to adapt it for a specific task or dataset.

27️⃣ What is LoRA (Low-Rank Adaptation)?

**LoRA** is a fine-tuning method that trains small adapter matrices within a model instead of updating all weights, making adaptation efficient and lightweight.

28️⃣ What is PEFT (Parameter-Efficient Fine-Tuning)?

**PEFT** refers to methods that fine-tune only a small part of a model (like LoRA, Prefix, or Adapter tuning), making training faster and requiring less data.

29️⃣ What is RAG (Retrieval-Augmented Generation)?

**RAG** combines a retriever (searches a vector database) with a generative model (LLM) to generate responses grounded in external documents or facts.

30️⃣ What are embeddings used for?

**Embeddings** convert text or images into dense vectors that capture meaning, enabling semantic search, similarity matching, and clustering.

31️⃣ What is prompt engineering?

**Prompt engineering** is designing clear and structured prompts to guide model outputs and improve performance for specific tasks.

32️⃣ What is temperature in text generation?

**Temperature** controls the randomness of text generation: low values make output more focused and deterministic, high values make it more creative and diverse.

33️⃣ What are top-k and top-p sampling?

**Top-k sampling** chooses from the k most probable next tokens.

**Top-p (nucleus) sampling** chooses from the smallest set of tokens whose cumulative probability exceeds p.

34️⃣ What causes hallucinations in LLMs?

**Hallucinations** occur when models generate plausible but false or unsupported information, often due to lack of grounding or poor context.

35️⃣ How do you reduce hallucinations?

Reduce hallucinations by grounding answers with retrieval (RAG), writing better prompts, fact-checking outputs, and using validation steps.

⚡ 6. Model Evaluation & Metrics

36️⃣ What is confusion matrix?

A **confusion matrix** is a table showing true positives, false positives, false negatives, and true negatives for classification results.

37️⃣ What is F1-score?

**F1-score** is the harmonic mean of precision and recall, balancing both metrics for classification tasks.

38️⃣ What are common regression metrics?

Common regression metrics include **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)**.

39️⃣ What is log loss?

**Log loss** measures the uncertainty of predictions compared to actual labels, penalizing confident but wrong predictions in classification.

40️⃣ How do you handle imbalanced data?

Handle imbalanced data using techniques like **SMOTE (oversampling)**, **undersampling**, **adjusting class weights**, or focusing on **F1-score** and **recall**.

🔧 7. MLOps / Deployment

41️⃣ What is MLOps?

**MLOps** is the practice of managing the end-to-end lifecycle of machine learning, from training and deployment to monitoring and maintenance.

42️⃣ How do you deploy an ML model?

Deploy an ML model by serving it via an API (**Flask/FastAPI**), containerizing with **Docker**, and orchestrating with tools like **Kubernetes**.

43️⃣ What is model drift?

**Model drift** is when a model’s performance degrades over time due to changes in data distribution or real-world conditions.

44️⃣ How do you detect drift?

Detect drift by monitoring prediction metrics, tracking input data distributions, and running concept drift tests.

45️⃣ What is model versioning?

**Model versioning** tracks and manages changes to models over time using tools like **MLflow** or **DVC**, ensuring reproducibility and traceability.

🧮 8. Practical Engineering & System Design

46️⃣ How would you build a recommendation system?

Use **collaborative filtering**, **embeddings**, or a **hybrid model** to generate recommendations.

Deploy via API and retrain the model regularly with new data.

47️⃣ How do you build a chatbot using LLMs?

Combine a **retriever** (like FAISS or Pinecone) with an **LLM** (like GPT or LLaMA) and use **prompt templates** (e.g., LangChain) to generate context-aware responses.

48️⃣ How do you optimize model inference speed?

Optimize inference speed using **quantization**, **batching**, **caching**, **model distillation**, or **GPU acceleration**.

49️⃣ How do you monitor production ML systems?

Monitor production ML systems by tracking **prediction accuracy**, **latency**, **drift**, and collecting **user feedback** and **logs**.

50️⃣ How do you ensure explainability in ML?

Ensure explainability by using tools like **SHAP**, **LIME**, **attention visualization**, and other model interpretability techniques.