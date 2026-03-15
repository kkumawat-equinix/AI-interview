Machine Learning Engineer Interview Q&A (2025 Edition)
⚙️ 1. ML Fundamentals

Q1. What is Machine Learning?

Q2. Difference between supervised, unsupervised, and reinforcement learning?

Q3. What is overfitting? How do you prevent it?

Q4. What is underfitting?

Q5. What is the bias–variance tradeoff?

Q6. What is cross-validation, and why is it important?

Q7. What are hyperparameters vs parameters?

Q8. Explain precision, recall, and F1-score.

Q9. What is regularization? Explain L1 and L2.

Q10. What’s the purpose of the learning rate in optimization?


📊 2. Data Preprocessing & Feature Engineering

Q11. How do you handle missing data?

Answer:
Missing data can be handled by imputation (mean, median, mode), removing rows/columns, or using algorithms that support missing values. Choice depends on data size and impact.


Q12. What is normalization vs standardization?

Answer:
Normalization scales data to a range (usually 0–1), while standardization transforms data to have mean 0 and standard deviation 1. Both help models converge and improve performance.


Q13. How do you handle categorical variables?

Answer:
Categorical variables are handled using encoding techniques like one-hot encoding, label encoding, or ordinal encoding, depending on the variable type and model requirements.


Q14. What is one-hot encoding?

Answer:
One-hot encoding converts categorical values into binary vectors, creating a new column for each category. It prevents ordinal relationships and is widely used in ML.


Q15. How do you detect and remove outliers?

Answer:
Outliers can be detected using statistical methods (z-score, IQR), visualization (boxplots), or model-based approaches. Removal depends on domain knowledge and impact on analysis.


Q16. What is feature selection vs feature extraction?

Answer:
Feature selection chooses relevant features from existing data, while feature extraction creates new features (e.g., PCA) from original data. Both improve model performance and reduce complexity.


Q17. What are PCA and its use cases?

Answer:
Principal Component Analysis (PCA) reduces dimensionality by transforming features into uncorrelated principal components. It’s used for noise reduction, visualization, and speeding up ML algorithms.


Q18. What is multicollinearity? How do you detect it?

Answer:
Multicollinearity is when features are highly correlated, causing instability in model coefficients. It’s detected using correlation matrices or VIF (Variance Inflation Factor).


Q19. Explain SMOTE (Synthetic Minority Oversampling).

Answer:
SMOTE creates synthetic samples for minority classes by interpolating between existing samples, helping balance imbalanced datasets and improve model performance.


Q20. How do you handle imbalanced datasets?

Answer:
Imbalanced datasets are handled by resampling (oversampling, undersampling), using algorithms robust to imbalance, adjusting class weights, or generating synthetic data (e.g., SMOTE).



📈 3. Algorithms (Classical ML)

Q21. How does linear regression work?

Answer:
Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a straight line (y = mx + c) to minimize error.


Q22. What assumptions does linear regression make?

Answer:
Assumptions: linearity, independence, homoscedasticity (constant variance), normality of errors, and no multicollinearity among predictors.


Q23. Difference between linear and logistic regression?

Answer:
Linear regression predicts continuous values; logistic regression predicts probabilities for binary outcomes using a sigmoid function.


Q24. What is a decision tree?

Answer:
A decision tree splits data into branches based on feature values, creating a tree structure for classification or regression. It’s easy to interpret and visualize.


Q25. How does a random forest improve over a single tree?

Answer:
Random forest combines multiple decision trees (ensemble) to reduce overfitting and improve accuracy by averaging predictions.


Q26. What is gradient boosting?

Answer:
Gradient boosting builds models sequentially, each correcting errors of the previous one, using gradient descent to minimize loss. It’s powerful for structured data.


Q27. Difference between AdaBoost, XGBoost, LightGBM, and CatBoost?

Answer:
AdaBoost uses weighted ensembles, XGBoost is optimized for speed and performance, LightGBM uses leaf-wise growth for efficiency, CatBoost handles categorical features natively.


Q28. How does KNN (K-nearest neighbors) work?

Answer:
KNN classifies data based on the majority class of its k nearest neighbors in feature space. It’s simple, non-parametric, and works well for small datasets.


Q29. What is Naive Bayes?

Answer:
Naive Bayes is a probabilistic classifier based on Bayes’ theorem, assuming feature independence. It’s fast, works well for text classification, and handles large datasets.


Q30. What is SVM (Support Vector Machine)? What is the kernel trick?

Answer:
SVM finds the optimal hyperplane to separate classes. The kernel trick allows SVM to operate in higher dimensions by transforming data, enabling non-linear classification.



🧮 4. Optimization & Training

Q31. What is gradient descent?

Answer:
Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize loss by moving in the direction of steepest descent.


Q32. Difference between batch, mini-batch, and stochastic gradient descent?

Answer:
Batch uses all data per update, mini-batch uses subsets, stochastic uses one sample. Mini-batch balances speed and stability, commonly used in deep learning.


Q33. What is an exploding or vanishing gradient?

Answer:
Exploding gradients cause large updates, destabilizing training; vanishing gradients cause tiny updates, slowing learning. Both are common in deep networks.


Q34. How does momentum help in optimization?

Answer:
Momentum accelerates gradient descent by adding a fraction of previous updates, helping escape local minima and speeding up convergence.


Q35. What are Adam, RMSProp, and SGD optimizers?

Answer:
SGD updates parameters with each batch; RMSProp adapts learning rates per parameter; Adam combines momentum and adaptive learning rates for fast, stable training.


Q36. What is early stopping and why is it useful?

Answer:
Early stopping halts training when validation performance stops improving, preventing overfitting and saving resources.


Q37. How do you tune hyperparameters efficiently?

Answer:
Hyperparameters are tuned using grid search, random search, Bayesian optimization, or automated tools. Efficient tuning improves model performance.


Q38. What is Bayesian Optimization?

Answer:
Bayesian Optimization uses probabilistic models to find optimal hyperparameters, balancing exploration and exploitation, and requiring fewer evaluations.


Q39. What is grid search vs random search?

Answer:
Grid search exhaustively tries all combinations; random search samples randomly. Random search is often more efficient for high-dimensional spaces.


Q40. What is dropout and why is it used?

Answer:
Dropout randomly disables neurons during training, preventing overfitting and improving generalization in neural networks.



🧠 5. Deep Learning

Q41. What is a neural network?

Answer:
A neural network is a computational model inspired by the human brain, consisting of interconnected layers of nodes (neurons) that learn patterns from data for tasks like classification and regression.

Q42. What are activation functions? Examples?

Answer:
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Examples include ReLU, sigmoid, and tanh.

Q43. What is ReLU and why is it preferred?

Answer:
ReLU (Rectified Linear Unit) outputs zero for negative values and the input for positive values. It’s preferred for its simplicity and helps mitigate vanishing gradient issues.

Q44. What is backpropagation?

Answer:
Backpropagation is an algorithm for training neural networks by computing gradients of the loss function and updating weights to minimize error.

Q45. What is a convolutional neural network (CNN)?

Answer:
A CNN is a neural network specialized for processing grid-like data (e.g., images) using convolutional layers to extract spatial features.

Q46. Explain pooling layers (max vs average).

Answer:
Pooling layers reduce the spatial size of feature maps. Max pooling selects the maximum value; average pooling computes the mean, both helping reduce computation and overfitting.

Q47. What is batch normalization?

Answer:
Batch normalization normalizes layer inputs during training, stabilizing learning, speeding up convergence, and allowing higher learning rates.

Q48. What is a recurrent neural network (RNN)?

Answer:
An RNN is a neural network designed for sequential data, where outputs depend on previous inputs, making it suitable for time series and language tasks.

Q49. What are LSTM and GRU?

Answer:
LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are advanced RNN architectures that address vanishing gradient problems and capture long-term dependencies in sequences.

Q50. What are attention mechanisms?

Answer:
Attention mechanisms allow neural networks to focus on relevant parts of input sequences, improving performance in tasks like translation and text generation.



🤖 6. Generative AI (GenAI Focus)

Q51. What is a generative model?

Answer:
A generative model learns the distribution of data to generate new samples similar to the training data, such as images, text, or audio.

Q52. What’s the difference between discriminative and generative models?

Answer:
Discriminative models predict labels given input data, while generative models learn to generate data and model the joint probability of inputs and outputs.

Q53. Explain how an autoencoder works.

Answer:
An autoencoder is a neural network that compresses input data into a lower-dimensional representation and then reconstructs it, used for denoising and dimensionality reduction.

Q54. What is a Variational Autoencoder (VAE)?

Answer:
A VAE is a type of autoencoder that learns a probabilistic latent space, enabling generation of new data samples and smooth interpolation between them.

Q55. What is a GAN (Generative Adversarial Network)?

Answer:
A GAN consists of a generator and a discriminator network competing against each other, resulting in realistic synthetic data generation.

Q56. What are diffusion models?

Answer:
Diffusion models generate data by iteratively denoising random noise, achieving high-quality results in image and audio synthesis.

Q57. What is a Transformer?

Answer:
A Transformer is a neural network architecture based on self-attention, excelling at sequence modeling tasks like NLP and used in large language models.

Q58. What is self-attention?

Answer:
Self-attention allows a model to weigh the importance of different parts of an input sequence, capturing dependencies regardless of their position.

Q59. How do large language models (LLMs) like GPT work?

Answer:
LLMs use Transformer architectures trained on massive text corpora to generate coherent, context-aware text based on input prompts.

Q60. What are embeddings in NLP?

Answer:
Embeddings are dense vector representations of words or sentences, capturing semantic meaning and enabling efficient processing in NLP tasks.



💬 7. NLP (Natural Language Processing)

Q61. What is tokenization?

Answer:
Tokenization is the process of splitting text into smaller units, such as words, subwords, or characters, for easier processing in NLP tasks.

Q62. What is stemming vs lemmatization?

Answer:
Stemming reduces words to their root form by removing suffixes, while lemmatization converts words to their base or dictionary form using linguistic rules.

Q63. What is TF-IDF?

Answer:
TF-IDF (Term Frequency–Inverse Document Frequency) is a statistical measure that evaluates the importance of a word in a document relative to a corpus, used for text representation.

Q64. What is word2vec?

Answer:
Word2vec is a neural network model that learns vector representations of words based on their context, capturing semantic relationships.

Q65. What is BERT and how is it different from GPT?

Answer:
BERT is a bidirectional Transformer model for NLP tasks, while GPT is a unidirectional (left-to-right) Transformer used for text generation. BERT excels at understanding context; GPT excels at generating text.

Q66. What is fine-tuning in NLP?

Answer:
Fine-tuning adapts a pre-trained NLP model to a specific task or dataset by further training on labeled data, improving performance for that task.

Q67. What are positional embeddings?

Answer:
Positional embeddings encode the position of tokens in a sequence, allowing Transformer models to capture order information.

Q68. What is masked language modeling?

Answer:
Masked language modeling is a training technique where some tokens are masked and the model learns to predict them, used in models like BERT.

Q69. What is the difference between encoder and decoder?

Answer:
Encoders process input sequences to create representations; decoders generate output sequences from those representations, used in tasks like translation.

Q70. What are attention heads?

Answer:
Attention heads are components in Transformer models that learn to focus on different parts of the input, capturing multiple relationships in parallel.



🧩 8. Model Evaluation & Metrics

Q71. What is a confusion matrix?

Answer:
A confusion matrix is a table used to evaluate classification models, showing the counts of true positives, false positives, true negatives, and false negatives.

Q72. Explain AUC-ROC curve.

Answer:
AUC-ROC curve plots the true positive rate against the false positive rate at various thresholds. AUC measures the model's ability to distinguish between classes.

Q73. What is log loss?

Answer:
Log loss (cross-entropy loss) measures the accuracy of a classifier by penalizing false predictions, especially those with high confidence.

Q74. What’s the difference between accuracy and precision?

Answer:
Accuracy is the proportion of correct predictions; precision is the proportion of true positives among predicted positives, important for imbalanced datasets.

Q75. What are regression evaluation metrics (MSE, RMSE, MAE)?

Answer:
MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error) measure prediction errors in regression models.

Q76. What is R² (coefficient of determination)?

Answer:
R² measures how well the model explains the variance in the target variable, with values closer to 1 indicating better fit.

Q77. What is cross-entropy loss?

Answer:
Cross-entropy loss quantifies the difference between predicted and actual probability distributions, commonly used in classification tasks.

Q78. How do you evaluate clustering performance?

Answer:
Clustering performance is evaluated using metrics like silhouette score, Davies-Bouldin index, and visual inspection of cluster separation.

Q79. What is silhouette score?

Answer:
Silhouette score measures how similar an object is to its own cluster compared to other clusters, ranging from -1 to 1.

Q80. What’s the difference between micro and macro averages in multi-class classification?

Answer:
Micro averaging aggregates contributions of all classes to compute metrics; macro averaging computes metrics for each class and averages them, useful for imbalanced classes.



🧱 9. Deployment & MLOps

Q81. What is MLOps?

Answer:
MLOps is a set of practices that combines machine learning, DevOps, and data engineering to automate and streamline the deployment, monitoring, and management of ML models in production.

Q82. How do you deploy an ML model to production?

Answer:
Deploy an ML model using APIs, containers (Docker), cloud services, or specialized serving platforms. Ensure scalability, security, and monitoring.

Q83. What is model drift and how do you detect it?

Answer:
Model drift occurs when model performance degrades due to changes in data distribution. Detect it by monitoring metrics and comparing predictions to actual outcomes.

Q84. How do you version your ML models?

Answer:
Version ML models using tools like MLflow, DVC, or custom naming conventions, tracking changes and ensuring reproducibility.

Q85. What’s the role of Docker in ML deployment?

Answer:
Docker packages ML models and dependencies into containers, enabling consistent, portable, and scalable deployment across environments.

Q86. What is CI/CD in ML pipelines?

Answer:
CI/CD automates building, testing, and deploying ML models, ensuring rapid, reliable updates and integration into production workflows.

Q87. What is TensorFlow Serving?

Answer:
TensorFlow Serving is a flexible, high-performance serving system for deploying TensorFlow models in production via APIs.

Q88. What is TorchServe?

Answer:
TorchServe is a tool for serving PyTorch models, providing APIs for inference, model management, and monitoring.

Q89. How do you monitor model performance post-deployment?

Answer:
Monitor model performance using metrics dashboards, logging predictions, tracking drift, and setting alerts for anomalies.

Q90. What is A/B testing in ML?

Answer:
A/B testing compares two model versions by splitting traffic, measuring performance, and selecting the best model based on real-world results.

Q91. How would you fine-tune an LLM (like GPT or LLaMA)?

Answer:
Fine-tune an LLM by training it further on domain-specific data using supervised learning, adjusting weights to improve performance for targeted tasks.

Q92. What is LoRA (Low-Rank Adaptation)?

Answer:
LoRA is a parameter-efficient fine-tuning method that adds low-rank matrices to model weights, enabling fast adaptation with minimal resources.

Q93. What is PEFT (Parameter-Efficient Fine-Tuning)?

Answer:
PEFT techniques optimize model adaptation by updating only a subset of parameters, reducing compute and memory requirements during fine-tuning.

Q94. How do you build a Retrieval-Augmented Generation (RAG) pipeline?

Answer:
A RAG pipeline combines a retriever (fetches relevant documents) and a generator (produces answers), enhancing LLMs with external knowledge for accurate responses.

Q95. What is a vector database?

Answer:
A vector database stores and indexes embeddings, enabling fast similarity search for tasks like retrieval, recommendation, and semantic search.

Q96. What are embeddings used for?

Answer:
Embeddings represent data (text, images) as dense vectors, capturing semantic meaning for efficient search, clustering, and input to ML models.

Q97. How do you evaluate generative outputs (text, image)?

Answer:
Evaluate generative outputs using metrics like BLEU, ROUGE (text), FID, IS (images), human judgment, and task-specific criteria.

Q98. What’s the role of prompt engineering?

Answer:
Prompt engineering designs and refines input prompts to guide LLMs, improving output quality, relevance, and reducing errors or hallucinations.

Q99. What are safety and hallucination issues in GenAI?

Answer:
Safety issues involve harmful or biased outputs; hallucinations are false or nonsensical responses. Mitigate with prompt design, filtering, and human review.

Q100. How do you optimize inference time for large models?

Answer:
Optimize inference by quantization, pruning, batching, using efficient hardware, and deploying models with scalable serving solutions.

🧮 11. Math & Statistics (Core)

Q101. What is probability distribution?

Answer:
A probability distribution describes how probabilities are assigned to possible values of a random variable, such as normal, binomial, or Poisson distributions.

Q102. What’s the difference between covariance and correlation?

Answer:
Covariance measures how two variables change together; correlation standardizes covariance, showing the strength and direction of their relationship (range -1 to 1).

Q103. What is the central limit theorem?

Answer:
The central limit theorem states that the sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the original distribution.

Q104. Explain Bayes’ theorem with an example.

Answer:
Bayes’ theorem calculates the probability of an event based on prior knowledge. Example: Updating disease probability after a positive test result.

Q105. What is entropy in information theory?

Answer:
Entropy measures the uncertainty or randomness in a dataset, quantifying the amount of information contained in a variable.

Q106. What is gradient mathematically?

Answer:
A gradient is a vector of partial derivatives indicating the direction and rate of fastest increase of a function, used in optimization.

Q107. What is the difference between convex and non-convex functions?

Answer:
Convex functions have a single global minimum; non-convex functions can have multiple local minima, making optimization harder.

Q108. What is a cost function?

Answer:
A cost function quantifies the error between predicted and actual values, guiding model training to minimize this error.

Q109. Explain eigenvalues and eigenvectors.

Answer:
Eigenvalues and eigenvectors describe how linear transformations scale and rotate data, used in PCA and other ML techniques.

Q110. What is a Markov chain?

Answer:
A Markov chain is a stochastic process where the next state depends only on the current state, not previous states.

Q111. What’s the difference between TensorFlow and PyTorch?

Answer:
TensorFlow and PyTorch are deep learning frameworks. PyTorch is more Pythonic and flexible; TensorFlow is production-oriented and supports deployment tools.

Q112. How do you use scikit-learn for ML tasks?

Answer:
Scikit-learn provides tools for data preprocessing, model training, evaluation, and pipelines for classical ML tasks in Python.

Q113. What is ONNX?

Answer:
ONNX (Open Neural Network Exchange) is an open format for representing ML models, enabling interoperability between frameworks.

Q114. What’s the use of Hugging Face Transformers?

Answer:
Hugging Face Transformers provides pre-trained models and tools for NLP, enabling easy fine-tuning and deployment of state-of-the-art models.

Q115. What is LangChain?

Answer:
LangChain is a framework for building applications with LLMs, enabling chaining of prompts, retrieval, and integration with external tools.

Q116. What is MLflow?

Answer:
MLflow is an open-source platform for managing ML experiments, tracking metrics, packaging models, and deploying them.

Q117. What’s the role of Kubernetes in ML?

Answer:
Kubernetes orchestrates containerized ML workloads, enabling scalable, automated deployment and management of models and data pipelines.

Q118. How do you scale model inference?

Answer:
Scale model inference by using batch processing, distributed systems, optimized hardware, and autoscaling infrastructure.

Q119. What’s the difference between batch inference and online inference?

Answer:
Batch inference processes multiple inputs at once, suitable for offline tasks; online inference handles real-time predictions for individual requests.

Q120. How do you quantize or compress a model?

Answer:
Quantize or compress models by reducing precision (e.g., float32 to int8), pruning weights, or using distillation, improving speed and reducing memory usage.

🧠 121–140 | Advanced ML Theory

What are the assumptions behind logistic regression?

How do you interpret model coefficients in logistic regression?

What’s the ROC curve intuition?

Why does gradient boosting use shallow trees?

What is label smoothing and why is it used?

What is the difference between hard and soft margin in SVM?

How does a kernel function work mathematically?

What is the curse of overparameterization?

How do you visualize high-dimensional data?

What is the difference between supervised pretraining and self-supervised learning?

What are attention masks used for in Transformers?

What is zero-shot learning?

What is few-shot learning?

What is contrastive learning?

How does CLIP (Contrastive Language–Image Pretraining) work?

What are the challenges in training LLMs from scratch?

How does reinforcement learning from human feedback (RLHF) work?

What is policy gradient in RL?

What is PPO (Proximal Policy Optimization)?

What’s the difference between actor-critic and value-based methods?



🧩 141–160 | Data Engineering & Pipelines

What is a data pipeline in ML?

How do you design a scalable data ingestion system?

What is ETL vs ELT?

What’s the purpose of feature normalization in pipelines?

How do you monitor data drift?

What is concept drift vs data drift?

How do you handle streaming data for ML?

What’s Apache Kafka used for in ML?

What are feature stores like Feast or Tecton?

How do you automate data preprocessing in production?

How would you implement online feature generation?

What is schema validation and why is it important?

What is a DAG (Directed Acyclic Graph) in data workflows?

What tools can orchestrate ML pipelines (Airflow, Kubeflow, Prefect)?

How do you log and track dataset versions?

What is data lineage?

How do you ensure reproducibility in ML experiments?

What are the benefits of TFRecords and Parquet formats?

What’s the difference between batch and real-time feature pipelines?

What is federated learning?



⚙️ 161–180 | MLOps & Deployment

What are the main stages of an MLOps lifecycle?

What are CI/CD pipelines for ML?

How do you deploy an ML model using Docker?

How would you deploy a model on AWS / GCP / Azure?

What is model registry?

What is canary deployment?

What’s the difference between batch inference and online inference?

How do you monitor latency and throughput of an ML API?

What is model rollback?

How do you perform shadow deployment?

What is A/B testing in ML deployment?

How do you handle model versioning?

What are monitoring metrics for ML models?

What is concept drift detection?

How do you retrain models automatically?

What is a serving layer in ML architecture?

What’s the difference between REST and gRPC for model serving?

How do you secure ML APIs?

What is Kubernetes and how does it help in MLOps?

What are microservices in the context of ML systems?



🤖 181–200 | GenAI & LLMs

What is fine-tuning vs prompt-tuning?

What is adapter tuning?

What is LoRA and why is it efficient?

What is quantization in LLMs?

What is pruning and how does it reduce model size?

How does a transformer encoder differ from a decoder?

Explain key, query, and value in attention mechanisms.

What is multi-head attention and why use it?

What is position encoding in Transformers?

What is BPE (Byte Pair Encoding)?

What is tokenization in LLMs?

What is perplexity in language models?

What is temperature in text generation?

What is top-k and top-p (nucleus) sampling?

What are hallucinations in LLMs and how can you reduce them?

How do you evaluate LLM responses automatically?

What are vector embeddings and how are they computed?

What’s the difference between OpenAI Embeddings and BERT embeddings?

How does a Retrieval-Augmented Generation (RAG) pipeline work?

How would you connect a vector database (like Pinecone or FAISS) to an LLM?



📊 201–220 | Evaluation, Metrics & Experimentation

What is model calibration?

How do you evaluate regression models beyond MSE?

What is a confusion matrix used for?

How do you compute AUC manually?

What is Cohen’s Kappa?

How do you evaluate unsupervised models?

What is a silhouette coefficient?

How do you evaluate a recommender system?

What is MAP@K and NDCG@K?

What is precision@k?

How do you design an offline evaluation for ranking models?

What’s the difference between validation and test data?

What is data leakage and how do you prevent it?

What’s an ablation study in ML?

What’s the purpose of random seeds in experiments?

What are the benefits of stratified sampling?

How do you compare two models statistically?

What’s a baseline model and why is it useful?

What is bootstrapping in evaluation?

What is confidence interval in ML metrics?



🧩 221–240 | Real-World ML System Design & Scenarios

Design an ML system to recommend YouTube videos.

How would you detect fraud transactions in real time?

How would you build a personalized chatbot for a company?

Design a system to detect fake news using LLMs.

Build a model that predicts user churn for a SaaS product.

How would you architect an end-to-end image classification pipeline?

How do you design a vector search engine?

How do you scale inference for millions of users?

How do you design a continuous training pipeline?

What caching strategies can you use for ML inference?

What is model distillation and when should you use it?

How do you integrate ML models with APIs and frontends?

What are common data privacy issues in ML systems?

How do you ensure explainability in your models?

What is SHAP and LIME for explainability?

How would you test ML models before deployment?

How do you log and debug model predictions?

What is a feedback loop in ML systems?

How do you handle adversarial attacks on models?

How would you build a GenAI-powered summarization tool end-to-end?