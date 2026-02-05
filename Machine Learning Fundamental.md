## 📚 Part 1: Core ML Concepts (Questions 1-50)

### 1. What is Machine Learning?
**Answer:** Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data, learn patterns, and make predictions or decisions. The core idea is that machines can identify patterns in data and make decisions with minimal human intervention.

**Key Types:** Supervised, Unsupervised, Reinforcement Learning

---

### 2. What are the different types of Machine Learning?
**Answer:** 
- **Supervised Learning:** Learning from labeled data (classification, regression)
- **Unsupervised Learning:** Finding patterns in unlabeled data (clustering, dimensionality reduction)
- **Reinforcement Learning:** Learning through reward/penalty feedback (game playing, robotics)
- **Semi-supervised Learning:** Mix of labeled and unlabeled data
- **Self-supervised Learning:** Creating labels from data itself

---

### 3. What is supervised learning? Give examples.
**Answer:** Supervised learning uses labeled training data where each input has a corresponding output. The algorithm learns the mapping function from input to output.

**Examples:**
- **Classification:** Email spam detection, image recognition, sentiment analysis
- **Regression:** House price prediction, stock price forecasting, temperature prediction

**Key Algorithms:** Linear/Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks

---

### 4. What is unsupervised learning? Give examples.
**Answer:** Unsupervised learning works with unlabeled data to discover hidden patterns or structures without predefined outputs.

**Examples:**
- **Clustering:** Customer segmentation, document grouping, anomaly detection
- **Dimensionality Reduction:** PCA for visualization, feature extraction
- **Association:** Market basket analysis (items bought together)

**Key Algorithms:** K-Means, DBSCAN, Hierarchical Clustering, PCA, t-SNE, Autoencoders

---

### 5. What is reinforcement learning?
**Answer:** RL is learning through interaction with an environment. An agent takes actions, receives rewards or penalties, and learns to maximize cumulative reward over time.

**Key Components:**
- **Agent:** Learner/decision maker
- **Environment:** What agent interacts with
- **State:** Current situation
- **Action:** Choices available
- **Reward:** Feedback signal
- **Policy:** Strategy for choosing actions

**Examples:** AlphaGo, self-driving cars, robot navigation, game AI

---
### 6. What is the difference between classification and regression?
**Answer:**

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Output** | Discrete/categorical | Continuous numerical |
| **Goal** | Predict class labels | Predict quantities |
| **Examples** | Spam/Not Spam, Cat/Dog | Price, Temperature, Age |
| **Metrics** | Accuracy, Precision, F1 | MSE, RMSE, MAE, R² |
| **Algorithms** | Logistic Regression, SVM | Linear Regression, Ridge |

---

### 7. What is overfitting?
**Answer:** Overfitting occurs when a model learns training data too well, including noise and outliers, resulting in poor generalization to new data.

**Symptoms:**
- High training accuracy, low test accuracy
- Model is too complex for the data
- Too many features relative to samples

**Solutions:**
- Regularization (L1/L2)
- Cross-validation
- Reduce model complexity
- More training data
- Dropout (neural networks)
- Early stopping
- Pruning (decision trees)

---

### 8. What is underfitting?
**Answer:** Underfitting occurs when a model is too simple to capture the underlying patterns in data, resulting in poor performance on both training and test sets.

**Symptoms:**
- Low training AND test accuracy
- Model is too simplistic
- High bias

**Solutions:**
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer
- Use more powerful algorithms

---

### 9. What is the bias–variance tradeoff?
**Answer:** The bias-variance tradeoff is the fundamental tension between model simplicity and flexibility.

**Bias:** Error from incorrect assumptions. High bias → underfitting  
**Variance:** Error from sensitivity to training data fluctuations. High variance → overfitting

**Total Error = Bias² + Variance + Irreducible Error**

**Goal:** Find the sweet spot that minimizes total error
- Simple models: High bias, low variance
- Complex models: Low bias, high variance

---

### 10. What is cross-validation? Why is it used?
**Answer:** Cross-validation is a technique to assess model performance by splitting data into multiple folds, training on some and validating on others.

**K-Fold CV:**
1. Split data into K folds
2. Train on K-1 folds, validate on 1
3. Repeat K times, rotating validation fold
4. Average performance across folds

**Why use it:**
- More reliable performance estimate
- Uses all data for both training and validation
- Reduces overfitting risk
- Better for small datasets
- Helps in hyperparameter tuning

**Variants:** Stratified K-Fold, Leave-One-Out, TimeSeriesSplit

---
### 11. What is a training set and testing set?
**Answer:**
- **Training Set (70-80%):** Data used to train the model, learning patterns and parameters
- **Testing Set (20-30%):** Unseen data used to evaluate final model performance

**Key Principle:** Never use test data during training to ensure unbiased evaluation

---

### 12. What is a validation set?
**Answer:** Validation set is used to tune hyperparameters and make model selection decisions.

**Purpose:**
- Tune hyperparameters (learning rate, number of layers, etc.)
- Select best model architecture
- Decide when to stop training (early stopping)
- Compare different models

**Split:** Training (60-70%) | Validation (10-20%) | Test (20-30%)

---

### 13. What is feature engineering?
**Answer:** Feature engineering is the process of creating, transforming, and selecting features to improve model performance.

**Techniques:**
- **Creation:** Polynomial features, interaction terms, aggregations
- **Transformation:** Log, sqrt, binning, encoding
- **Extraction:** PCA, text embeddings, image features
- **Selection:** Filter, wrapper, embedded methods
- **Domain-specific:** Date → day/month/year, text → TF-IDF

**Impact:** Often more important than model choice for performance

---

### 14. What is feature scaling?
**Answer:** Feature scaling normalizes feature ranges to prevent features with larger scales from dominating.

**Two Main Types:**

**Standardization (Z-score normalization):**
- Formula: `z = (x - μ) / σ`
- Result: Mean=0, Std=1
- Use when: Features follow normal distribution
- Algorithms: Linear models, SVM, Neural Networks

**Normalization (Min-Max scaling):**
- Formula: `x_scaled = (x - min) / (max - min)`
- Result: Range [0, 1]
- Use when: Need bounded range
- Sensitive to outliers

**When needed:** Distance-based algorithms (KNN, SVM), gradient descent optimization

---

### 15. What is the difference between parametric and non-parametric models?
**Answer:**

**Parametric Models:**
- Fixed number of parameters
- Makes assumptions about data distribution
- Faster, simpler, interpretable
- Examples: Linear Regression, Logistic Regression, Naive Bayes
- Risk: High bias if assumptions wrong

**Non-parametric Models:**
- Number of parameters grows with data
- Fewer assumptions about data
- More flexible, can model complex patterns
- Examples: KNN, Decision Trees, Random Forest, SVM with RBF kernel
- Risk: Overfitting, computational cost

---

### 16. What is a loss function?
**Answer:** Loss function measures how wrong the model's predictions are. It quantifies the difference between predicted and actual values.

**Common Loss Functions:**
- **Regression:** MSE, RMSE, MAE, Huber Loss
- **Binary Classification:** Binary Cross-Entropy
- **Multi-class Classification:** Categorical Cross-Entropy
- **SVM:** Hinge Loss

**Goal:** Minimize loss during training through optimization

---

### 17. What is gradient descent?
**Answer:** Gradient descent is an iterative optimization algorithm to minimize the loss function by updating parameters in the direction of steepest descent.

**Algorithm:**
1. Initialize parameters randomly
2. Calculate gradient (derivative) of loss w.r.t. parameters
3. Update: `θ = θ - α * ∇J(θ)`
4. Repeat until convergence

**α (learning rate):** Controls step size
- Too small: Slow convergence
- Too large: May overshoot minimum

**Types:**
- Batch GD: Uses all data (slow)
- Stochastic GD: Uses one sample (noisy)
- Mini-batch GD: Uses small batches (best balance)

---

### 18. What is stochastic gradient descent (SGD)?
**Answer:** SGD updates parameters using one random training example at a time, rather than the entire dataset.

**Advantages:**
- Faster iterations
- Can escape local minima due to noise
- Works with large datasets
- Online learning capability

**Disadvantages:**
- Noisy updates
- Slower convergence
- Requires learning rate tuning

**Mini-batch SGD:** Best practice - uses small batches (32, 64, 128) for balance between speed and stability

---

### 19. What is regularization?
**Answer:** Regularization prevents overfitting by adding a penalty term to the loss function, discouraging complex models.

**Purpose:**
- Reduce model complexity
- Prevent overfitting
- Improve generalization
- Handle multicollinearity

**Types:**
- **L1 (Lasso):** Adds |coefficients| sum
- **L2 (Ridge):** Adds coefficients² sum
- **Elastic Net:** Combines L1 + L2
- **Dropout:** Randomly drops neurons (neural networks)

**Modified Loss:** `J(θ) = Loss + λ * Penalty`

---

### 20. Difference between L1 and L2 regularization?
**Answer:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Penalty** | Sum of absolute values | Sum of squares |
| **Formula** | λΣ\|w\| | λΣw² |
| **Feature Selection** | Yes (drives coefficients to 0) | No (shrinks but doesn't eliminate) |
| **Use Case** | Sparse models, feature selection | All features matter |
| **Robustness** | Handles multicollinearity | Distributes weight across correlated features |
| **Solution** | Not differentiable at 0 | Differentiable everywhere |

**Elastic Net:** Combines both: `λ₁Σ|w| + λ₂Σw²`

---

### 21. What is a confusion matrix?
**Answer:** A confusion matrix is a table showing model performance on classification tasks.

**Components:**
- **True Positive (TP):** Correctly predicted positive
- **True Negative (TN):** Correctly predicted negative
- **False Positive (FP):** Type I error (predicted positive, actually negative)
- **False Negative (FN):** Type II error (predicted negative, actually positive)

**Use:** Calculate precision, recall, accuracy, F1-score

---

### 22. What are precision, recall, and F1-score?
**Answer:**

**Precision = TP / (TP + FP)**
- "Of all predicted positives, how many are correct?"
- Use when: False positives are costly (spam detection)

**Recall (Sensitivity) = TP / (TP + FN)**
- "Of all actual positives, how many did we find?"
- Use when: False negatives are costly (disease detection)

**F1-Score = 2 * (Precision * Recall) / (Precision + Recall)**
- Harmonic mean of precision and recall
- Use when: Need balance between precision and recall
- Use when: Classes are imbalanced

**Trade-off:** Increasing precision often decreases recall and vice versa

---

### 23. What is accuracy? When is accuracy not useful?
**Answer:**

**Accuracy = (TP + TN) / (TP + TN + FP + FN)**
- Percentage of correct predictions

**Not Useful When:**
1. **Imbalanced datasets:** 95% negative class → always predicting negative gives 95% accuracy but is useless
2. **Unequal costs:** Missing cancer (FN) is worse than false alarm (FP)
3. **Multi-class problems:** Need per-class metrics

**Better Alternatives:**
- Precision, Recall, F1-score
- ROC-AUC
- Balanced accuracy
- Cohen's Kappa

---

### 24. What is ROC curve?
**Answer:** ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various classification thresholds.

**Axes:**
- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall

**Interpretation:**
- Perfect classifier: Curve goes through top-left corner
- Random classifier: Diagonal line
- Better model: More area under curve

**Use:** Compare models, select optimal threshold

---

### 25. What is AUC (Area Under Curve)?
**Answer:** AUC is the area under the ROC curve, measuring model's ability to distinguish between classes.

**Range:** 0 to 1
- **1.0:** Perfect classifier
- **0.9-1.0:** Excellent
- **0.8-0.9:** Good
- **0.7-0.8:** Fair
- **0.5:** Random (no discriminative power)
- **<0.5:** Worse than random

**Advantages:**
- Threshold-independent
- Handles class imbalance well
- Single number for model comparison

---

### 26. What is correlation?
**Answer:** Correlation measures the linear relationship between two variables, ranging from -1 to +1.

**Pearson Correlation Coefficient:**
`r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² * Σ(yi - ȳ)²]`

**Interpretation:**
- **+1:** Perfect positive correlation
- **0:** No correlation
- **-1:** Perfect negative correlation

**Important:** Correlation ≠ Causation

**Types:**
- Pearson: Linear relationships
- Spearman: Monotonic relationships (non-linear)
- Kendall: Ordinal associations

---

### 27. What is covariance?
**Answer:** Covariance measures how two variables change together.

**Formula:** `Cov(X,Y) = Σ[(xi - x̄)(yi - ȳ)] / (n-1)`

**Interpretation:**
- **Positive:** Variables increase together
- **Negative:** One increases, other decreases
- **Zero:** No linear relationship

**Issue:** Scale-dependent (unbounded)

**Correlation = Covariance / (σx * σy)** → Normalized version

---

### 28. What is multicollinearity?
**Answer:** Multicollinearity occurs when independent variables in a regression model are highly correlated.

**Problems:**
- Unstable coefficient estimates
- Large standard errors
- Difficulty interpreting individual feature effects
- Coefficient signs may be wrong

**Detection:**
- **VIF (Variance Inflation Factor):** VIF > 10 indicates multicollinearity
- **Correlation matrix:** High pairwise correlations
- **Condition number:** Large values indicate problems

**Solutions:**
- Remove correlated features
- PCA for dimensionality reduction
- Ridge regression (L2 regularization)
- Domain knowledge to select features

---

### 29. What is dimensionality reduction?
**Answer:** Reducing the number of features while preserving important information.

**Why:**
- Curse of dimensionality
- Reduce computation
- Visualization (2D/3D)
- Remove noise
- Address multicollinearity
- Prevent overfitting

**Methods:**
- **Feature Selection:** Select subset of original features
  - Filter: Statistical tests
  - Wrapper: Recursive feature elimination
  - Embedded: Lasso, tree importance
  
- **Feature Extraction:** Create new features
  - PCA: Linear transformation
  - t-SNE: Non-linear, for visualization
  - Autoencoders: Neural network approach

---

### 30. What is PCA? (Principal Component Analysis)
**Answer:** PCA is an unsupervised linear dimensionality reduction technique that transforms data into orthogonal principal components.

**How it works:**
1. Standardize data
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Sort by eigenvalue (variance explained)
5. Select top K components
6. Transform data

**Principal Components:**
- Linear combinations of original features
- Ordered by variance explained
- Orthogonal (uncorrelated)

**Use Cases:**
- Dimensionality reduction
- Noise reduction
- Visualization
- Feature extraction

**Limitation:** Assumes linear relationships, less interpretable

---

### 31. What is kNN algorithm?
**Answer:** k-Nearest Neighbors is a non-parametric, instance-based algorithm that classifies based on majority vote of k nearest neighbors.

**Algorithm:**
1. Choose k (number of neighbors)
2. Calculate distance from test point to all training points
3. Find k nearest neighbors
4. Classification: Majority vote | Regression: Average

**Distance Metrics:**
- Euclidean (most common)
- Manhattan
- Minkowski
- Cosine

**Pros:** Simple, no training phase, handles non-linear boundaries  
**Cons:** Slow prediction, sensitive to scaling, curse of dimensionality

**k Selection:** Use cross-validation; odd k for binary classification

---

### 32. What is linear regression?
**Answer:** Linear regression models the relationship between dependent variable Y and independent variables X using a linear equation.

**Simple:** `Y = β₀ + β₁X + ε`  
**Multiple:** `Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε`

**Objective:** Minimize sum of squared residuals (OLS - Ordinary Least Squares)

**Output:** Continuous numerical values

**Evaluation:** R², MSE, RMSE, MAE

---

### 33. What assumptions does linear regression make?
**Answer:**

1. **Linearity:** Relationship between X and Y is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of residuals
4. **Normality:** Residuals are normally distributed
5. **No multicollinearity:** Independent variables are not highly correlated

**Checking Assumptions:**
- Residual plots
- Q-Q plots for normality
- VIF for multicollinearity
- Durbin-Watson test for autocorrelation

**Violations:** Use transformations, robust regression, or different models

---

### 34. What is logistic regression?
**Answer:** Logistic regression is used for binary classification, predicting probability of an event using sigmoid function.

**Formula:** `P(Y=1|X) = 1 / (1 + e^-(β₀ + β₁X))`

**Output:** Probability [0, 1]  
**Decision:** If P > 0.5 → Class 1, else Class 0

**Loss Function:** Binary cross-entropy (log loss)

**Advantages:**
- Probabilistic output
- Interpretable coefficients
- Works well for linearly separable data
- Regularization available (L1/L2)

**Multi-class:** Softmax regression (multinomial logistic regression)

---

### 35. What is a decision tree?
**Answer:** Decision tree is a tree-structured classifier that makes decisions by splitting data based on feature values.

**Structure:**
- **Root node:** Entire dataset
- **Internal nodes:** Feature tests
- **Branches:** Outcomes of tests
- **Leaf nodes:** Class labels or values

**Splitting Criteria:**
- Classification: Gini impurity, Information gain (entropy)
- Regression: Variance reduction

**Pros:**
- Interpretable
- Handles non-linear relationships
- No feature scaling needed
- Handles missing values

**Cons:**
- Prone to overfitting
- Unstable (small data changes → different tree)
- Biased toward features with many categories

**Solutions:** Pruning, Random Forest, Gradient Boosting

---

### 36. What is entropy and information gain?
**Answer:**

**Entropy:** Measure of impurity/randomness in data  
`H(S) = -Σ p(i) * log₂(p(i))`

**Values:**
- 0: Pure (all same class)
- 1: Maximum impurity (equal distribution)

**Information Gain:** Reduction in entropy after splitting  
`IG = H(parent) - Σ [|Sᵢ|/|S| * H(Sᵢ)]`

**Use:** Select best feature to split on (maximize information gain)

**Example:**
- Parent entropy: 1.0 (50-50 split)
- After split: 0.5 (pure children)
- Information Gain: 0.5

---

### 37. What is Gini impurity?
**Answer:** Gini impurity measures the probability of incorrectly classifying a randomly chosen element.

**Formula:** `Gini = 1 - Σ p(i)²`

**Range:** 0 (pure) to 0.5 (binary, maximum impurity)

**vs Entropy:**
- Gini: Faster to compute (no logarithm)
- Entropy: More sensitive to impurity
- Both give similar results in practice
- CART (Classification and Regression Trees) uses Gini

**Decision trees use Gini or entropy to find optimal splits**

---

### 38. What is pruning in decision trees?
**Answer:** Pruning reduces tree complexity by removing sections that provide little predictive power, preventing overfitting.

**Types:**

**Pre-pruning (Early Stopping):**
- Limit max depth
- Minimum samples per leaf
- Minimum samples to split
- Stop when information gain < threshold

**Post-pruning (Backward):**
- Grow full tree
- Remove nodes that don't improve validation performance
- Cost-complexity pruning (α parameter)

**Benefits:** Better generalization, simpler model, faster prediction

---

### 39. What is a random forest?
**Answer:** Random Forest is an ensemble method that builds multiple decision trees and combines their predictions.

**How it works:**
1. Bootstrap sampling (with replacement)
2. For each tree, randomly select subset of features
3. Build decision tree on sample
4. Repeat for n_trees
5. Aggregate: Majority vote (classification) or average (regression)

**Why better than single tree:**
- Reduces overfitting
- Reduces variance
- More robust
- Better generalization
- Feature importance

**Hyperparameters:** n_estimators, max_depth, max_features, min_samples_split

---

### 40. What is bagging?
**Answer:** Bagging (Bootstrap Aggregating) trains multiple models on different random subsets of data and averages predictions.

**Process:**
1. Create k bootstrap samples (sampling with replacement)
2. Train model on each sample
3. Aggregate predictions
   - Classification: Voting
   - Regression: Averaging

**Benefits:**
- Reduces variance
- Prevents overfitting
- Works well with unstable models (decision trees)
- Parallel training

**Example:** Random Forest is bagging with decision trees

---

### 41. What is boosting?
**Answer:** Boosting is an ensemble technique that sequentially trains weak learners, each focusing on mistakes of previous ones.

**Process:**
1. Train base model
2. Focus on misclassified samples
3. Train next model on weighted data
4. Combine all models
5. Final prediction: Weighted voting

**Key Idea:** Convert weak learners to strong learner

**Types:**
- AdaBoost: Adaptive Boosting
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

**vs Bagging:** Boosting is sequential, reduces bias; Bagging is parallel, reduces variance

---

### 42. What is gradient boosting?
**Answer:** Gradient Boosting builds trees sequentially, each correcting errors of previous trees using gradient descent.

**Algorithm:**
1. Start with initial prediction (mean)
2. Calculate residuals (errors)
3. Train tree to predict residuals
4. Add tree to ensemble with learning rate
5. Update predictions
6. Repeat

**Key Concept:** Each tree learns the gradient (residual) of loss function

**Hyperparameters:**
- n_estimators: Number of trees
- learning_rate: Shrinkage factor
- max_depth: Tree complexity
- subsample: Fraction of samples per tree

**Strengths:** High accuracy, handles mixed data types, feature importance

---

### 43. What is XGBoost?
**Answer:** XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting.

**Key Features:**
- **Regularization:** L1 and L2 to prevent overfitting
- **Parallel processing:** Fast training
- **Handling missing values:** Built-in handling
- **Tree pruning:** Max depth, then prune backward
- **Cross-validation:** Built-in CV
- **Early stopping:** Automatic
- **Custom loss functions:** Flexible

**Why Popular:**
- State-of-the-art performance
- Wins Kaggle competitions
- Faster than traditional GBM
- Handles large datasets

**Hyperparameters:** learning_rate, max_depth, n_estimators, subsample, colsample_bytree

---

### 44. What is Naïve Bayes classifier?
**Answer:** Naïve Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence.

**Bayes Theorem:** `P(A|B) = P(B|A) * P(A) / P(B)`

**For Classification:**  
`P(Class|Features) = P(Features|Class) * P(Class) / P(Features)`

**"Naive" Assumption:** Features are conditionally independent given the class

**Types:**
- **Gaussian NB:** Continuous features (normal distribution)
- **Multinomial NB:** Discrete counts (text classification)
- **Bernoulli NB:** Binary features

**Pros:** Fast, works with small data, good for text classification  
**Cons:** Independence assumption rarely true

---

### 45. What is SVM (Support Vector Machine)?
**Answer:** SVM finds the optimal hyperplane that maximizes the margin between classes.

**Key Concepts:**
- **Hyperplane:** Decision boundary separating classes
- **Support Vectors:** Data points closest to hyperplane
- **Margin:** Distance between hyperplane and nearest points
- **Objective:** Maximize margin

**For non-linearly separable data:**
- Use kernel trick
- Map to higher dimensions

**Pros:**
- Effective in high dimensions
- Memory efficient (only uses support vectors)
- Versatile (different kernels)

**Cons:**
- Slow on large datasets
- Sensitive to feature scaling
- Hard to interpret

---

### 46. What is the kernel trick?
**Answer:** Kernel trick maps data to higher-dimensional space where it becomes linearly separable, without explicitly computing coordinates.

**Kernel Function:** `K(x, x') = φ(x) · φ(x')`

**Common Kernels:**
- **Linear:** `K(x, x') = x · x'`
- **Polynomial:** `K(x, x') = (x · x' + c)^d`
- **RBF (Radial Basis Function/Gaussian):** `K(x, x') = exp(-γ||x - x'||²)`
- **Sigmoid:** `K(x, x') = tanh(αx · x' + c)`

**Advantage:** Compute dot products in high dimensions without explicit transformation (computationally efficient)

**Use:** SVM, kernel PCA, kernel ridge regression

---

### 47. What is clustering?
**Answer:** Clustering is unsupervised learning that groups similar data points together without labels.

**Types:**
- **Partitioning:** K-means, K-medoids
- **Hierarchical:** Agglomerative, divisive
- **Density-based:** DBSCAN, OPTICS
- **Model-based:** Gaussian Mixture Models

**Use Cases:**
- Customer segmentation
- Document grouping
- Image segmentation
- Anomaly detection
- Gene sequence analysis

**Evaluation:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz index

---

### 48. What is K-means clustering?
**Answer:** K-means partitions data into K clusters by minimizing within-cluster variance.

**Algorithm:**
1. Randomly initialize K centroids
2. Assign each point to nearest centroid
3. Recalculate centroids (mean of assigned points)
4. Repeat 2-3 until convergence

**Objective:** Minimize `Σ ||xᵢ - cₖ||²` (within-cluster sum of squares)

**Hyperparameter:** K (number of clusters) - use elbow method or silhouette score

**Pros:** Simple, fast, scalable  
**Cons:** Needs K specified, sensitive to initialization, assumes spherical clusters, sensitive to outliers

**Variants:** K-means++, Mini-batch K-means

---

### 49. What is the elbow method?
**Answer:** Elbow method determines optimal K for K-means by plotting inertia (within-cluster sum of squares) vs K.

**Process:**
1. Run K-means for different K values (e.g., 1-10)
2. Calculate inertia for each K
3. Plot K vs Inertia
4. Look for "elbow" point where decrease slows

**Inertia:** `WCSS = Σ Σ ||x - cₖ||²`

**Elbow Point:** Point where adding more clusters doesn't significantly reduce inertia

**Limitation:** Elbow not always clear; combine with silhouette score

---

### 50. What is a silhouette score?
**Answer:** Silhouette score measures how well each point fits its cluster compared to other clusters.

**Formula:** `s(i) = (b(i) - a(i)) / max(a(i), b(i))`

Where:
- **a(i):** Average distance to points in same cluster
- **b(i):** Average distance to points in nearest other cluster

**Range:** -1 to +1
- **+1:** Perfect clustering
- **0:** On cluster boundary
- **-1:** Wrong cluster

**Average silhouette score** across all points measures overall clustering quality

**Use:** Compare different K values, better than elbow method

---

## 💻 Part 2: Practical ML Coding Questions (Questions 51-100)

### 51. How do you handle missing values in a dataset? Show different approaches in code.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. Remove rows with missing values
df_cleaned = df.dropna()

# 2. Remove columns with too many missing values
df_cleaned = df.dropna(axis=1, thresh=len(df)*0.7)

# 3. Fill with constant
df['column'].fillna(0, inplace=True)

# 4. Fill with statistics
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)

# 5. SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 6. KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# 7. Iterative Imputer (MICE)
iter_imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)
```

---

### 52. How do you detect and remove outliers?

```python
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

# 1. IQR Method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['column'] >= Q1-1.5*IQR) & (df['column'] <= Q3+1.5*IQR)]

# 2. Z-score Method
z_scores = np.abs(stats.zscore(df['column']))
df_clean = df[z_scores < 3]

# 3. Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(df[numeric_columns])
df_clean = df[outliers == 1]

# 4. Capping (Winsorization)
lower = df['column'].quantile(0.01)
upper = df['column'].quantile(0.99)
df['column'] = df['column'].clip(lower, upper)
```

---

### 53. How do you perform feature scaling? Provide code for StandardScaler and MinMaxScaler.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. StandardScaler (Z-score: mean=0, std=1)
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)

# 2. MinMaxScaler (Range [0,1])
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

# 3. RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(scaler_robust.fit_transform(df), columns=df.columns)

# In pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

---


### 54. How do you encode categorical variables (LabelEncoder, OneHotEncoder, TargetEncoder)?

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# 1. Label Encoding (Ordinal)
label_encoder = LabelEncoder()
df['city_encoded'] = label_encoder.fit_transform(df['city'])

# 2. One-Hot Encoding (Nominal)
df_onehot = pd.get_dummies(df, columns=['city', 'color'], drop_first=True)

# Using sklearn
ohe = OneHotEncoder(sparse=False, drop='first')
city_encoded = ohe.fit_transform(df[['city']])

# 3. Target Encoding (Mean encoding)
target_encoder = TargetEncoder()
df['city_target'] = target_encoder.fit_transform(df['city'], df['price'])

# 4. Frequency Encoding
df['city_freq'] = df['city'].map(df['city'].value_counts())

# In Pipeline
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['price']),
        ('cat', OneHotEncoder(drop='first'), ['city', 'color'])
    ])
```

---

### 55. What is the difference between Label Encoding and One-Hot Encoding?

**Label Encoding:** Converts categories to integers (red=0, blue=1, green=2)
**One-Hot Encoding:** Creates binary columns for each category

| Aspect | Label Encoding | One-Hot Encoding |
|--------|---------------|------------------|
| **Output** | Single column with integers | Multiple binary columns |
| **Use Case** | Ordinal data (low/medium/high) | Nominal data (colors, cities) |
| **Problem** | Implies order/magnitude | No implied order |
| **Memory** | Efficient | More memory needed |
| **Models** | Tree-based (handle well) | Linear models (need this) |

```python
# Label Encoding: [red, blue, green] → [0, 1, 2]
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['color'])

# One-Hot Encoding: color_red, color_blue, color_green
df_onehot = pd.get_dummies(df, columns=['color'])
```

---

### 56. How to split data into train, validation, and test sets with sklearn?

```python
from sklearn.model_selection import train_test_split

# Method 1: Two-step split (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Method 2: Function for cleaner split
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    test_ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=test_ratio, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Method 3: Stratified split (for imbalanced datasets)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Method 4: Time-series split (no shuffling)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
```

---
### 57. How do you handle imbalanced datasets? Show SMOTE / class weighting usage.

```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. Class Weights
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)

# 2. SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_smote)}")

# 3. ADASYN
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

# 4. Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)

# 5. Combination: SMOTE + Tomek
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_combined, y_combined = smote_tomek.fit_resample(X_train, y_train)

# 6. Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)

# Always use appropriate metrics (not accuracy!)
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
```

---

### 58. How do you check feature importance for tree-based models?

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
from sklearn.inspection import permutation_importance

# 1. Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_imp_df.head(10))

# 2. XGBoost Feature Importance
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)

# 3. Permutation Importance (model-agnostic)
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)

# 4. SHAP Values (most accurate)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# 5. Feature importance from coefficients (linear models)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
```

---

### 59. How do you tune hyperparameters using GridSearchCV and RandomizedSearchCV?

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# 1. GridSearchCV (exhaustive search)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_

# 2. RandomizedSearchCV (faster, random sampling)
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,  # Number of random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)

# 3. XGBoost hyperparameter tuning
import xgboost as xgb

param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42)
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, n_jobs=-1)
grid_xgb.fit(X_train, y_train)

# 4. View all results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)
print(results_df[['params', 'mean_test_score', 'std_test_score']].head())
```

---
### 60. How do you implement cross-validation manually?

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Manual K-Fold Cross-Validation
def manual_cross_validation(X, y, model, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
        
        print(f"Fold {fold+1}: {score:.4f}")
    
    print(f"\nMean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores

# Stratified K-Fold (for classification)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# Time Series Split (for time-series data)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

---

### 61. How do you detect data leakage? Give examples.

**Data Leakage:** When training data contains information about the target that wouldn't be available at prediction time.

**Common Examples:**

```python
# ❌ WRONG: Scaling before split (leakage!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage: test data info leaks to train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ✅ CORRECT: Scale after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, no fit!

# ❌ WRONG: Feature selection on all data
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Leakage!
X_train, X_test = train_test_split(X_selected)

# ✅ CORRECT: Feature selection only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# ❌ WRONG: Using future information in time-series
df['target'] = df['value'].shift(-1)  # Using future value!

# ✅ CORRECT: Only use past information
df['lag_1'] = df['value'].shift(1)  # Previous value

# Detection methods:
# 1. Check for suspiciously high accuracy
# 2. Feature importance - check if leaked features have high importance
# 3. Validate with fresh, unseen data
# 4. Check correlation between features and target

# Example: Detect features highly correlated with target
correlations = df.corr()['target'].abs().sort_values(ascending=False)
print("Features highly correlated with target:")
print(correlations[correlations > 0.9])
```

---

### 62. How do you implement early stopping in ML/DL?

```python
# 1. XGBoost with early stopping
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    early_stopping_rounds=10,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print(f"Best iteration: {xgb_model.best_iteration}")

# 2. Keras/TensorFlow early stopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# 3. PyTorch early stopping (manual)
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage in training loop
early_stopping = EarlyStopping(patience=10)
for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# 4. Scikit-learn with partial_fit (manual)
best_score = 0
patience = 10
no_improvement = 0

for iteration in range(1000):
    model.partial_fit(X_train, y_train)
    score = model.score(X_val, y_val)
    
    if score > best_score:
        best_score = score
        no_improvement = 0
    else:
        no_improvement += 1
    
    if no_improvement >= patience:
        print(f"Early stopping at iteration {iteration}")
        break
```

---
### 63. How do you evaluate a model using confusion matrix?
**Answer:** Build the confusion matrix (TP/FP/FN/TN) and derive metrics like precision, recall, F1.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
print(classification_report(y_test, y_pred))
```

---

### 64. How do you choose evaluation metrics for unbalanced datasets?
**Answer:** Don’t rely on accuracy. Use:
- **Recall** (missing positives is costly), **Precision** (false alarms are costly), **F1** (balance)
- **PR-AUC** (usually best for heavy imbalance), **ROC-AUC** (threshold-free)
- **Balanced accuracy** / **MCC** for a single robust score

---

### 65. How do you save and load a trained model using joblib/pickle?
**Answer:** Save the full **pipeline** (preprocessing + model). `joblib` is standard for scikit-learn.

```python
import joblib

joblib.dump(pipeline, "model.joblib")
pipeline = joblib.load("model.joblib")
y_pred = pipeline.predict(X_test)
```

---

### 66. How do you monitor model drift in production?
**Answer:** Track:
- **Data drift:** input distributions change (PSI/KS tests, feature stats)
- **Performance drift:** model metrics over time once labels arrive
- **Concept drift:** $X \to y$ relationship changes (metrics drop even if inputs look similar)

Set alerts, keep dashboards, and retrain on schedule or when drift triggers.

---

### 67. How do you select top features using SelectKBest?
**Answer:** Select features using training data only (best inside a pipeline to avoid leakage).

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=20)),
    ("model", LogisticRegression(max_iter=2000))
])
pipe.fit(X_train, y_train)
```

---

### 68. How do you perform PCA with sklearn?
**Answer:** Standardize → fit PCA on train → transform train/test.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42))
])
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.transform(X_test)
```

---

### 69. How do you detect multicollinearity? Show VIF calculation.
**Answer:** Use correlation + **VIF**. Rule of thumb: VIF > 5–10 is a red flag.

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

Xv = df[feature_cols].dropna()
vif = pd.DataFrame({
    "feature": Xv.columns,
    "VIF": [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]
}).sort_values("VIF", ascending=False)
print(vif)
```

---

### 70. How do you build a logistic regression model end-to-end in Python?
**Answer:** Split → preprocess → train → evaluate (pipeline keeps it clean).

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
pipe.fit(X_train, y_train)
print("F1:", f1_score(y_test, pipe.predict(X_test)))
```

---

### 71. How do you build a random forest model?
**Answer:** Train a `RandomForestClassifier/Regressor` and evaluate on a holdout set.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

---

### 72. How do you tune a random forest for optimal performance?
**Answer:** Use cross-validation and tune key params: `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`.

---

### 73. How do you implement XGBoost for classification?
**Answer:** Train `XGBClassifier` and use early stopping on a validation set.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred = xgb.predict(X_test)
```

---

### 74. How do you prevent overfitting in XGBoost?
**Answer:** Use:
- **Early stopping**, smaller `learning_rate`
- Control complexity: `max_depth`, `min_child_weight`
- Randomness: `subsample`, `colsample_bytree`
- Regularization: `reg_alpha` (L1), `reg_lambda` (L2)

---

### 75. How do you perform feature selection using SHAP values?
**Answer:** Compute mean absolute SHAP importance and keep top-$k$ features.

```python
import numpy as np
import shap

explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X_train)
importance = np.abs(shap_vals).mean(axis=0)
top_features = X_train.columns[np.argsort(importance)[::-1][:20]]
print(list(top_features))
```

---

### 76. How do you debug a model that is overfitting badly?
**Answer:**
- Check **data leakage** and split issues
- Compare train vs validation metrics and learning curves
- Reduce complexity / add regularization / early stop
- Improve data quality or add more data

---

### 77. How do you handle categorical features with more than 100 levels?
**Answer:** Avoid huge one-hot. Use:
- **Target encoding** (with CV to prevent leakage)
- **Frequency encoding** / group rare labels as “Other”
- **Hashing trick** or **CatBoost** for strong categorical handling

---

### 78. How do you preprocess text data for ML?
**Answer:** Clean (lowercase, remove URLs/punctuation), tokenize, optionally remove stopwords/lemmatize, then vectorize (TF-IDF) or use embeddings.

---

### 79. How do you convert text to tf-idf vectors?
**Answer:** Fit TF-IDF on train text and transform test text.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(train_text)
X_test_vec = tfidf.transform(test_text)
```

---

### 80. How do you deploy a scikit-learn model using FastAPI?
**Answer:** Save the pipeline, load it once in FastAPI, expose `/predict`.

```python
# app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.joblib")

class Item(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(item: Item):
    pred = model.predict([item.features])[0]
    return {"prediction": int(pred)}
```

---

### 81. How do you create an ML pipeline using sklearn Pipeline?
**Answer:** Use `ColumnTransformer` + `Pipeline` so preprocessing and training are consistent.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000))])
pipe.fit(X_train, y_train)
```

---

### 82. How do you implement k-means clustering step by step?
**Answer:** Scale → fit KMeans → get labels and centroids.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
```

---

### 83. How do you choose the optimal number of clusters?
**Answer:** Use **elbow** (inertia vs k), **silhouette score**, and check cluster stability + interpretability.

---

### 84. How do you evaluate clustering performance without labels?
**Answer:** Use internal metrics: **silhouette** (higher better), **Davies–Bouldin** (lower), **Calinski–Harabasz** (higher), plus sanity checks on cluster sizes.

---

### 85. How do you handle time-series data?
**Answer:** Keep time order, create lag/rolling/seasonality features, and evaluate using time-based splits (not random shuffles).

---

### 86. How do you split time-series data without leakage?
**Answer:** Use chronological split or walk-forward CV (no shuffling).

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for tr_idx, te_idx in tscv.split(X):
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
```

---

### 87. How do you implement ARIMA or SARIMA models?
**Answer:** Use `SARIMAX` in `statsmodels` (SARIMA = ARIMA + seasonality).

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = model.fit(disp=False)
forecast = res.forecast(steps=12)
```

---

### 88. How do you detect seasonality and trend in time-series?
**Answer:** Use decomposition (e.g., STL), check ACF/PACF peaks at seasonal lags, and visualize by day/week/month.

---

### 89. How do you generate lag features in a time-series dataset?
**Answer:** Use `shift()` for lags and `rolling()` for window statistics.

```python
df = df.sort_values("date")
df["lag_1"] = df["y"].shift(1)
df["lag_7"] = df["y"].shift(7)
df["roll_mean_7"] = df["y"].rolling(7).mean()
```

---

### 90. How do you implement rolling windows in pandas?
**Answer:** Use `rolling()` for moving stats and `expanding()` for growing windows.

```python
df["ma_30"] = df["y"].rolling(window=30, min_periods=1).mean()
df["std_30"] = df["y"].rolling(window=30, min_periods=1).std()
```

---

### 91. How do you build a neural network in TensorFlow/Keras?
**Answer:** Define the model → compile → fit with validation.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
```

---

### 92. How do you build a neural network in PyTorch?
**Answer:** Define `nn.Module`, choose loss/optimizer, and run a training loop.

```python
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(X_train_t.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for _ in range(10):
    logits = net(X_train_t).squeeze(1)
    loss = loss_fn(logits, y_train_t.float())
    opt.zero_grad(); loss.backward(); opt.step()
```

---

### 93. How do you implement dropout, batch normalization?
**Answer:** Dropout reduces overfitting; BatchNorm stabilizes training.

```python
# Keras example
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

---

### 94. How do you use callbacks such as EarlyStopping and ModelCheckpoint?
**Answer:** Stop when validation stops improving and save the best checkpoint.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best.keras", monitor="val_loss", save_best_only=True),
]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks)
```

---

### 95. How do you visualize learning curves?
**Answer:** Plot train vs validation performance as data size increases (or loss vs epochs for DL).

---

### 96. How do you handle exploding/vanishing gradients?
**Answer:**
- Exploding: **gradient clipping**, smaller learning rate
- Vanishing: **ReLU/GeLU**, good init, **BatchNorm**, **residual connections**
- For sequences: prefer **LSTM/GRU/Transformers** over vanilla RNNs

---

### 97. How do you load, clean, and preprocess image datasets?
**Answer:** Use dataset loaders and transforms: resize, normalize, and augment.

```python
from torchvision import datasets, transforms

t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder("data/images", transform=t)
```

---

### 98. How do you fine-tune a pretrained model (Transfer Learning)?
**Answer:** Freeze backbone → train new head → unfreeze some layers and fine-tune with a lower LR.

---

### 99. How do you serve a DL model with a REST API?
**Answer:** Load model once on startup, preprocess input, run inference, return JSON (FastAPI is common). Use batching and `no_grad()` in PyTorch.

---

### 100. How do you track experiments with MLflow / Weights & Biases?
**Answer:** Log params, metrics, artifacts, and the model so runs are reproducible.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "random_forest")
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(pipe, "model")
```


## 🚀 Part 3: Advanced ML Topics (Questions 101-150)

### A. Advanced Machine Learning (101–115)

### 101. Explain the mathematical intuition behind self-attention.

**Self-attention** allows each token to "look at" all other tokens and decide which ones are most relevant to understanding it.

**Mathematical Flow:**
1. Input embeddings: $X \in \mathbb{R}^{n \times d}$ (n tokens, d dimensions)
2. Create **Query (Q)**, **Key (K)**, **Value (V)** matrices via learned projections:
   - $Q = XW_Q$, $K = XW_K$, $V = XW_V$
3. Compute **attention scores** (dot product similarity):
   - $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
4. The softmax creates weights showing how much each token should attend to others
5. Multiply these weights by V to get context-aware representations

**Intuition:**
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What information do I provide?"
- Scaling by $\sqrt{d_k}$ prevents softmax saturation in high dimensions

```python
import torch
import torch.nn.functional as F

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q  # [n, d_k]
    K = X @ W_k  # [n, d_k]
    V = X @ W_v  # [n, d_v]
    
    scores = Q @ K.T / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)  # [n, n]
    output = attention_weights @ V  # [n, d_v]
    return output, attention_weights
```

---

### 102. What problem do Transformers solve that RNNs and LSTMs could not?

**Key Problems Solved:**

| **Problem** | **RNN/LSTM Issue** | **Transformer Solution** |
|-------------|-------------------|-------------------------|
| **Sequential Processing** | Must process tokens one-by-one (sequential dependency) | Parallel processing of all tokens |
| **Long-range Dependencies** | Information degrades over long sequences (vanishing gradients) | Direct attention to any token regardless of distance |
| **Training Speed** | Slow due to sequential nature | Massively parallelizable → 10-100x faster |
| **Computational Complexity** | $O(n)$ sequential steps | $O(1)$ layers (but $O(n^2)$ attention) |

**Concrete Example:**
- Sentence: "The animal didn't cross the street because **it** was too tired"
- RNN: Must pass info about "animal" through many steps to reach "it"
- Transformer: "it" directly attends to "animal" in one step

**Trade-off:** Transformers have $O(n^2)$ memory/compute for attention (quadratic in sequence length), but this is offset by parallelization benefits.

---

### 103. How does multi-head attention improve model performance?

**Multi-head attention** runs multiple attention mechanisms in parallel, each learning different relationships.

**Why It's Better:**
1. **Different representation subspaces**: Each head can focus on different aspects
   - Head 1: Syntax (subject-verb agreement)
   - Head 2: Semantic similarity
   - Head 3: Positional relationships
2. **Richer representations**: Captures diverse patterns simultaneously
3. **Ensemble effect**: Multiple "views" improve robustness

**Mathematical Form:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters:**
- Instead of one large attention with $d_{model}$ dimensions
- Use $h$ heads, each with $d_k = d_{model}/h$ dimensions
- Same total parameters but better expressiveness

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Project and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(context)
```

---

### 104. What are positional encodings and why are they required?

**Problem:** Self-attention is **permutation invariant** → it doesn't know token order!
- "cat chased mouse" vs "mouse chased cat" would be identical without position info

**Solution:** Add positional information to embeddings so the model knows token positions.

**Two Main Approaches:**

**1. Sinusoidal Positional Encoding (Original Transformer)**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Advantages:**
- Can extrapolate to longer sequences than training
- Deterministic (no learning needed)
- Different frequencies encode relative positions

**2. Learned Positional Embeddings**
- Simply learn an embedding for each position (like word embeddings)
- Used in BERT, GPT-2

```python
import torch
import torch.nn as nn
import math

# Sinusoidal positional encoding
def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Learned positional embeddings
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)
```

**Modern Variants:**
- **RoPE** (Rotary Positional Embeddings): Used in LLaMA, better for long sequences
- **ALiBi** (Attention with Linear Biases): No explicit embeddings, just bias attention scores

---

### 105. Explain LayerNorm vs BatchNorm.

| **Aspect** | **BatchNorm** | **LayerNorm** |
|------------|--------------|--------------|
| **Normalizes over** | Batch dimension (across samples) | Feature dimension (within each sample) |
| **Formula** | $\frac{x - \mu_{batch}}{\sigma_{batch}}$ | $\frac{x - \mu_{layer}}{\sigma_{layer}}$ |
| **Use Case** | CNNs (images) | Transformers, RNNs (sequences) |
| **Batch Dependency** | Requires large batch size | Works with batch size = 1 |
| **Training vs Inference** | Different behavior (uses running stats at inference) | Identical behavior |
| **Position in Network** | After convolution, before activation | After self-attention or FFN |

**Why Transformers Use LayerNorm:**
1. **Sequence length variability**: Batch samples may have different lengths → BatchNorm doesn't work well
2. **Stable with small batches**: Training effectiveness doesn't depend on batch size
3. **Per-sample normalization**: Each token sequence normalized independently

```python
import torch
import torch.nn as nn

# BatchNorm (normalizes across batch for each feature)
batch_norm = nn.BatchNorm1d(num_features=512)
x = torch.randn(32, 512, 100)  # [batch, features, seq_len]
x = batch_norm(x)  # Norm over batch dimension

# LayerNorm (normalizes across features for each sample)
layer_norm = nn.LayerNorm(normalized_shape=512)
x = torch.randn(32, 100, 512)  # [batch, seq_len, features]
x = layer_norm(x)  # Norm over feature dimension

# Manual LayerNorm
def layer_norm_manual(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)
```

**Interview Tip:** Transformers use LayerNorm because it's stable for variable-length sequences and doesn't depend on batch statistics.

---

### 106. What is the difference between fine-tuning and feature extraction?

| **Aspect** | **Feature Extraction** | **Fine-tuning** |
|------------|----------------------|----------------|
| **Pretrained Weights** | Frozen (no updates) | Unfrozen (updated) |
| **Training** | Only train new head/classifier | Train entire model (or partial layers) |
| **Compute** | Fast, low memory | Slower, more memory |
| **Data Required** | Works with small datasets | Needs more data for best results |
| **Risk** | Lower risk of overfitting | Can overfit on small data |
| **Use Case** | Similar task to pretraining | Task differs significantly |

**Feature Extraction:**
```python
import torch
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 classes

# Only train the new classifier
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**Fine-tuning:**
```python
# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Replace final layer
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Train entire model (or use differential learning rates)
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Last conv block
    {'params': model.fc.parameters(), 'lr': 1e-3}        # Classifier
])
```

**Best Practice (Gradual Unfreezing):**
1. Start with feature extraction (train only head)
2. Unfreeze last few layers and fine-tune with low LR
3. Optionally unfreeze entire model with very low LR

---

### 107. How does model quantization reduce inference cost?

**Quantization** converts high-precision weights (FP32) to low-precision (INT8, INT4) → smaller model, faster inference.

**Benefits:**
- **4x smaller model size** (FP32 → INT8)
- **2-4x faster inference** (integer operations are faster)
- **Lower memory bandwidth** (critical for edge devices)
- **Energy efficient** (important for mobile/IoT)

**Types of Quantization:**

| **Type** | **When Applied** | **Accuracy** | **Speed** |
|----------|----------------|--------------|-----------|
| **Post-Training Quantization** | After training | Good | Fast to apply |
| **Quantization-Aware Training (QAT)** | During training | Best | Slower to train |

**Example: PyTorch Quantization**
```python
import torch
import torch.quantization as quant

# 1. Post-Training Static Quantization
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Prepare model for quantization
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# Calibrate with representative data
with torch.no_grad():
    for batch in calibration_data:
        model_prepared(batch)

# Convert to quantized model
model_quantized = quant.convert(model_prepared)

# 2. Dynamic Quantization (for LSTM/Transformer)
model_dynamic_quant = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 3. Check size reduction
print(f"Original model: {os.path.getsize('model.pth') / 1e6:.2f} MB")
print(f"Quantized model: {os.path.getsize('model_quant.pth') / 1e6:.2f} MB")
```

**INT8 vs INT4:**
- **INT8**: 1-2% accuracy drop, widely supported
- **INT4**: 4-8x compression, 3-5% accuracy drop, requires specialized hardware

**Interview Tip:** Quantization is essential for deploying LLMs on edge devices (e.g., LLaMA 7B in FP16 is 14GB, INT4 is ~4GB).

---

### 108. What is knowledge distillation? How is the student model trained?

**Knowledge Distillation:** Train a small **student** model to mimic a large **teacher** model's behavior.

**Why It Works:**
- Teacher's "soft targets" (probability distributions) contain more information than hard labels
- Student learns from teacher's confidence and reasoning patterns

**Training Process:**

1. **Teacher Model**: Large, accurate pretrained model (frozen)
2. **Student Model**: Smaller model to be trained
3. **Distillation Loss**:
   $$L_{distill} = \alpha \cdot L_{CE}(y, y_{student}) + (1-\alpha) \cdot L_{KL}(y_{teacher}, y_{student})$$
   - $L_{CE}$: Standard cross-entropy with true labels
   - $L_{KL}$: KL divergence between teacher and student predictions
   - $\alpha$: Balance between label and teacher knowledge

4. **Temperature Scaling**: Soften probabilities for better knowledge transfer
   $$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$
   - Higher $T$ → softer probabilities
   - Typical $T = 2$ to $5$ during distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss (with true labels)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss (with teacher predictions)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)  # Scale by T^2
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Training loop
teacher_model.eval()
student_model.train()

for batch in train_loader:
    images, labels = batch
    
    # Get teacher predictions (no grad)
    with torch.no_grad():
        teacher_logits = teacher_model(images)
    
    # Get student predictions
    student_logits = student_model(images)
    
    # Compute distillation loss
    loss = distillation_loss(student_logits, teacher_logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Types:**
- **Response-based**: Match final outputs (standard distillation)
- **Feature-based**: Match intermediate layer representations
- **Relation-based**: Match relationships between samples

**Real-world Examples:**
- **DistilBERT**: 40% smaller than BERT, 97% accuracy, 60% faster
- **TinyBERT**: 7.5x smaller, 96% accuracy

---

### 109. How does curriculum learning improve optimization?

**Curriculum Learning:** Train the model on progressively harder examples (easy → medium → hard), mimicking how humans learn.

**Why It Helps:**
1. **Faster convergence**: Model learns basic patterns first
2. **Better generalization**: Gradual difficulty prevents overfitting to hard examples
3. **Stability**: Easier start prevents early training instability
4. **Better local minima**: Smooth path through loss landscape

**Implementation Strategies:**

**1. Difficulty-based (by feature)**
```python
# Sort samples by difficulty metric
def compute_difficulty(sample):
    # Examples: Length, noise level, classification entropy
    return len(sample['text'])  # Text length as proxy

# Sort and split into curriculum stages
samples_sorted = sorted(dataset, key=compute_difficulty)
stage1 = samples_sorted[:len(samples_sorted)//3]    # Easy
stage2 = samples_sorted[len(samples_sorted)//3:2*len(samples_sorted)//3]  # Medium
stage3 = samples_sorted[2*len(samples_sorted)//3:]  # Hard

# Train sequentially
for epoch in range(10):
    train(model, stage1)
for epoch in range(10):
    train(model, stage1 + stage2)
for epoch in range(10):
    train(model, stage1 + stage2 + stage3)
```

**2. Self-paced (model determines difficulty)**
```python
def self_paced_curriculum(model, dataset, lambda_):
    losses = []
    
    # Compute loss for each sample
    for sample in dataset:
        output = model(sample['input'])
        loss = criterion(output, sample['label'])
        losses.append((loss.item(), sample))
    
    # Select easier samples (low loss)
    threshold = np.percentile([l for l, _ in losses], lambda_)
    selected_samples = [s for l, s in losses if l <= threshold]
    
    return selected_samples

# Gradually increase lambda (include more samples)
for epoch in range(100):
    lambda_ = min(100, epoch * 2)  # 0 → 100%
    subset = self_paced_curriculum(model, dataset, lambda_)
    train(model, subset)
```

**3. Transfer-based (task progression)**
```python
# Start with related but easier tasks
# Stage 1: Binary classification
train(model, binary_task)

# Stage 2: Multi-class (3 classes)
train(model, multiclass_3)

# Stage 3: Full problem (10 classes)
train(model, multiclass_10)
```

**Real-world Use Cases:**
- **NLP**: Train on short sentences first, then long ones
- **Vision**: Start with high-contrast images, add noisy ones
- **RL**: Easy game levels before hard ones
- **Machine Translation**: Short sentences → complex sentences

---

### 110. Explain gradient checkpointing and why it's used.

**Gradient Checkpointing** trades compute for memory by recomputing activations during backward pass instead of storing them.

**The Memory Problem:**
- Forward pass stores all activations for backprop
- For deep networks: Memory $\propto$ depth × batch size
- Large transformers/LLMs can't fit in GPU memory

**How It Works:**
1. **Forward pass**: Only save checkpoints at certain layers (discard intermediate activations)
2. **Backward pass**: Recompute missing activations on-the-fly when needed
3. **Trade-off**: ~20-30% slower training, but 5-10x less memory

**Example: Without vs With Checkpointing**
```python
# Without checkpointing (normal training)
# Memory: O(n) where n = number of layers
def forward_normal(x, layers):
    activations = [x]  # Store all
    for layer in layers:
        x = layer(x)
        activations.append(x)  # Memory grows!
    return x, activations

# With checkpointing
# Memory: O(sqrt(n)) with strategic checkpoints
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(x, layers):
    # Only checkpoint every k layers
    for i, layer in enumerate(layers):
        if i % checkpoint_frequency == 0:
            x = checkpoint(layer, x)  # Recompute during backward
        else:
            x = layer(x)
    return x
```

**PyTorch Implementation:**
```python
import torch
from torch.utils.checkpoint import checkpoint_sequential

# Method 1: Automatic checkpointing
model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(100)])
segments = 10  # Checkpoint every 10 layers

def forward_with_checkpointing(input):
    return checkpoint_sequential(model, segments, input)

# Method 2: Manual checkpointing
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x):
        # Checkpoint the attention and FFN blocks
        x = x + checkpoint(self.attention, x)
        x = x + checkpoint(self.ffn, x)
        return x

# Method 3: HuggingFace Transformers
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
model.gradient_checkpointing_enable()  # One line!
```

**Memory Savings Example:**
- **GPT-3 (175B params)**: Without checkpointing → ~1TB memory needed
- **With checkpointing**: ~100-200GB (fits on multi-GPU)

**When to Use:**
- Training very deep networks (>50 layers)
- Large batch sizes needed but limited GPU memory
- Training LLMs (almost always enabled)
- Fine-tuning large models on single GPU

---

### 111. What is catastrophic forgetting? How do modern models mitigate it?

**Catastrophic Forgetting:** When a neural network "forgets" previously learned tasks after learning new ones.

**Example:**
1. Train model on French → English translation (90% accuracy)
2. Train same model on German → English (85% accuracy)
3. Test on French → English again → **30% accuracy** (forgot French!)

**Why It Happens:**
- Neural networks optimize globally → new task overwrites old weights
- Unlike humans, no mechanism to protect important knowledge

**Mitigation Strategies:**

**1. Elastic Weight Consolidation (EWC)**
- Identify important weights for old task (using Fisher Information)
- Add penalty when updating those weights

```python
class EWC:
    def __init__(self, model, dataset, lambda_=1000):
        self.model = model
        self.lambda_ = lambda_
        self.fisher = self._compute_fisher(dataset)
        self.old_params = {n: p.clone() for n, p in model.named_parameters()}
    
    def _compute_fisher(self, dataset):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for batch in dataset:
            self.model.zero_grad()
            output = self.model(batch)
            loss = F.cross_entropy(output, batch['labels'])
            loss.backward()
            
            for n, p in self.model.named_parameters():
                fisher[n] += p.grad.data ** 2
        
        return {n: f / len(dataset) for n, f in fisher.items()}
    
    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            loss += (self.fisher[n] * (p - self.old_params[n]) ** 2).sum()
        return self.lambda_ * loss

# Training with EWC
ewc = EWC(model, old_task_data)
for batch in new_task_data:
    loss = criterion(model(batch), batch['labels'])
    loss = loss + ewc.penalty()  # Protect old knowledge
    loss.backward()
    optimizer.step()
```

**2. Progressive Neural Networks**
- Add new columns/modules for new tasks
- Old parameters frozen, new task uses old features via lateral connections

**3. Memory Replay (Experience Replay)**
- Store subset of old task data
- Mix old and new data during training

```python
class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, task_data):
        # Store samples from old task
        samples = random.sample(task_data, min(len(task_data), self.buffer_size))
        self.buffer.extend(samples)
        if len(self.buffer) > self.buffer_size:
            self.buffer = random.sample(self.buffer, self.buffer_size)
    
    def get_batch(self, new_batch):
        # Mix old and new data
        old_samples = random.sample(self.buffer, len(new_batch) // 2)
        return new_batch + old_samples

replay_buffer = ReplayBuffer()
replay_buffer.add(task1_data)

for batch in task2_data:
    mixed_batch = replay_buffer.get_batch(batch)
    train(model, mixed_batch)
```

**4. Parameter Isolation**
- Use **adapters** (small trainable modules) for each task
- Main model frozen, only adapters updated

**5. Knowledge Distillation**
- Keep old model as teacher
- New model must match old predictions on old tasks

**Modern LLM Approach (RLHF Context):**
- Use **PEFT methods** (LoRA, adapters) → only train small modules
- **Regularization loss** to keep close to base model
- **Multi-task training** from the start

---

### 112. How does contrastive learning work? (e.g., SimCLR, CLIP)

**Contrastive Learning:** Learn representations by pulling similar examples together and pushing dissimilar ones apart.

**Core Idea:**
- Positive pairs (similar): Same image with augmentations, image-text pairs
- Negative pairs (dissimilar): Different images/texts
- Goal: Maximize agreement for positives, minimize for negatives

**Mathematical Objective (InfoNCE Loss):**
$$L = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}$$
- $z_i, z_j$: Positive pair embeddings
- $z_k$: Negative samples
- $\tau$: Temperature parameter
- $\text{sim}$: Cosine similarity

**SimCLR (Visual Self-Supervised Learning):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder  # e.g., ResNet
        self.projection = nn.Sequential(
            nn.Linear(encoder.output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=-1)

def simclr_loss(z_i, z_j, temperature=0.5):
    """
    z_i, z_j: [batch_size, projection_dim] - positive pairs
    """
    batch_size = z_i.shape[0]
    
    # Concatenate positive pairs
    z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # [2N, 2N]
    
    # Create mask for positive pairs
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -9e15)  # Remove self-similarity
    
    # Positive pairs are at positions (i, i+N) and (i+N, i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ])
    
    # Compute loss
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=-1)
    return loss.mean()

# Training
model = SimCLR(encoder=ResNet50())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    # Apply two different augmentations to same image
    x_i = augment_1(batch)  # Crop, color jitter, blur
    x_j = augment_2(batch)  # Different augmentation
    
    z_i = model(x_i)
    z_j = model(x_j)
    
    loss = simclr_loss(z_i, z_j)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**CLIP (Image-Text Contrastive Learning):**

```python
class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder  # Vision Transformer
        self.text_encoder = text_encoder    # Transformer
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def forward(self, images, texts):
        # Encode and normalize
        image_features = F.normalize(self.image_encoder(images), dim=-1)
        text_features = F.normalize(self.text_encoder(texts), dim=-1)
        
        # Compute similarity matrix
        logits = image_features @ text_features.T / self.temperature
        
        # Symmetric loss (image→text and text→image)
        labels = torch.arange(len(images), device=images.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2

# Zero-shot classification using CLIP
def zero_shot_classify(model, image, class_names):
    # Create text prompts
    prompts = [f"A photo of a {c}" for c in class_names]
    
    with torch.no_grad():
        image_features = model.image_encoder(image)
        text_features = torch.stack([model.text_encoder(p) for p in prompts])
        
        # Compute similarities
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    
    return class_names[similarity.argmax()]
```

**Why It Works:**
1. **No labels needed**: Self-supervised (create positives via augmentation)
2. **Rich representations**: Must learn invariances and semantics
3. **Transfer learning**: Representations work well for downstream tasks

**Key Applications:**
- **SimCLR**: Image classification with 10% labeled data achieves 90% supervised performance
- **CLIP**: Zero-shot classification, image search, multimodal understanding

---

### 113. Why is cross-entropy widely used for classification?

**Cross-entropy** measures the difference between predicted probability distribution and true distribution.

**Mathematical Form:**
$$\text{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- $y_i$: True label (one-hot: $[0,0,1,0]$)
- $\hat{y}_i$: Predicted probability
- For binary: $\text{CE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

**Why It's the Best Choice:**

**1. Probabilistic Interpretation**
- Equivalent to **maximum likelihood estimation** (MLE)
- Minimizing CE = maximizing likelihood of correct class

**2. Relationship to KL Divergence**
$$\text{CE}(p, q) = H(p) + D_{KL}(p \| q)$$
- $H(p)$: Entropy of true distribution (constant)
- Minimizing CE = minimizing KL divergence

**3. Gradient Properties**
- Output gradient: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$
- Simple, well-behaved gradients (no saturation like MSE)

**4. Handles Class Probabilities Naturally**
- Outputs are probabilities (via softmax)
- Penalizes confident wrong predictions heavily

**Comparison with Alternatives:**

| **Loss** | **Formula** | **Use Case** | **Issue** |
|----------|------------|-------------|-----------|
| **Cross-Entropy** | $-\sum y_i \log(\hat{y}_i)$ | Classification (standard) | Perfect ✓ |
| **MSE** | $(y - \hat{y})^2$ | Regression | Wrong for classification (poor gradients) |
| **Hinge** | $\max(0, 1-y\hat{y})$ | SVM | Not probabilistic |
| **Focal Loss** | $-(1-\hat{y})^\gamma \log(\hat{y})$ | Imbalanced data | Specialized |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary Cross-Entropy
bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid
logits = model(x)  # Raw scores
loss = bce_loss(logits, labels.float())

# Multi-class Cross-Entropy
ce_loss = nn.CrossEntropyLoss()  # Includes softmax
logits = model(x)  # [batch_size, num_classes]
loss = ce_loss(logits, labels)  # labels are class indices

# Manual implementation
def cross_entropy_manual(logits, labels):
    # Apply softmax
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of true class
    true_class_probs = probs[torch.arange(len(labels)), labels]
    
    # Compute negative log likelihood
    return -torch.log(true_class_probs + 1e-10).mean()

# With label smoothing (regularization)
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = ce_smooth(logits, labels)
```

**Interview Tip:** Cross-entropy is optimal for classification because it directly optimizes the likelihood of the correct class and has clean gradients that avoid saturation.

---

### 114. Explain the differences between Adam, AdamW, and RMSProp.

**Optimizers** adapt learning rates per parameter for faster convergence.

| **Optimizer** | **Key Feature** | **Best For** | **Issue** |
|--------------|----------------|-------------|-----------|
| **SGD** | Constant LR | Simple problems, with momentum | Slow, requires LR tuning |
| **RMSProp** | Adaptive LR per parameter | RNNs, non-stationary problems | Can decay LR too fast |
| **Adam** | RMSProp + momentum | General purpose (most popular) | Poor generalization sometimes |
| **AdamW** | Adam + decoupled weight decay | Transformers, LLMs (SOTA) | Slightly slower |

**RMSProp (Root Mean Square Propagation):**
$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$
- Maintains moving average of squared gradients
- Adapts LR per parameter (large gradients → smaller steps)
- $\beta \approx 0.9$, $\eta = 0.001$

**Adam (Adaptive Moment Estimation):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(RMSProp)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

- Combines momentum + adaptive LR
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$

**AdamW (Adam with Decoupled Weight Decay):**
- **Problem with Adam**: Weight decay coupled with gradient, affecting adaptive LR
- **AdamW**: Separates weight decay from gradient update

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
- $\lambda$: Weight decay coefficient (typically 0.01)

```python
import torch
import torch.optim as optim

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (better for transformers)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch), labels)
    loss.backward()
    optimizer.step()

# With learning rate scheduling
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
for epoch in range(100):
    train(model, optimizer)
    scheduler.step()
```

**When to Use Each:**

```python
# Computer Vision (CNNs)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# RNNs / Older architectures
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# General ML / Small networks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transformers / LLMs (BEST)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```

**Modern Variants:**
- **Lion**: More memory-efficient than AdamW
- **Adafactor**: Memory-efficient for very large models
- **LAMB**: Large batch training (BERT)

**Interview Tip:** For transformers and LLMs, always use **AdamW** with proper weight decay. For CNNs, SGD with momentum often generalizes better.

---

### 115. What is a mixture-of-experts model, and why is MoE efficient?

**Mixture-of-Experts (MoE):** A model architecture with multiple specialized "expert" networks and a gating network that routes inputs to relevant experts.

**Architecture:**
1. **Experts**: $N$ independent neural networks (e.g., FFN layers)
2. **Gating Network**: Learns to route each input to top-$k$ experts
3. **Output**: Weighted combination of selected experts

**Mathematical Form:**
$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$
- $E_i$: Expert $i$'s output
- $G(x)_i$: Gating weight for expert $i$ (via softmax)
- In practice, only top-$k$ experts activated (sparse routing)

**Why MoE is Efficient:**

| **Aspect** | **Dense Model** | **MoE Model** |
|------------|----------------|--------------|
| **Parameters** | All used per token | Only ~1-2 experts per token |
| **Compute** | $O(d_{model} \times d_{ff})$ | $O(d_{model} \times d_{ff} / N)$ per token |
| **Model Size** | 7B params → 7B active | 100B params → 7B active (sparsity) |
| **Scaling** | Linear cost increase | Sub-linear (conditional compute) |

**Efficiency Gains:**
- **10-100x more parameters** with similar compute
- **Specialization**: Each expert learns different patterns
- **Sparse activation**: Only relevant experts process each input

**Implementation Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Compute gating scores
        gate_logits = self.gate(x)  # [batch, num_experts]
        
        # Select top-k experts per input
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [batch, top_k]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Route to selected experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # [batch]
            gate_weight = top_k_gates[:, i].unsqueeze(-1)  # [batch, 1]
            
            # Apply corresponding expert (batched)
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += gate_weight[mask] * expert_output
        
        return output

# Load balancing loss (ensure experts used equally)
def load_balancing_loss(gate_logits, top_k_indices, num_experts):
    # Count how often each expert is selected
    expert_counts = torch.zeros(num_experts, device=gate_logits.device)
    for i in range(num_experts):
        expert_counts[i] = (top_k_indices == i).float().sum()
    
    # Penalize imbalance
    target_count = top_k_indices.numel() / num_experts
    return ((expert_counts - target_count) ** 2).mean()
```

**Real-World Examples:**

**1. Switch Transformer (Google, 2021)**
- 1.6 **trillion** parameters
- Only 30B active per token (top-1 routing)
- 7x faster than dense model with same quality

**2. GPT-4 (Rumored Architecture)**
- ~1.8T total parameters
- 8 experts, top-2 routing → ~220B active
- Enables high capacity with reasonable inference cost

**Challenges:**
1. **Load Balancing**: Some experts may be underused (need auxiliary loss)
2. **Training Complexity**: Requires careful initialization
3. **Communication Overhead**: Expert parallelism across GPUs

```python
# Training with load balancing
def train_moe(model, batch):
    output, gate_logits, top_k_indices = model(batch)
    
    # Task loss
    task_loss = criterion(output, labels)
    
    # Load balancing loss (encourage uniform expert usage)
    balance_loss = load_balancing_loss(gate_logits, top_k_indices, num_experts)
    
    # Combined loss
    total_loss = task_loss + 0.01 * balance_loss
    
    total_loss.backward()
    optimizer.step()
```

**Interview Tip:** MoE enables massive scale while keeping inference cost manageable—you get a trillion-parameter model that costs like a 100B model to run. The key is **sparse activation** via routing.

---


---

### B. Large Language Models (LLMs) & GenAI (116–135)
### 116. How does a decoder-only Transformer (GPT architecture) work?
### 117. What is the architecture difference between GPT, BERT, and T5?
### 118. How does tokenization influence LLM performance?
### 119. Explain Rotary Positional Embeddings (RoPE).
### 120. What is the KV cache? Why does it speed up inference?
### 121. What is LoRA and why is it better than full fine-tuning?
### 122. Explain QLoRA and 4-bit quantization.
### 123. What are PEFT methods? List 4 common ones.
### 124. How do you prevent hallucinations in LLMs?
### 125. What is supervised fine-tuning (SFT)? How is it different from RLHF?
### 126. What is RLHF? Explain reward modeling, training, preference data.
### 127. What are guardrails in LLM-based systems?
### 128. How do you evaluate LLMs? (MT-Bench, HELM, MMLU, BLEU, Rouge)
### 129. What is instruction tuning?
### 130. Explain the concept of “chain-of-thought prompting.”
### 131. What is retrieval-augmented generation (RAG)?
### 132. What vector databases are used in RAG?
### 133. Explain chunking strategies for RAG pipelines.
### 134. What are embedding models? How do they differ from LLMs?
### 135. Explain how LlamaIndex or LangChain orchestrate LLM pipelines.

---


---

### C. Agentic AI & Autonomous AI Systems (136–150)
### 136. What is an AI agent? How does it differ from a chatbot?
### 137. Explain deliberate reasoning vs reactive reasoning in agents.
### 138. What is ReAct? How does it combine reasoning + acting?
### 139. What is a memory module in agent architecture?
### 140. Explain tool-use in agentic systems with examples.
### 141. How do multi-agent systems collaborate?
### 142. What is the difference between single-agent and multi-agent planning?
### 143. What are agent workflows in LangChain?
### 144. How do agents decide when to call external APIs/tools?
### 145. Explain planning algorithms used in agents (MCTS, A*).
### 146. What is retrieval-augmented agents?
### 147. How do agents maintain long-term memory? (Vector DB / episodic)
### 148. How do agents handle uncertainty and tool failures?
### 149. What is task decomposition in agentic systems?
### 150. What safety risks exist in autonomous AI agents?
