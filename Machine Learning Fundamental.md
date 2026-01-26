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
### 64. How do you choose evaluation metrics for unbalanced datasets?
### 65. How do you save and load a trained model using joblib/pickle?
### 66. How do you monitor model drift in production?
### 67. How do you select top features using SelectKBest?
### 68. How do you perform PCA with sklearn?
### 69. How do you detect multicollinearity? Show VIF calculation.
### 70. How do you build a logistic regression model end-to-end in Python?
### 71. How do you build a random forest model?
### 72. How do you tune a random forest for optimal performance?
### 73. How do you implement XGBoost for classification?
### 74. How do you prevent overfitting in XGBoost?
### 75. How do you perform feature selection using SHAP values?
### 76. How do you debug a model that is overfitting badly?
### 77. How do you handle categorical features with more than 100 levels?
### 78. How do you preprocess text data for ML?
### 79. How do you convert text to tf-idf vectors?
### 80. How do you deploy a scikit-learn model using FastAPI?
### 81. How do you create an ML pipeline using sklearn Pipeline?
### 82. How do you implement k-means clustering step by step?
### 83. How do you choose the optimal number of clusters?
### 84. How do you evaluate clustering performance without labels?
### 85. How do you handle time-series data?
### 86. How do you split time-series data without leakage?
### 87. How do you implement ARIMA or SARIMA models?
### 88. How do you detect seasonality and trend in time-series?
### 89. How do you generate lag features in a time-series dataset?
### 90. How do you implement rolling windows in pandas?
### 91. How do you build a neural network in TensorFlow/Keras?
### 92. How do you build a neural network in PyTorch?
### 93. How do you implement dropout, batch normalization?
### 94. How do you use callbacks such as EarlyStopping and ModelCheckpoint?
### 95. How do you visualize learning curves?
### 96. How do you handle exploding/vanishing gradients?
### 97. How do you load, clean, and preprocess image datasets?
### 98. How do you fine-tune a pretrained model (Transfer Learning)?
### 99. How do you serve a DL model with a REST API?
### 100. How do you track experiments with MLflow / Weights & Biases?


## 🚀 Part 3: Advanced ML Topics (Questions 101-150)

### A. Advanced Machine Learning (101–115)
### 101. Explain the mathematical intuition behind self-attention.
### 102. What problem do Transformers solve that RNNs and LSTMs could not?
### 103. How does multi-head attention improve model performance?
### 104. What are positional encodings and why are they required?
### 105. Explain LayerNorm vs BatchNorm.
### 106. What is the difference between fine-tuning and feature extraction?
### 107. How does model quantization reduce inference cost?
### 108. What is knowledge distillation? How is the student model trained?
### 109. How does curriculum learning improve optimization?
### 110. Explain gradient checkpointing and why it's used.
### 111. What is catastrophic forgetting? How do modern models mitigate it?
### 112. How does contrastive learning work? (e.g., SimCLR, CLIP)
### 113. Why is cross-entropy widely used for classification?
### 114. Explain the differences between Adam, AdamW, and RMSProp.
### 115. What is a mixture-of-experts model, and why is MoE efficient?

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
