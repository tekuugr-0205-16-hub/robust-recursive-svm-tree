# Robust Recursive SVM Tree

## Project Overview

This project explores a hybrid machine learning approach that combines the recursive structure of Decision Trees with the separating capability of Support Vector Machines (SVMs).

The main objective of the project is to investigate whether replacing traditional Decision Tree splits with SVM-based splits can improve robustness when the training data contains noise and outliers.

Instead of using standard feature-threshold rules at each node, the proposed model trains a Linear Support Vector Machine at every internal node to partition the data recursively.

The project compares the proposed Hybrid SVM Tree against a traditional Decision Tree under noisy training conditions using the Breast Cancer Wisconsin dataset.

---

# Motivation

Traditional Decision Trees are simple, interpretable, and efficient. However, they are often sensitive to:

- Noisy labels
- Feature outliers
- Irregular decision boundaries
- Small variations in the training data

This can lead to unstable splits and reduced generalization performance.

The idea behind this project was to test whether margin-based partitioning from SVMs could produce more stable and robust recursive splits inside a tree architecture.

The goal was not only to improve accuracy, but also to improve robustness under corrupted training conditions.

---

# Core Idea

A traditional Decision Tree creates splits using rules such as:

```python
feature <= threshold
```

In this project, every internal node trains a:

```python
LinearSVC()
```

The SVM then decides how to separate the data into two partitions, and the tree continues recursively from there.

This creates a recursive tree-like structure while using SVM decision boundaries instead of standard threshold-based splitting.

---

# Dataset

The project uses the Breast Cancer Wisconsin dataset provided by Scikit-learn.

The dataset is a binary classification problem used to classify tumors as:

- Malignant
- Benign

Dataset Information:
- 569 samples
- 30 numerical features
- Binary classification task

The dataset was selected because it is well-structured and suitable for evaluating classification robustness under noisy conditions.

---

# Data Preprocessing

Several preprocessing steps were applied before training the models.

## Feature Standardization

The features were standardized using:

```python
StandardScaler()
```

This is especially important because SVMs are sensitive to feature scaling.

---

## PCA Visualization

Principal Component Analysis (PCA) was used to project the dataset into two dimensions for visualization purposes.

This helps visualize:
- class distribution
- decision boundaries
- data separation

The PCA projection was only used for visualization and not for the main training process.

---

# Noise Injection

One of the main focuses of the project is robustness evaluation.

To simulate imperfect real-world datasets, artificial noise was intentionally added to the training data.

## Label Noise

A percentage of class labels were randomly flipped.

Configuration used:

```text
15% label noise
```

This simulates situations where data labels are incorrect or unreliable.

---

## Feature Outliers

Random outliers were injected into feature values.

Configuration used:

```text
10% feature outliers
```

This simulates corrupted or abnormal data samples.

---

# Models Used

## Baseline Model

A traditional Decision Tree Classifier was used as the baseline model.

Configuration:
- max_depth = 4
- min_samples_split = 8

This provides a fair comparison against the proposed hybrid model.

---

## Proposed Hybrid SVM Tree

The custom Hybrid SVM Tree uses:

- recursive tree structure
- LinearSVC at each internal node
- recursive binary partitioning
- confidence-based leaf predictions

The model dynamically trains an SVM at every split and recursively partitions the dataset based on SVM predictions.

---

# Robustness Evaluation

To evaluate robustness, both models were tested across multiple noise levels using Stratified K-Fold Cross Validation.

Noise levels tested:

```text
0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%
```

The project measures how performance degrades as the dataset becomes increasingly noisy.

---

# Evaluation Metrics

The models were evaluated using several metrics:

- Accuracy
- ROC/AUC Score
- Confusion Matrix
- Classification Report
- Cross Validation Accuracy

These metrics help compare:
- classification performance
- robustness stability
- resistance to noisy conditions

---

# Visualizations

The project generates several visual outputs to analyze model behavior.

## Decision Boundary Comparison

Compares the decision regions of:
- traditional Decision Tree
- Hybrid SVM Tree

This helps visualize how both models separate the classes.

Generated File:
```text
02_decision_boundaries.png
```

---

## Robustness Curves

Shows how model accuracy changes as noise levels increase.

Generated File:
```text
03_robustness_curves.png
```

---

## Evaluation Dashboard

Contains:
- confusion matrices
- ROC curve comparison
- metric comparison charts

Generated File:
```text
04_evaluation_dashboard.png
```

---

# Results

The Hybrid SVM Tree demonstrated more stable behavior under noisy conditions compared to the baseline Decision Tree.

Key observations:
- better resistance to noisy labels
- smoother decision boundaries
- improved robustness against outliers
- competitive ROC/AUC performance
- slower performance degradation as noise increased

The results suggest that combining recursive tree structures with margin-based partitioning can improve robustness in certain classification tasks.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Structure

```text
robust-recursive-svm-tree/
│
├── Robust_Recursive_SVM_Tree1.ipynb
├── README.md
├── requirements.txt
├── 02_decision_boundaries.png
├── 03_robustness_curves.png
└── 04_evaluation_dashboard.png
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/robust-recursive-svm-tree.git
```

Move into the project directory:

```bash
cd robust-recursive-svm-tree
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the notebook in Google Colab or Jupyter Notebook.

You can also run the Python script version directly:

```bash
python robust_recursive_svm_tree.py
```

---

# Future Improvements

Possible future improvements include:

- Kernel SVM node partitioning
- Multi-class classification support
- Tree pruning strategies
- Hyperparameter optimization
- Comparison with Random Forest and XGBoost
- GPU acceleration
- Better probability estimation methods

---

# Conclusion

This project was developed as an experimental research-oriented machine learning project focused on robustness under noisy data conditions.

The work explores how combining:
- recursive tree structures
with
- margin-maximizing classifiers

can produce more stable learning behavior in corrupted environments.

The project emphasizes not only classification accuracy, but also model stability and robustness under difficult training conditions.
