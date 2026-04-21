# Overview
This project builds a multi-label classification model to identify communication functions in doctors’ responses from online health consultations.
The model uses TF-IDF for text representation and Logistic Regression with a One-vs-Rest approach. A semi-supervised learning method (pseudo-labeling) is also explored to evaluate whether unlabeled data can improve model performance.

# Dataset
This project uses the Doctor's Answer Text Dataset in Indonesian Contains Information on Medical Interview Patterns, available on Mendeley Data:
https://data.mendeley.com/datasets/p8d5bynh3m/1
The dataset contains Indonesian health consultation texts collected from an online consultation platform. It consists of:
- Labeled data (500 samples)
  Doctor responses annotated with six communication functions:
  -  1-FR (Relationship Building)
  -  2-GI (Gathering Information)
  -  3-PI (Providing Information)
  -  4-DM (Decision Making)
  -  5-EDTRB (Enabling Behaviour)
  -  6-RE (Responding to Emotion)
- Unlabeled data (~497,000 samples)
  Used for prediction and semi-supervised experiments.

Note:
The dataset is not included in this repository due to its large size. Please download it directly from the source link above.

# To Improve

## 1. Class Imbalance
The current dataset is heavily imbalanced, labels like `2-GI` (37 samples) and `6-RE` (21 samples) are severely underrepresented compared to `1-FR` and `3-PI` (400 samples each). This causes the model to predict almost nothing for rare classes (e.g., `6-RE` scores F1 = 0.00).

- Apply oversampling techniques such as SMOTE on the TF-IDF feature space
- Use cost-sensitive learning with stronger per-class weights in Logistic Regression
- Collect or manually annotate more samples specifically for underrepresented labels

## 2. Text Preprocessing
The current `text_clean_pipeline` is minimal (lowercasing, removing special characters).

- Add stopword removal using a comprehensive Indonesian stopword list
- Use a proper Indonesian stemmer (PySastrawi) consistently across both labeled and unlabeled data, as the labeled set already has a `Process_Data` column with stemmed text that is not being used for training
- Evaluate whether using `Process_Data` / `tokens_stemmed` columns instead of `Text_Clean` improves results

## 3. Hyperparameter Tuning
The model currently uses default/fixed hyperparameters.

- Run cross-validated grid search over TF-IDF parameters (`max_features`, `ngram_range`, `min_df`, `max_df`) and Logistic Regression parameters (`C`, `solver`, `penalty`)
- Use stratified multi-label cross-validation (via `iterstrat` or `scikit-multilearn`) to get more reliable evaluation given the small labeled dataset size

## 4. Explore Alternative Multi-Label Classifiers
One-vs-Rest Logistic Regression does not model label dependencies.

- Try Label Powerset or Classifier Chain approaches to capture correlations between labels (e.g., a response with `4-DM` is likely to also have `3-PI`)
- Experiment with tree-based models (Random Forest, etc) inside the One-vs-Rest wrapper

## 5. Expand Labeled Data via Active Learning
With only 500 labeled samples, the model's ceiling is limited.

- Implement active learning: use the trained model to identify the most uncertain unlabeled samples and prioritize them for manual annotation
- This is especially valuable for rare labels (`2-GI`, `6-RE`) where even a small number of additional annotations could significantly improve recall

## 6. Robust Evaluation
The current evaluation uses a single train/test split (80/20) on 500 samples, which may produce unstable metrics.

- Use k-fold cross-validation (stratified for multi-label) to get more stable F1 estimates
- Report per-label F1, precision, and recall alongside micro/macro averages
- Track performance across semi-supervised iterations with a held-out validation set separate from the test set, to avoid using test data for early stopping decisions
