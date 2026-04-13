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
