# Credit Card Fraud Detection

**Which model catches fraud best, and which data do we really need for it?**

Machine Learning project, Ironhack Data Analytics Bootcamp (individual project, October 2026).

📊 **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1PxnFf6c5tsSbt9eFr47TGJNjy0-b8Drr/edit?usp=sharing)  
📋 **Project board:** [Trello](https://trello.com/b/VHfKSApJ/credit-card-fraud-detection-mlproject)

---

## Project overview

The goal is to predict whether a credit card transaction is **fraud (1)** or **not fraud (0)**, and to find out:

1. Which machine learning model catches fraud best
2. Which features the model really needs (fewer features = less data to collect)

## Dataset

- **Source:** [Credit Card Fraud dataset on Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud) (`card_transdata.csv`, CC0: Public Domain)
- **Size:** 1,000,000 transactions, 7 features, 1 target (`fraud`)
- **Imbalanced:** 8.74% fraud, 91.26% not fraud
- No missing values and no duplicates

| Feature | Description |
|---|---|
| `distance_from_home` | Distance from home where the transaction happened |
| `distance_from_last_transaction` | Distance from the last transaction |
| `ratio_to_median_purchase_price` | Purchase price compared to the customer's median purchase price |
| `repeat_retailer` | Transaction at a retailer used before (0/1) |
| `used_chip` | Chip used (0/1) |
| `used_pin_number` | PIN used (0/1) |
| `online_order` | Online order (0/1) |

> The dataset `card_transdata.csv` is included in this repository (original source: Kaggle, link above).

## Repository structure

| File | Content |
|---|---|
| `01_data_exploration.ipynb` | First look at the data: size, data types, data quality, target distribution, summary statistics |
| `02_feature_engineering_and_modeling.ipynb` | Baseline KNN, fraud patterns, a new feature, scaling tests, feature selection, model comparison, confusion matrix, feature importance, balancing, conclusion |

## Approach

1. **Data exploration:** fraud rate per feature, medians of the continuous features
2. **Train-test split:** 80% train (800,000) / 20% test (200,000), **stratified** so both sets keep 8.74% fraud
3. **Baseline:** KNN (k = 3), then feature scaling (StandardScaler, MinMaxScaler) and a new feature (`online_no_pin`)
4. **Model comparison** on the same data: Logistic Regression, Decision Tree, Bagging, Random Forest, Gradient Boosting, AdaBoost
5. **Evaluation:** precision, recall and F1 score for the fraud class (accuracy alone is misleading with imbalanced data)
6. **Feature importance** (Random Forest) and a test with fewer features
7. **Imbalance:** random oversampling and undersampling of the training data for Logistic Regression

## Results

Test set: 200,000 transactions (17,481 fraud). Scores for the fraud class.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 |
| Bagging | 1.000 | 1.000 | 1.000 |
| **Decision Tree** | **1.000** | **1.000** | **1.000** |
| AdaBoost | 0.998 | 0.999 | 0.999 |
| Gradient Boosting | 1.000 | 0.997 | 0.999 |
| KNN, scaled (MinMaxScaler) | 0.997 | 0.993 | 0.995 |
| Logistic Regression + oversampling | 0.545 | 0.934 | 0.688 |
| Logistic Regression | 0.906 | 0.420 | 0.574 |

**Best model: Decision Tree.** It caught all 17,481 fraud cases (0 missed) and raised only 2 false alarms out of 182,519 normal transactions. Random Forest and Bagging are just as good, but a single tree is faster and its rules can be read and explained.

## Key findings

- **Fraud patterns:** fraud is far more common in online orders (12.7% vs. 1.3% in store) and without a PIN (9.7% vs. 0.3%). A typical fraud purchase is about 5 times the customer's usual amount.
- **Scaling matters for KNN:** recall rose from 93% to 99% after scaling.
- **Tree-based models perform best,** because fraud in this data depends on combinations of conditions (e.g. online AND no PIN AND high price).
- **6 of 7 features are enough:** `repeat_retailer` can be dropped without losing performance. `distance_from_last_transaction` is needed: without it, recall drops from 100% to 93%.
- **Balancing** (oversampling) raised Logistic Regression's recall from 42% to 93%, at the cost of many more false alarms. It still stayed far behind the tree models.

## Limitations

- The dataset is most likely **synthetic**: the fraud labels seem to follow simple rules with round thresholds (e.g. purchase ratio > 4, distance > 100). The near-perfect scores show that the models learned these rules; real-world performance would be lower.
- **One train-test split** and **no hyperparameter tuning** (time limit).
- **F1** treats missed fraud and false alarms as equally costly. A real bank would likely focus on recall, with a minimum precision.

## Next steps

- Cross-validation to check that the results are stable
- Hyperparameter tuning with Grid Search or Random Search
- SMOTE for Logistic Regression
- Test the approach on a real-world fraud dataset

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter Notebook

## How to run

1. Clone the repository (`card_transdata.csv` is in the same folder as the notebooks).
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run `01_data_exploration.ipynb`, then `02_feature_engineering_and_modeling.ipynb`.

## Author

**Pariya Lakani**
[LinkedIn](https://www.linkedin.com/in/lakani-dataanalyst) · [GitHub](https://github.com/pari14830)
