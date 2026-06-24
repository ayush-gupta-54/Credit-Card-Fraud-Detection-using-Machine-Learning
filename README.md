# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project detects whether a credit card transaction is genuine or fraudulent using machine learning algorithms. Credit card fraud detection is an important real-world application of machine learning because fraudulent transactions are rare but can cause serious financial loss.

The system is trained on the Kaggle Credit Card Fraud Detection dataset. It uses transaction features such as `Time`, `Amount`, and anonymized PCA features `V1` to `V28` to classify transactions as genuine or fraudulent.

The final trained model is integrated with a Streamlit web application where users can select a transaction from the dataset or manually enter transaction values to get a prediction.

---

## Problem Statement

To build a machine learning model that can classify credit card transactions as genuine or fraudulent by learning patterns from historical transaction data and handling the issue of class imbalance effectively.

---

## Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Dataset link:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

The dataset file used in this project is:

```text
creditcard.csv
```

---

## Dataset Description

The dataset contains credit card transaction records.

Important columns:

| Column    | Description                         |
| --------- | ----------------------------------- |
| Time      | Time elapsed between transactions   |
| V1 to V28 | Anonymized PCA-transformed features |
| Amount    | Transaction amount                  |
| Class     | Target column                       |

Target column:

```text
Class = 0 → Genuine Transaction
Class = 1 → Fraudulent Transaction
```

Each row in the dataset represents one transaction.

The columns `V1` to `V28` are not individual transactions. They are anonymized numerical features generated using PCA to protect sensitive customer and transaction information.

---

## Machine Learning Type

This project is a **supervised machine learning** project because the dataset contains labeled outputs.

It is also a **binary classification** problem because the model predicts one of two classes:

```text
Genuine Transaction
Fraudulent Transaction
```

---

## Main Challenge: Imbalanced Dataset

The dataset is highly imbalanced. Genuine transactions are much higher in number compared to fraudulent transactions.

Because of this, accuracy alone is not a reliable metric. A model can get high accuracy by predicting most transactions as genuine, but it may fail to detect actual frauds.

To handle this issue, this project uses:

```python
class_weight="balanced"
```

This gives more importance to the minority class, which is fraud, during model training.

---

## Algorithms Used

The project uses and compares two machine learning algorithms:

1. Logistic Regression
2. Random Forest Classifier

### Logistic Regression

Logistic Regression is used as a baseline classification model. It predicts the probability of a transaction being fraudulent.

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees and makes the final prediction using majority voting. It helps reduce overfitting and gives stable results.

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC-AUC Score

For fraud detection, recall and ROC-AUC are especially important because detecting actual fraud transactions is more critical than only achieving high accuracy.

---

## Project Workflow

```text
1. Load the dataset
2. Check class distribution
3. Separate input features and target column
4. Split data into training and testing sets
5. Train Logistic Regression and Random Forest models
6. Evaluate models using suitable metrics
7. Select the best model based on ROC-AUC score
8. Save the best model using joblib
9. Load the saved model in Streamlit app
10. Predict whether a transaction is genuine or fraudulent
```

---

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── creditcard.csv
├── train_model.py
├── app.py
├── fraud_detection_model.pkl
└── README.md
```

---

## Requirements

Install the required Python libraries using:

```bash
pip install pandas scikit-learn imbalanced-learn joblib streamlit
```

---

## How to Run the Project

### Step 1: Download Dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Place the file `creditcard.csv` in the project folder.

---

### Step 2: Train the Model

Run the training file:

```bash
python train_model.py
```

This will train the models, compare their performance, and save the best model as:

```text
fraud_detection_model.pkl
```

---

### Step 3: Run the Streamlit App

After the model is saved, run:

```bash
streamlit run app.py
```

The web app will open in your browser.

---

## Streamlit App Features

The Streamlit app provides two input methods:

### 1. Select Transaction from Dataset

The user can enter a row number from the dataset. The app will fetch that transaction and predict whether it is genuine or fraudulent.

Example fraudulent row number:

```text
541
```

### 2. Enter Custom Transaction Values

The user can manually enter values for:

```text
Time, V1, V2, ..., V28, Amount
```

Since `V1` to `V28` are PCA-transformed features, random values should not be entered. For proper testing, values can be copied from an actual transaction row in the dataset.

---

## Sample Fraudulent Transaction Values

These values are taken from a known fraudulent transaction in the dataset.

```text
Time: 406
V1: -2.312226542
V2: 1.951992011
V3: -1.609850732
V4: 3.997905588
V5: -0.522187865
V6: -1.426545319
V7: -2.537387306
V8: 1.391657248
V9: -2.770089277
V10: -2.772272145
V11: 3.202033207
V12: -2.899907388
V13: -0.595221881
V14: -4.289253782
V15: 0.389724120
V16: -1.140747180
V17: -2.830055675
V18: -0.016822468
V19: 0.416955705
V20: 0.126910559
V21: 0.517232371
V22: -0.035049369
V23: -0.465211076
V24: 0.320198199
V25: 0.044519167
V26: 0.177839798
V27: 0.261145003
V28: -0.143275875
Amount: 0
```

Expected output:

```text
Fraudulent Transaction Detected
```

---

## Important Note about V1 to V28

`V1` to `V28` are not transactions. Each row in the dataset is one transaction.

The columns `V1` to `V28` are hidden mathematical features created using PCA. Because of this, their exact real-world meaning is not directly understandable.

Fraud is not detected simply because values are negative or positive. The model detects fraud based on the overall pattern and relationship among all feature values.

---

## Model Output

The app displays:

```text
Genuine Transaction
```

or

```text
Fraudulent Transaction Detected
```

It also shows the fraud probability.

Example:

```text
Fraud Probability: 91.25%
```

---

## Advantages

* Solves a real-world financial security problem
* Uses supervised machine learning
* Handles imbalanced data
* Compares multiple ML algorithms
* Uses proper evaluation metrics
* Provides a simple Streamlit user interface
* Easy to demonstrate and explain

---

## Limitations

* The dataset is anonymized, so original transaction details are not visible
* Real-time fraud detection is not implemented
* Custom input is difficult because `V1` to `V28` are PCA-transformed values
* The model depends on historical transaction patterns
* New fraud techniques may require retraining the model

---

## Future Enhancements

* Add real-time transaction monitoring
* Use XGBoost or LightGBM for improved performance
* Add SHAP explainability to explain predictions
* Add a fraud analytics dashboard
* Deploy the app using Streamlit Cloud
* Add alert system using email or SMS
* Use deep learning autoencoders for anomaly detection

---

## Conclusion

This project demonstrates how machine learning can be used to detect fraudulent credit card transactions. Since fraud cases are rare, the project focuses on handling class imbalance and evaluating the model using suitable metrics such as precision, recall, F1-score, confusion matrix, and ROC-AUC score. The trained model is integrated with a Streamlit interface for easy prediction and demonstration.
