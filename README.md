# Project: NLP (DLBAIPNLP01) - Task 2: Text Classification for Spam Detection

This repository contains the project submission for the IU International University of Applied Sciences module **DLBAIPNLP01: Project: NLP**. The project implements and evaluates a machine learning pipeline for classifying SMS messages as "Spam" or "Ham" (legitimate).

---

## Project Overview

The objective of this project was to build, train, and evaluate a text classification model based on the requirements of Task 2. The system is designed to preprocess raw text data, train on a labeled dataset, and accurately classify new, unseen messages.

The final system uses a **Linear Support Vector Machine (SVM)**, which was chosen over a Naive Bayes model after a comparative analysis proved it provided a superior balance of accuracy and recall.

### Key Components
* **`spam_detection.ipynb`**: A Jupyter Notebook containing the entire data science pipeline, from data loading to final model evaluation.
* **`spam.csv`**: The "SMS Spam Collection Dataset" used for training and testing.
* **`messages.csv`**: The "Ling-Spam Dataset", one of the new big dataset to test the generalisation performance of the chosen model.
* **`enron_spam_data`**: The "Enron-Spam Dataset", the biggest dataset in this project. It was used in the end also to test the generalisation performance of the chosen model. This Dataset is so Large (> 25 mb) that Github didn't allow it to be uploaded. Therefore, click this this [link](https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data), which will take you to the kaggle website where you can easily download it and use in the project.

---

## Methodology

The project followed a standard, end-to-end NLP workflow:

1.  **Data Loading & Cleaning**: The `spam.csv` dataset was loaded into a `pandas` DataFrame. Unnecessary columns were dropped, and the features were renamed to `label` and `message`.
2.  **Data Preprocessing**:
    * **Label Encoding**: The `label` column ('ham'/'spam') was mapped to numerical values (`0`/`1`).
    * **Vectorization**: The `TfidfVectorizer` from `scikit-learn` was used to convert the raw text `message` data into a numerical TF-IDF matrix. This step also handled stop-word removal and tokenization.
3.  **Model Training & Evaluation**:
    * The data was split 80/20 into training and test sets.
    * **Model 1 (Baseline)**: A `MultinomialNB` (Naive Bayes) model was trained.
    * **Model 2 (Final)**: A `LinearSVC` (Support Vector Machine) model was trained.
4.  **Comparative Analysis**: The models were compared using `classification_report` and `confusion_matrix`.
    * **Naive Bayes**: Achieved perfect **Precision (1.0)** but poor **Recall (0.75)**, failing to identify 25% of all spam.
    * **LinearSVC (SVM)**: Achieved a more balanced **98% Accuracy**, with **Precision (0.96)** and superior **Recall (0.87)**.
5.  **Final Model Selection**: The **LinearSVC (SVM)** was chosen as the final model. A qualitative test on new messages proved it was more effective at catching real-world spam (like phishing attempts) that the Naive Bayes model missed.

---

## How to Run

You can run this project in Google collab by downloading this repository and simply opening the `spam_detection.ipynb` file in it and loading the csv.
To run this project locally, you need to have Python 3 and the required libraries installed: pandas, seaborn, matplotlib, sklearn: TfidVectorizer, MultinomialNB, LinearSVC, Pipeline.

### 1. Clone the Repository

```
git clone https://github.com/KushTrip/Project-NLP-Text-Classification-for-Spam-Detection
cd Project-NLP-Text-Classification-for-Spam-Detection
```
### 2. Install Requirements

```
pip install pandas scikit-learn matplotlib seaborn jupyter 
```
### 3. Run the Notebook

```
jupyter notebook spam_detection.ipynb
```
