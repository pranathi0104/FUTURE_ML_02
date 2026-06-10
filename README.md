# Support Ticket Classification & Prioritization

## 📌 Project Overview

This project was developed as part of the **Future Interns Machine Learning Internship Program**.

The objective of this project is to build an NLP-based machine learning system capable of automatically classifying IT support tickets into predefined categories. Automating ticket classification helps organizations reduce manual effort, improve response times, and streamline support operations.

---

## 🎯 Problem Statement

Organizations receive thousands of support requests daily. Manually reviewing and categorizing tickets can be time-consuming and error-prone.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify support tickets into appropriate categories.

---

## 📂 Dataset

Dataset: **all_tickets_processed_improved_v3.csv**

### Categories

* Access
* Administrative Rights
* HR Support
* Hardware
* Internal Project
* Miscellaneous
* Purchase
* Storage

### Dataset Size

* Total Records: **47,837 Support Tickets**
* Categories: **8**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* LinearSVC
* Matplotlib
* Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Loading

Loaded and explored the support ticket dataset.

### 2. Text Preprocessing

Performed:

* Lowercasing
* Punctuation removal
* Special character removal
* Stopword removal

### 3. Feature Engineering

Used TF-IDF Vectorization:

```python
TfidfVectorizer(
    stop_words='english',
    min_df=2,
    max_df=0.9,
    ngram_range=(1,2)
)
```

Generated feature matrix:

```text
(47837, 145149)
```

### 4. Model Training

Algorithm used:

```text
Linear Support Vector Classifier (LinearSVC)
```

Train-Test Split:

```text
80% Training
20% Testing
```

### 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📊 Results

### Model Accuracy

```text
85.6%
```

### Classification Performance

| Category              | Precision | Recall | F1-Score |
| --------------------- | --------: | -----: | -------: |
| Access                |      0.92 |   0.89 |     0.90 |
| Administrative Rights |      0.84 |   0.73 |     0.78 |
| HR Support            |      0.85 |   0.85 |     0.85 |
| Hardware              |      0.82 |   0.87 |     0.84 |
| Internal Project      |      0.89 |   0.82 |     0.86 |
| Miscellaneous         |      0.82 |   0.83 |     0.82 |
| Purchase              |      0.96 |   0.89 |     0.92 |
| Storage               |      0.92 |   0.87 |     0.89 |

### Overall Metrics

* Accuracy: **85.6%**
* Weighted Precision: **0.86**
* Weighted Recall: **0.86**
* Weighted F1-Score: **0.86**

---

## 🤖 Example Prediction

Input Ticket:

```text
I forgot my password and cannot access my account.
```

Predicted Category:

```text
Access
```

---

## 📈 Business Impact

This solution can help organizations:

* Automatically route support tickets
* Reduce manual ticket classification effort
* Improve support response times
* Increase operational efficiency
* Scale customer and IT support processes

---

## 📁 Repository Structure

```text
FUTURE_ML_02
│
├── data
│   └── all_tickets_processed_improved_v3.csv
│
├── images
│   ├── confusion_matrix.png
│   ├── classification_report.png
│   └── prediction_output.png
│
├── notebook
│   └── Support_Ticket_Classification.ipynb
│
├── report
│   └── Task2_Report.docx
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Conclusion

A machine learning-based support ticket classification system was successfully developed using NLP techniques and LinearSVC. The model achieved **85.6% accuracy** on a dataset of **47,837 support tickets** across eight categories.

This project demonstrates the practical application of machine learning for automating support operations and improving business efficiency.

---

### Internship Task

**Future Interns – Machine Learning Task 2 (2026)**

**Support Ticket Classification & Prioritization**
