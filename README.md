
# Emotion Detection Using Machine Learning:

This project focuses on detecting emotions from text using a range of machine learning models. It includes training, evaluating, and tuning models for emotion classification, and a web-based interface built with Streamlit for real-time predictions and visualizations.

---

## 🔧 Features

- **Preprocessing pipeline** for emoji conversion, lemmatization, and stopword removal.
- **Model training and comparison** using SVM, Logistic Regression, Naive Bayes, Random Forest, and Gradient Boosting.
- **Hyperparameter tuning** via GridSearchCV.
- **Evaluation** with F1 score, accuracy, confusion matrix, and training time visualizations.
- **Interactive GUI** using Streamlit for single and batch text predictions.
- **Model and metrics persistence** using `joblib` and JSON.

---

## 🧠 Models Used

- Support Vector Machine (Linear)
- Logistic Regression
- Random Forest
- Naive Bayes
- Gradient Boosting
- Tuned Best Model (best performing model with hyperparameter optimization)

---

## 📁 Project Structure

```
.
├── advanced_model_comparison.py      # Script to preprocess, train, evaluate, and save models
├── emotion_detection_gui.py          # Streamlit app for emotion detection
├── train.txt                         # Training dataset (text;emotion format)
├── val.txt                           # Validation dataset
├── test.txt                          # Testing dataset
├── *.pkl                             # Saved models and vectorizer
├── model_metrics.json                # Performance metrics of models
├── confusion_matrix_*.png            # Confusion matrix images
└── model_comparison_f1.png           # Bar chart of F1 scores
```

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Key packages:**

- `scikit-learn`
- `nltk`
- `pandas`
- `matplotlib`, `seaborn`
- `streamlit`
- `plotly`
- `emoji`
- `joblib`

---

## 🚀 How to Run

### 1. Train and evaluate models

```bash
python advanced_model_comparison.py
```

This will:
- Preprocess data from `train.txt`, `val.txt`, and `test.txt`
- Train multiple models
- Evaluate and save metrics
- Generate visualizations and confusion matrices

### 2. Launch the Streamlit GUI

```bash
streamlit run emotion_detection_gui.py
```

This will:
- Let you enter text or upload files for emotion prediction
- Display model predictions, probabilities, metrics, and confusion matrices

---

## 📊 Output Artifacts

- `best_model_*.pkl`: Serialized trained models
- `tuned_best_model.pkl`: Best model after tuning
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for prediction
- `model_metrics.json`: Summary of F1 scores, accuracy, and training time
- PNGs for confusion matrices and performance charts

---

## 📌 Notes

- Ensure `train.txt`, `val.txt`, and `test.txt` are in the same directory before running the training script.
- The GUI uses the **Tuned Best Model** for default predictions and visualizations.
