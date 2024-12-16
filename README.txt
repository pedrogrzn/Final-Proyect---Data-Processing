README

# Final Project: Document Processing for Regression Task

## Project Overview

This project explores a regression task involving recipe reviews, predicting ratings based on text features. We implement multiple document vectorization techniques and machine learning models, including neural networks and pre-trained transformer-based models.

---

## Methodology

### 1. **Data Inspection & Analysis**

- **Dataset:** Recipe reviews with features like categories, ratings, and textual descriptions.
- **Initial Analysis:**
  - Inspected missing values and distribution of ratings.
  - Exploded categories for individual analysis.
  - Visualized correlations and description length versus rating.

---

### 2. **Text Preprocessing**

- **Library Used:** SpaCy
- **Steps:**
  - Tokenization
  - Lemmatization
  - Stopword removal

---

### 3. **Document Vectorization Techniques**

1. **TF-IDF:**

   - Maximum features: 5000
   - Created TF-IDF feature matrix.

2. **Word2Vec:**

   - Trained with Gensim (vector size: 100, window: 5).
   - Averaged word embeddings for documents.

3. **BERT Embeddings:**

   - Used `bert-base-uncased` from Hugging Face.
   - Extracted [CLS] token embeddings.

---

### 4. **Model Training & Evaluation**

#### a. **Random Forest Regressor**

- Model: Scikit-learn’s Random Forest
- Evaluation: MSE, R²

#### b. **Neural Network (PyTorch)**

- Architecture: Fully connected layers with ReLU activation.
- Training: Adam optimizer, MSE Loss.

#### c. **Fine-Tuned BERT Model**

- Used `BertForSequenceClassification` for regression.
- DataLoader setup with tokenization.
- Evaluation: MSE, R², and prediction visualizations.

---

### 5. **Validation and Results**

- Validation methodology explained with comparisons.
- Metrics: MSE, R², and scatter plots for actual vs. predicted ratings.

---

## Conclusion

This project demonstrates a robust pipeline for document regression using diverse techniques from TF-IDF to fine-tuned transformers. Each approach was evaluated, showcasing trade-offs in performance, scalability, and interpretability.

---

## How to Run the Code

1. Load the provided dataset `full_format_recipes.json`.
2. Execute the main Python script.
3. Follow the evaluation metrics and visualizations in the output.

---

## Dependencies

- Python 3.x
- Pandas, Matplotlib, Seaborn
- Scikit-learn, Gensim
- PyTorch, Hugging Face Transformers
- SpaCy (English model)

---

## Author

Project developed as a final assignment for a data processing course.

Pedro Garzon and Mateo Tode

