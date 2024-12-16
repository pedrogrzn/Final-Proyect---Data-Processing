
# Final Project: Detailed Code Functionality and Structure

## Code Overview

This document provides an in-depth explanation of the code implementation for the regression task using recipe reviews. It describes key variables, functions, classes, and important sections of the code.

---

## Code Structure

1. *Data Loading & Inspection:*
   - *File Loaded:* full_format_recipes.json
   - *Main Variables:*
     - data: Pandas DataFrame containing the full dataset.
     - categories_exploded: DataFrame with expanded categories for detailed analysis.
       *Key Code Snippet:*
   python
   data = pd.read_json('full_format_recipes.json')
   data.info()   # Inspect dataset structure
   data.describe()  # Statistical summary
   

---

## Data Analysis & Visualization

- *Libraries Used:* Pandas, Matplotlib, Seaborn
- *Analysis Steps:*
  - Display dataset statistics.
  - Visualize rating distribution.
  - Analyze average ratings by category.
  - Generate correlation heatmap between numeric variables.
    *Key Code Snippets:*
  python
  plt.figure(figsize=(8, 6))
  sns.histplot(data['rating'], bins=20, kde=True)
  plt.title('Rating Distribution')
  plt.show()
  

---

## Text Preprocessing

1. *Text Cleaning Function:*
   *Function:* spacy_preprocess(text)
   - *Input:* Raw text data from the directions column.
   - *Steps:* Tokenization, lemmatization, stopword removal, lowercasing.
   - *Output:* Preprocessed text ready for vectorization.
     *Key Code Snippet:*
   python
   import spacy
   nlp = spacy.load('en_core_web_sm')

   def spacy_preprocess(text):
       if isinstance(text, list):
           text = " ".join(text)
       if not isinstance(text, str) or text.strip() == "":
           return ""

       doc = nlp(text.lower())
       tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
       return " ".join(tokens)
   

---

## Document Vectorization

1. *TF-IDF Vectorization:*

   - *Object:* tfidf_vectorizer
   - *Generated Variable:* tfidf_matrix

   *Key Code Snippet:*

   python
   from sklearn.feature_extraction.text import TfidfVectorizer

   tfidf_vectorizer = TfidfVectorizer(max_features=5000)
   tfidf_matrix = tfidf_vectorizer.fit_transform(data_subset['processed_directions'])
   print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')
   

2. *Word2Vec Embeddings:*

   - *Model Training:* Tokenized texts.
   - *Embedding Function:* get_word2vec_embeddings(text, model)

   *Key Code Snippet:*

   python
   from gensim.models import Word2Vec

   tokenized_directions = data_subset['processed_directions'].apply(lambda x: x.split())
   word2vec_model = Word2Vec(sentences=tokenized_directions, vector_size=100, window=5, min_count=2, workers=4)

   def get_word2vec_embeddings(text, model):
       tokens = text.split()
       valid_tokens = [model.wv[token] for token in tokens if token in model.wv]
       if not valid_tokens:
           return np.zeros(model.vector_size)
       return np.mean(valid_tokens, axis=0)
   

3. *BERT Embeddings:*

   - *Model Loaded:* bert-base-uncased
   - *Embedding Function:* get_bert_embeddings(text, model, tokenizer)

   *Key Code Snippet:*

   python
   from transformers import BertTokenizer, BertModel
   import torch

   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   bert_model = BertModel.from_pretrained('bert-base-uncased')

   def get_bert_embeddings(text, model, tokenizer):
       inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
       with torch.no_grad():
           outputs = model(**inputs)
       return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
   

---

## Model Training & Evaluation

1. *Data Preparation:*

   - Splitting: Train-test split (80-20).
   - Scaling: StandardScaler for features.

   *Key Code Snippet:*

   python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   X = bert_embeddings
   y = data_subset['rating'].values

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   

2. *Models Implemented:*

   - *Random Forest Regressor:*

     - *Model Object:* rf_model
     - *Metrics:* MSE, R²

   - *Neural Network (PyTorch):*

     - *Class:* NeuralNet
     - *Training Loop:* Adam optimizer, MSE Loss

   - *Fine-Tuned BERT Model:*

     - *Model Class:* BertForSequenceClassification
     - *Evaluation:* Predictions, MSE, R²

   *Key Code Snippet:*

   python
   from sklearn.ensemble import RandomForestRegressor
   rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
   rf_model.fit(X_train, y_train)
   

---

## Evaluation & Metrics

- *Metrics:* MSE, R²
- *Visualizations:*
  - Scatter plot: Actual vs. Predicted ratings.
  - Training loss over epochs for neural networks.
    *Key Code Snippet:*
  python
  import matplotlib.pyplot as plt

  plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue')
  plt.plot([0, 5], [0, 5], color='red', linestyle='--')
  plt.xlabel('Actual Ratings')
  plt.ylabel('Predicted Ratings')
  plt.title('Model Performance')
  plt.grid(True)
  plt.show()
  

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

Mateo Tode and Pedro Garzón
