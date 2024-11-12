# Offensive-Humour-Classification
Offensive Humour Text Classification

Offensive Humour Detection
This repository contains the code and resources for the project on detecting offensive humor using machine learning and natural language processing techniques. The dataset used for this project is the Reddit Jokes dataset.

Introduction
Detecting offensive humor in text is a challenging task that involves understanding both the linguistic content and the context of the joke. This project aims to build models that can effectively classify jokes as offensive or non-offensive.

Dataset
The Offensive Humor Dataset, sourced from Kaggle, comprises text samples classified into four levels of offensiveness: mildly offensive, moderately offensive, highly offensive, and extremely offensive. Each entry in the dataset features a joke or humorous statement, annotated based on its perceived level of offensiveness. This dataset, boasting a total of 92,153 jokes, is instrumental for advancing the development and assessment of models dedicated to detecting and categorizing offensive content within humor. Categorized into four distinct types, the dataset encompasses:

Clean Jokes (7,450 examples) Dark Jokes (79,230 examples) Dirty Jokes (5,473 examples) News articles categorized as non-jokes (10,710 examples)

Preprocessing
The preprocessing steps include:

Tokenization
Removing stop words Lowercasing Lemmatization

Embeddings
We generated word embeddings using the following methods: N-gram TF-IDF GloVe FastText Word2Vec (Skip-gram) Models

The following traditional machine learning models were used to classify the jokes:
Support Vector Machine (SVM) Random Forest (RF) XGBoost K-Nearest Neighbors (KNN) Neural Tangent Kernel (NTK)

Novel Approach
Our novel approach includes: Embedding BERT embeddings into a Parallel Neural Network.

Evaluation
We evaluated the models using various metrics including accuracy, precision, recall, F1-score, and AUC-ROC.

Conclusion
This study explored offensive humor classification using traditional machine learning models and a parallel neural network approach. Leveraging TF-IDF, n-gram, GloVe, FastText, Word2Vec, and BERT embeddings, models like SVM, Random Forest, KNN, NTK, and XGBoost were trained. With comprehensive preprocessing and data balancing, notable performance gains were achieved. The parallel neural network, adept at capturing intricate nuances, further enhanced accuracy. While traditional models excelled with structured features, the neural network demonstrated superior pattern modeling. This dual approach highlights the complementary roles of traditional and deep learning techniques in advancing NLP tasks.
