Fake Review Classification and Topic Modeling
Introduction:
Fake Review Classification and Topic Modeling represent a sophisticated application of Natural Language Processing (NLP) that focuses on detecting fraudulent reviews and uncovering thematic insights in textual data. Fake reviews, often employed to sway public perception, erode trust in online platforms. Advanced machine learning techniques, including deep learning and pre-trained models like RoBERTa, are leveraged to identify such reviews. Complementing this, topic modeling helps extract underlying themes, offering actionable insights into customer sentiments and concerns. By integrating these approaches, organizations can enhance transparency, improve decision-making, and foster trustworthy digital ecosystems.
Data Collection:
Dataset Details: Contains 40,432 product reviews with text and metadata, including ratings and votes.
Attributes: 4 features present in the dataset.
Data Preprocessing:
The following steps were applied for text cleaning and preparation:
1. Tokenization: Splitting text into individual words.
2. Removal of Non-Alphabetic Tokens: Filtering out characters that are not alphabetic.
3. Lowercasing: Standardizing text by converting to lowercase.
4. Stopword Removal: Eliminating common English stopwords using NLTK’s stopword list.
5. Lemmatization: Reducing words to their base forms using WordNetLemmatizer.
6. Stemming: Further reducing words to their stems using PorterStemmer.
7. TF-IDF Vectorization: Transforming text into numerical vectors with a vocabulary of 5,000 words, excluding stopwords.
Methodology:
1. Elbow Method for Optimal Cluster Identification:
The Elbow Method was utilized to determine the optimal number of clusters for k-means clustering. By plotting the Within-Cluster-Sum of Squares (WCSS) against cluster count, the 'elbow point' was identified to prevent overfitting or underfitting.
2. Clustering:
K-Means Clustering:
An unsupervised algorithm that partitions data into K clusters based on similarity. Iteratively assigns data points to the nearest centroid and adjusts centroids to minimize WCSS.
Optimal K Value: 4 clusters.
Effectiveness Evaluation: Silhouette score: 0.00963 (poor performance).
DBSCAN Clustering:
A density-based algorithm identifying clusters as dense regions separated by sparser regions, effectively handling noise and outliers. Does not require a predefined number of clusters.
Effectiveness Evaluation: Silhouette score: 0.373 (better performance than K-means).
Topic Modeling:
Latent Dirichlet Allocation (LDA):
LDA is a topic modeling algorithm assuming documents are mixtures of topics, with each topic being a distribution of words. Provides interpretable results, aiding in applications like text summarization, sentiment analysis, and information retrieval.
Fake Review Classification:
Following clustering and topic modeling, fake reviews were classified using machine learning and deep learning approaches.
1. Machine Learning Algorithms:
   - Logistic Regression (LR)
   - Support Vector Machine (SVM)
2. Deep Learning Algorithm:
   - LSTM (Long Short-Term Memory): A deep learning model for sequential data.
3. Pre-trained Model:
   - RoBERTa (Robustly Optimized BERT Approach): A transformer-based NLP model optimized for tasks like text classification and sentiment analysis.
Model Evaluation:
Evaluation metrics included:
 - Machine Learning Models: Accuracy, precision, recall, and F1-score.
 - Deep Learning Models (LSTM, RoBERTa): Accuracy and loss.
Conclusion:
A bar chart comparison of model accuracies revealed the following:
 - LSTM: Achieved the highest accuracy at 90%, making it the most effective model.
 - SVM: Performed well with an accuracy of 87%.
 - Logistic Regression (LR): Achieved 85% accuracy, slightly outperforming RoBERTa.
 - RoBERTa: Scored the lowest accuracy at 83%, indicating a need for further tuning or dataset-specific optimization.
Recommendation: LSTM emerged as the most suitable model for this task due to its superior accuracy, while RoBERTa's performance could be improved with additional optimization.
