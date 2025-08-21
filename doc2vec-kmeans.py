import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import nltk

# Download 'punkt' tokenizer data if not already downloaded
# nltk.download('punkt')

# Sample data
data = [
    "The economy is growing rapidly with an increase in job opportunities.",
    "The government announced new policies for economic growth.",
    "Artificial intelligence and machine learning are transforming industries.",
    "The new technology in smartphones is revolutionary.",
    "Global warming and climate change are critical issues affecting the planet.",
    "Scientists are researching renewable energy sources to combat climate change.",
    "The soccer team won the championship after a thrilling match.",
    "The basketball game last night was very exciting.",
    "The chef introduced a new recipe that has become very popular.",
    "The restaurant offers a variety of dishes from different cuisines."
]

# Tokenize the documents
tokenized_data = [word_tokenize(doc.lower()) for doc in data]

# Step 1: Build Vocabulary
word_counts = defaultdict(int)
for doc in tokenized_data:
    for word in doc:
        word_counts[word] += 1

vocab = sorted(word_counts.keys())

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

# Step 2: Prepare Training Data
tagged_data = []
for i, doc in enumerate(tokenized_data):
    tagged_data.append((doc, [f"DOC_{i}"]))

# Step 3: Initialize Model Parameters
vector_size = 20  # Dimensionality of the document vectors
learning_rate = 0.01
epochs = 50

# Initialize document and word vectors randomly
doc_vectors = np.random.rand(len(tagged_data), vector_size)
word_vectors = np.random.rand(vocab_size, vector_size)

# Step 4: Train Doc2Vec Model
for epoch in range(epochs):
    for doc, tags in tagged_data:
        doc_vec_sum = np.zeros(vector_size)
        for word in doc:
            word_idx = word_to_idx[word]
            doc_vec_sum += word_vectors[word_idx]

        doc_idx = int(tags[0].split('_')[1])
        doc_vectors[doc_idx] += learning_rate * doc_vec_sum / len(doc)

# Step 5: Normalize Document Vectors
doc_vectors_normalized = normalize(doc_vectors, norm='l2')

# Step 6: Apply K-means Clustering
num_clusters = 3  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(doc_vectors_normalized)
labels = kmeans.labels_

# Step 7: Print Clusters
for i, doc in enumerate(data):
    print(f"Document {i+1}: {doc}")
    print(f"Cluster: {labels[i]}")
    print()
