import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import json
import string

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Read the Excel file
df = pd.read_excel('dish.xlsx')

# Replace NaN with an empty string
df = df.fillna('')

# Prepare data for clustering
recipe_titles = df['Recipe_title'].tolist()
characteristics = df[['Taste', 'Odour', 'Colour', 'Texture', 'Description']].astype(str).values.tolist()

# Combine the characteristics into a single string for each recipe
combined_data = [" ".join(row) for row in characteristics]

# Tokenize the combined characteristics
tokenized_data = [" ".join(word_tokenize(doc.lower())) for doc in combined_data]

# Step 1: TF-IDF Vectorization with stop words removal and punctuation handling
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',  # Removes common stop words
    token_pattern=r'\b\w+\b'  # Removes punctuation
)
tfidf_matrix = vectorizer.fit_transform(tokenized_data)

# Step 2: Find the optimal number of PCA components based on explained variance
explained_variances = []
for n_components in range(2, 101):  # Testing for components from 2 to 100
    pca = PCA(n_components=n_components)
    pca.fit(tfidf_matrix.toarray())
    explained_variances.append(sum(pca.explained_variance_ratio_))

# Plot explained variance to help visualize the elbow point
plt.figure(figsize=(8, 6))
plt.plot(range(2, 101), explained_variances, marker='o')
plt.title('Explained Variance by Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Total Explained Variance')
plt.grid(True)
plt.show()

# Use the elbow point (or a high explained variance) for n_components selection
# For example, we choose 0.95 variance threshold
pca = PCA(n_components=0.95)
tfidf_reduced = pca.fit_transform(tfidf_matrix.toarray())

# Step 3: Normalize the data
tfidf_normalized = normalize(tfidf_reduced, norm='l2')

# Step 4: Apply K-Means Clustering
num_clusters = 35  # Form at least 15 clusters
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(tfidf_normalized)

# Step 5: Dimensionality Reduction for Visualization
pca_2d = PCA(n_components=2)
tfidf_2d = pca_2d.fit_transform(tfidf_normalized)

# Step 6: Identify Key Characteristics for Each Cluster
cluster_centers = kmeans.cluster_centers_
inverse_transform = pca.inverse_transform(cluster_centers)
characteristics_names = vectorizer.get_feature_names_out()

key_characteristics = []
for center in inverse_transform:
    top_indices = np.argsort(center)[::-1][:5]  # Top 5 features defining the cluster
    top_features = [characteristics_names[i] for i in top_indices]
    key_characteristics.append(", ".join(top_features))

# Step 7: Plot Clusters with Adjusted Size
plt.figure(figsize=(14, 10))  # Increased the height from 6 to 10

# Subplot 1: Scatter Plot of Clusters
plt.subplot(2, 1, 1)  # Changed to 2 rows and 1 column
unique_labels = set(labels)
colors = [plt.cm.jet(float(i) / len(unique_labels)) for i in range(len(unique_labels))]
for cluster_id in unique_labels:
    color = colors[cluster_id]
    label = f'Cluster {cluster_id}'
    plt.scatter(
        tfidf_2d[labels == cluster_id, 0],
        tfidf_2d[labels == cluster_id, 1],
        c=color,
        label=label
    )
plt.title('K-Means Clustering of Recipes')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Subplot 2: Bar Graph of Cluster Sizes with Characteristics
plt.subplot(2, 1, 2)  # Changed to 2 rows and 1 column
cluster_sizes = [list(labels).count(cluster_id) for cluster_id in unique_labels]
cluster_labels = [f'Cluster {cluster_id}' for cluster_id in unique_labels]
bars = plt.bar(cluster_labels, cluster_sizes, color=[colors[i] for i in unique_labels])
plt.title('Number of Items in Each Cluster')
plt.xlabel('Clusters')
plt.ylabel('Number of Items')
plt.xticks(rotation=45, ha='right')

# Add number labels above each bar and display key characteristics
for bar, characteristics in zip(bars, key_characteristics):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height}\n({characteristics})',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# Step 8: Save Clustering Results to a File
clustering_results = {}
for cluster_id in unique_labels:
    clustering_results[f'Cluster {cluster_id}'] = {
        'Characteristics': key_characteristics[cluster_id],
        'Recipes': [
            recipe_titles[i] for i in range(len(recipe_titles)) if labels[i] == cluster_id
        ]
    }

with open('clustering_results.json', 'w') as json_file:
    json.dump(clustering_results, json_file, indent=4)

print("Clustering results saved to 'clustering_results.json'.")
