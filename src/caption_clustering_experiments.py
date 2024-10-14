# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import spacy
import transformers
import torch

from datasets import Dataset
from hdbscan import HDBSCAN
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    precision_score,
    recall_score
)
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from umap import UMAP
from wordcloud import WordCloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

data_dir = "data"
train_data_path = os.path.join(data_dir, 'train_data_llava.json')
test_data_path = os.path.join(data_dir, 'test_data_llava.json')
models_dir = "models"

id2label = {
  0: 'Cultural_Religious',
  1: 'Fauna_Flora',
  2: 'Gastronomy',
  3: 'Nature',
  4: 'Sports',
  5: 'Urban_Rural'
}

label2id = {
  'Cultural_Religious': 0,
  'Fauna_Flora': 1,
  'Gastronomy': 2,
  'Nature': 3,
  'Sports': 4,
  'Urban_Rural': 5
}

llama_model_id = 'google/flan-t5-large'
labels = ['Sports', 'Nature', 'Urban and rural', 'Flora and fauna', 'Gastronomy', 'Cultural and religious']


def generate_df_from_json(json_path):
  with open(json_path) as f:
    data: dict = json.load(f)
    data_formated= {'image_name': [], 'caption': [], 'label': []}
    for key, value in data.items():
      data_formated['image_name'].append(key)
      data_formated['caption'].append(value['caption'])
      data_formated['label'].append(value['label'])
  
  df = pd.DataFrame.from_dict(data_formated)
  return df

def generate_zero_shot_classification_prompt(words, labels):
    prompt = (
        "Giveng the following keywords from a cluster, classify the cluster into one category.\n\n"
        f"Keywords: {', '.join(words)}\n\n"
        f"Categories: {', '.join(labels)}"
    )

    return prompt

def classify_text(model, tokenizer, words, labels, max_length=150):
    prompt = generate_zero_shot_classification_prompt(words, labels)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)

    # Decode the generated text
    category = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the category from the generated text
    return category

embedding_model = SentenceTransformer("all-MiniLM-L12-v2").to('cuda')
def generate_embeddings(df: pd.DataFrame, text_col: str):
    sentences = df[text_col].tolist()
    embeddings = embedding_model.encode(sentences)

    return embeddings

train_df = generate_df_from_json(train_data_path)
test_df = generate_df_from_json(test_data_path)
train_df = pd.concat([train_df, test_df], axis=0)
train_df['Label'] = train_df['label'].replace(id2label)
    
# Generate embeddings
train_embeddings = generate_embeddings(train_df, 'caption')

# Dim reduction
reducer = UMAP(n_neighbors=15, n_components=20, metric='cosine')
train_reduced = reducer.fit_transform(train_embeddings)

#### Clustering ####
# HDBSCAN
clusterer = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='leaf', prediction_data=True, approx_min_span_tree=False)
clusterer.fit(train_reduced)
train_df['HDBSCAN Cluster'] = clusterer.labels_

# KMeans
kmeans_clusterer = AgglomerativeClustering(n_clusters=6)
kmeans_clusterer.fit(train_reduced)
train_df['KMeans Cluster'] = kmeans_clusterer.labels_


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    llama_model_id,
    load_in_4bit=True,
    device_map="auto",  # This automatically places the model on the available GPUs
    torch_dtype=torch.float16,  # You can use bfloat16 or float16 for better memory usage
)

for method in ['HDBSCAN', 'KMeans']:
    #### BoW ####
    
    # Step 1: Combine all documents in the same cluster into a single document
    clustered_docs = train_df.groupby(f'{method} Cluster')['caption'].apply(' '.join).reset_index()
    clustered_docs['caption'] = clustered_docs['caption'].str.replace('image', '')
    clustered_docs['caption'] = clustered_docs['caption'].str.replace('shows', '')
    
    # Step 2: Calculate the TF-IDF for the combined documents
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(clustered_docs['caption'])
    
    # Step 3: Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Step 4: Extract the top 10 words with the highest TF-IDF score for each cluster
    top_n = 10
    clusters_words = {}
    for idx, row in enumerate(tfidf_matrix):
        representative_words = []
        cluster = clustered_docs[f'{method} Cluster'][idx]
        # Get the TF-IDF scores for the row and corresponding feature names
        tfidf_scores = zip(feature_names, row.toarray().flatten())
        # Convert to dictionary (word: tf-idf score)
        tfidf_dict = {word: score for word, score in tfidf_scores}
        # Sort by TF-IDF score in descending order and get the top N words
        tfidf_scores = zip(feature_names, row.toarray().flatten())
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
        for word, score in sorted_scores:
            representative_words.append(word)
        clusters_words[cluster] = representative_words
    
    cluster_mapping = {}
    for cluster, words in clusters_words.items():
        category = classify_text(model, tokenizer, words, labels)
        print(f"Words: {', '.join(words)}")
        print(f"Category: {category}")
        print("----------------------------")
        cluster_mapping[cluster] = category
    
    train_df['zero_shot_label'] = train_df[f'{method} Cluster'].map(cluster_mapping)
    zslabels2id = {
        'Cultural and religious': 0,
        'Flora and fauna': 1,
        'Gastronomy': 2,
        'Nature': 3,
        'Sports': 4,
        'Urban and rural': 5 
    }
    train_df['zs_id'] = train_df['zero_shot_label'].map(zslabels2id)
    y_true = train_df['label'].tolist()
    y_pred = train_df['zs_id'].tolist()
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    print(f"Precision: {str(round(precision*100, 2))}")
    print(f"Recall: {str(round(recall*100, 2))}")
    print(f"Accuracy: {str(round(acc*100, 2))}")
    
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    classes = "method," + ",".join(label2id.keys())
    results = f"{method}," + ",".join([str(round(acc*100, 2)) for acc in cm.diagonal()])
