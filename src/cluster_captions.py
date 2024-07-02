import json
import nltk
from sklearn.cluster import KMeans
from nltk.cluster.kmeans import KMeansClusterer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch

def load_captions(captions_path: str):
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    return captions

def get_bert_embeddings(captions: dict, model, tokenizer, device):
    model.to(device)
    embeddings = []
    for _, caption in tqdm(captions.items(), desc="Genarating embeddings"):
        inputs = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the [CLS] token representation as the embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)
    return embeddings

def cluster_embeddings(embeddings, num_clusters=5):
    clusterer = KMeansClusterer(num_clusters, 
                                distance=nltk.cluster.util.cosine_distance, 
                                repeats=2, avoid_empty_clusters=True)
    clusters = clusterer.cluster(embeddings, True)
    return clusters, clusterer.means()

def save_clusters(captions, labels, save_path):
    with open(save_path, 'w') as f:
        for label, caption in zip(labels, captions.items()):
            f.write(f"{label}: {caption[0]} - {caption[1]}\n")
    print(f"Clusters saved to {save_path}")

if __name__ == "__main__":
    captions = load_captions("data/captions.json")
    print("Captions loaded ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    embeddings = get_bert_embeddings(captions, model, tokenizer, device)
    print("Embeddings generated ...")
    
    labels, _ = cluster_embeddings(embeddings, num_clusters=3)
    print("Clustering done ...")
    
    save_clusters(captions, labels, "data/clusters.txt")

