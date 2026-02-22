import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "resume_index.faiss"

model = SentenceTransformer(MODEL_NAME)

def get_embedding(text):
    return model.encode([text])[0].astype("float32")

def chunk_text(text, min_length=40):
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) >= min_length:
            chunks.append(line)
    return chunks

def build_or_load_index(dimension):
    if os.path.exists(INDEX_FILE):
        logging.info("Loading existing FAISS index")
        return faiss.read_index(INDEX_FILE)
    else:
        logging.info("Creating new FAISS index")
        return faiss.IndexFlatL2(dimension)

def save_index(index):
    faiss.write_index(index, INDEX_FILE)
    logging.info("Index saved to disk")

def add_to_index(index, vectors):
    index.add(vectors)
    save_index(index)
    return index

def search(index, query_vector, k=5):
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    return distances, indices