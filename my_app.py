import gradio as gr
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import vision
#import streamlit as st
#from img_to_img_search import rec_images

import os
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_my_key.json'

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

with open("embeddings.pkl", "rb") as f:
    embeddings  = pickle.load(f)
    
embeddings = embeddings.astype("float32")

embedding_size = embeddings.shape[1]
n_clusters = 5
num_results = 5

quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(
    quantizer, embedding_size, n_clusters, 
    faiss.METRIC_INNER_PRODUCT,)

index.train(embeddings)
index.add(embeddings)

def describe_image(image_path):
    client = vision.ImageAnnotatorClient()

    # Use the image path directly with the Vision API
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return [label.description for label in labels]

def _search(query):
    images = []
    if query:
        
        query_embedding = model.encode(query)
        query_embedding = np.array(query_embedding).astype("float32")
        query_embedding = query_embedding.reshape(1,-1)
        _, indices = index.search(query_embedding, num_results)
        
        images = [f"images/{i+2}.png" for i in indices[0]]
    # elif query_image is not None:
    #     images = rec_images
    # else:
    #     images = []
    return images


def process_input(query_text, query_image):
    if query_text: 
        return _search(query_text)
    
    elif query_image:  # If image input is provided
        # Process image input (dummy function)
        
        processed_images = describe_image(query_image)
        return _search(" ".join(processed_images))

inputs = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(lines=1, label="Input Text"),
        gr.Image(type="filepath", label="Input Image")
    ],
    outputs=gr.Gallery(preview=True, label="Gallery Output"),
    title="Image Gallery",
)

inputs.launch()

