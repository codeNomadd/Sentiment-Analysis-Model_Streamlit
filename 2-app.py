import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# S3 Configuration
bucket_name = "mlops-irmuun"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

s3 = boto3.client('s3')

# Function to download the model from S3
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)  # Create directories
                s3.download_file(bucket_name, s3_key, local_file)

# Streamlit UI
st.title("Machine Learning Model Deployment at the Server")

# Button to download the model
if st.button("Download Model"):
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)

# Text area for user input
text = st.text_area("Enter Your Review", "Type...")

# Predict button
if st.button("Predict"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    try:
        # Try loading the local model
        classifier = pipeline('text-classification', model=local_path, device=device)
    except Exception as e:
        # Fallback to a pre-trained Hugging Face model for testing
        st.error(f"Model loading failed: {e}. Using a fallback model.")
        classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', device=device)

    with st.spinner("Predicting..."):
        output = classifier(text)
        label = output[0]['label']
        score = output[0]['score']

        # Debugging Output
        st.write(f"DEBUG: Label = {label}, Score = {score}")

        # Fancy output with emojis
        if 'positive' in label.lower():
            st.success(f"🌟 **Positive Sentiment**: {score:.2%} confidence")
        elif 'negative' in label.lower():
            st.error(f"💔 **Negative Sentiment**: {score:.2%} confidence")
        else:
            st.warning(f"🤔 **Neutral/Unknown Sentiment**: {score:.2%} confidence")
