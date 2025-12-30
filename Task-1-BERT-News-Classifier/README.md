# 📰 News Topic Classification using BERT

## Objective
Fine-tune a BERT transformer model to classify news headlines into four categories.

## Dataset
AG News Dataset (Hugging Face)

## Model
- bert-base-uncased
- Fine-tuned for 4-class classification

## Evaluation Metrics
- Accuracy: 94.7%
- Weighted F1 Score: 94.7%

## Deployment
Interactive Streamlit web application for real-time predictions.

## Tech Stack
- PyTorch
- Hugging Face Transformers
- Streamlit
- Scikit-learn
- 
## Model Weights
The trained BERT model is hosted on Google Drive and is downloaded automatically at runtime due to GitHub file size limits.
🔗 Download link:
https://drive.google.com/drive/folders/1m_LtfZ4To9PEeW9y2SzuPJUVM4jkBWq9?usp=drive_link

After downloading, place the `bert-news-classifier` folder in the same directory as `app.py` before running the Streamlit app.

## Deployment

This project is deployed on Streamlit Cloud.

## 🚀 Live Demo

[![Open in Streamlit](https://bertnewsclassifier.streamlit.app/)


