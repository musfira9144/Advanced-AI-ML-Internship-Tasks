import os
import torch
import streamlit as st
import gdown
from transformers import BertTokenizer, BertForSequenceClassification

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="BERT News Classifier", page_icon="📰")

st.title("📰 News Topic Classification using BERT")

# --------------------------------------------------
# Model download settings
# --------------------------------------------------
MODEL_DIR = "bert-news-classifier"

# Google Drive FOLDER ID
GDRIVE_FOLDER_ID = "1m_LtfZ4To9PEeW9y2SzuPJUVM4jkBWq9"

@st.cache_resource
def load_model():
    """
    Downloads model from Google Drive (if not present)
    and loads tokenizer + model.
    """

    # Download model if not already present
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading model... Please wait ⏳")
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=MODEL_DIR,
            quiet=False
        )

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# Labels
# --------------------------------------------------
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# --------------------------------------------------
# User input
# --------------------------------------------------
text = st.text_area(
    "Enter a news headline:",
    placeholder="Apple shares rise after strong quarterly earnings"
)

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"🧠 Predicted Category: **{LABELS[pred]}**")
