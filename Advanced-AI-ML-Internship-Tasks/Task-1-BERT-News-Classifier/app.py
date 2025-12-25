import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="BERT News Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Topic Classification using BERT")
st.write("Enter a news headline and the model will predict its category.")

# --------------------------------------------------
# Load Model & Tokenizer (Cached for performance)
# --------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned BERT model and tokenizer
    from the local directory.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-news-classifier")
    model = BertForSequenceClassification.from_pretrained("bert-news-classifier")
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# Label Mapping
# --------------------------------------------------
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# --------------------------------------------------
# User Input
# --------------------------------------------------
user_text = st.text_area(
    "📝 Enter News Headline:",
    placeholder="Apple unveils new AI-powered iPhone features"
)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔍 Classify News"):
    if user_text.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        # Tokenize input text
        inputs = tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted class index
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Display result
        st.success(
            f"🧠 **Predicted Category:** {LABELS[predicted_class]}"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Developed as part of AI/ML Engineering Internship | DevelopersHub")
