import streamlit as st
import pickle
import spacy
from nltk.stem.porter import PorterStemmer
nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha]
    tokens = [token for token in tokens if not nlp.vocab[token].is_stop]
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    return " ".join(tokens) if tokens else ""

# Load vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.stop()
    
st.title("📩 SMS Spam Detection")

input_sms = st.text_area("📌 Enter your message below:")

if st.button("Submit"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Check if transformed text is empty
        if not transformed_sms:
            st.warning("⚠️ Unable to process message. Try a different input.")
        else:
            # 3. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 4. Predict
            result = model.predict(vector_input)[0]

            # 5. Display Result
            if result == 1:
                st.error("⚠️ **Spam Detected!**")
            else:
                st.success("✅ **Not Spam!**")
