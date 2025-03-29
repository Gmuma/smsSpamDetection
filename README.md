**📩 SMS Spam Detection**

**📌 Overview**

This project is an SMS Spam Detection System that uses Natural Language Processing (NLP) and Machine Learning to classify SMS messages as Spam or Not Spam. It is built using Scikit-Learn, Streamlit, and Python.

**🚀 Features**

  Text Preprocessing (Cleaning, Tokenization, Lemmatization)

  Machine Learning Model (Multinomial Naive Bayes)

  Model Performance Evaluation (Accuracy, Precision, Recall)

  Streamlit Web App for User-Friendly Interaction

  Deployment Ready

**📂 Project Structure**

    SMS-Spam-Detection/
  │-- app.py               # Streamlit web app
  
  │-- model.pkl            # Trained ML model
  
  │-- vectorizer.pkl       # TF-IDF Vectorizer
  
  │-- dataset.csv          # SMS dataset
  
  │-- README.md            # Project documentation
  
  │-- requirements.txt     # Dependencies

**🛠 Installation & Setup**

  git clone 
  
    https://github.com/Gmuma/smsSpamDetection.git
    
    cd SMS-Spam-Detection

  Install Dependencies

    pip install -r requirements.txt
    
  Run the Streamlit App

    streamlit run app.py

**📊 Workflow**

  1️⃣ Data Cleaning

    Remove special characters, numbers, and stopwords.
    
    Convert text to lowercase.

  2️⃣ Exploratory Data Analysis (EDA)

    Visualize the distribution of spam vs. non-spam messages.

    Analyze word frequency.

  3️⃣ Text Preprocessing

    Tokenization using Spacy.

    Lemmatization to normalize words.

    Vectorization using TF-IDF.

  4️⃣ Model Building

    Train multiple ML models (Naive Bayes, SVM, Decision Trees, etc.).
    
    Select the best model based on accuracy and precision.

  5️⃣ Evaluation

    Compare models using Accuracy, Precision, Recall, and F1-Score.
    
    Fine-tune the best model.

  6️⃣ Deployment

    Save the trained model and vectorizer using Pickle.
    
    Build a Streamlit UI for real-time SMS classification.

**🎯 Future Improvements**

  Add Deep Learning models (LSTMs, Transformers).
    
  Improve text preprocessing techniques.
    
  Implement real-time SMS filtering.

**📌 Author**

  Your Name: Gm. Umamaheswara Rao

**📝 License**

  This project is open-source and available under the MIT License.
