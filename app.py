# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Utility Functions
# -----------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stopwords.words('english') and i not in string.punctuation]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

# -----------------------------
# Caching for speed
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
    df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
    df.drop_duplicates(inplace=True)
    df['target'] = df['target'].map({'ham':0,'spam':1})
    df['transformed_text'] = df['text'].apply(transform_text)
    df['num_char'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    return df

@st.cache_resource
def load_model_vectorizer():
    with open("vectorizer.pkl","rb") as f:
        tfidf = pickle.load(f)
    with open("final_model.pkl","rb") as f:
        model = pickle.load(f)
    return tfidf, model

df = load_data()
tfidf, model = load_model_vectorizer()

# Precompute WordClouds
spam_wc = WordCloud(width=300, height=200, background_color='white').generate(
    df[df['target']==1]['transformed_text'].str.cat(sep=" ")
)
ham_wc = WordCloud(width=300, height=200, background_color='white').generate(
    df[df['target']==0]['transformed_text'].str.cat(sep=" ")
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","EDA","Model Performance","Prediction","Batch Prediction","Dashboard"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("🔥 SMS Spam Detection Project 🔥")
    st.subheader("About the Project")
    st.markdown("""
    This project detects whether an SMS message is **Spam** or **Ham** (not spam) using machine learning.  
    It leverages a **Naive Bayes** classifier trained on a dataset of SMS messages.  

    **Features of this App:**
    - Explore SMS data through multiple charts.
    - Compare different models' performance.
    - Predict single SMS or batch of messages.
    - Dashboard to visualize top words and message stats.
    """)
    st.subheader("How to Use the App")
    st.markdown("""
    1. Navigate through sidebar to explore different pages.  
    2. Use **EDA** page to understand the dataset.  
    3. **Model Performance** page shows accuracies and precision of ML models.  
    4. **Prediction** page: Enter SMS text to check if it is Spam or Ham.  
    5. **Batch Prediction**: Upload CSV to get predictions for multiple messages.  
    6. **Dashboard**: View top words, message length distributions, and more.  
    """)
    st.markdown("---")
    st.markdown("✅ **Made by Ismail Shah**")

# -----------------------------
# EDA PAGE
# -----------------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    st.subheader("Target Distribution")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.pie(df['target'].value_counts(), labels=['Ham','Spam'], autopct="%0.2f%%", colors=['#5DADE2','#E74C3C'])
    st.pyplot(fig1)
    
    st.subheader("Message Lengths")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_char'], bins=30, kde=True, ax=ax2, color='green')
    st.pyplot(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_words'], bins=30, kde=True, ax=ax3, color='orange')
    st.pyplot(fig3)
    
    fig4, ax4 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_sentences'], bins=30, kde=True, ax=ax4, color='purple')
    st.pyplot(fig4)
    
    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(8,5))
    sns.heatmap(df[['target','num_char','num_words','num_sentences']].corr(), annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)
    
    st.subheader("WordCloud for Spam")
    fig6, ax6 = plt.subplots(figsize=(7,4))
    ax6.imshow(spam_wc)
    ax6.axis('off')
    st.pyplot(fig6)
    
    st.subheader("WordCloud for Ham")
    fig7, ax7 = plt.subplots(figsize=(7,4))
    ax7.imshow(ham_wc)
    ax7.axis('off')
    st.pyplot(fig7)
    
    st.subheader("Top 20 Words in Spam")
    spam_words = pd.Series(" ".join(df[df['target']==1]['transformed_text']).split()).value_counts().head(20)
    fig8, ax8 = plt.subplots(figsize=(7,4))
    sns.barplot(x=spam_words.values, y=spam_words.index, palette="Reds_r", ax=ax8)
    st.pyplot(fig8)
    
    st.subheader("Top 20 Words in Ham")
    ham_words = pd.Series(" ".join(df[df['target']==0]['transformed_text']).split()).value_counts().head(20)
    fig9, ax9 = plt.subplots(figsize=(7,4))
    sns.barplot(x=ham_words.values, y=ham_words.index, palette="Blues_r", ax=ax9)
    st.pyplot(fig9)
    
    st.subheader("Message Length vs Target")
    fig10, ax10 = plt.subplots(figsize=(7,4))
    sns.boxplot(x='target', y='num_char', data=df, ax=ax10, palette=['#5DADE2','#E74C3C'])
    ax10.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig10)

# -----------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    X = tfidf.transform(df['transformed_text']).toarray()
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
    
    y_pred = model.predict(X_test)
    
    st.subheader("Accuracy & Precision")
    st.write("Accuracy:", accuracy_score(y_test,y_pred))
    st.write("Precision:", precision_score(y_test,y_pred))
    
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(7,4))
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)
    
    st.subheader("Prediction Distribution")
    fig_dist, ax_dist = plt.subplots(figsize=(7,4))
    sns.countplot(x=y_pred, palette=['#5DADE2','#E74C3C'], ax=ax_dist)
    ax_dist.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig_dist)
    
# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":
    st.title("✉️ Single SMS Prediction")
    user_input = st.text_area("Enter SMS Text")
    if st.button("Predict"):
        if user_input.strip() != "":
            transformed = transform_text(user_input)
            vector = tfidf.transform([transformed]).toarray()
            pred = model.predict(vector)[0]
            st.success("Prediction: **Spam**" if pred==1 else "Prediction: **Ham**")
        else:
            st.warning("Please enter a message.")

# -----------------------------
# BATCH PREDICTION PAGE
# -----------------------------
elif page == "Batch Prediction":
    st.title("📑 Batch SMS Prediction")
    file = st.file_uploader("Upload CSV", type=['csv'])
    if file is not None:
        batch_df = pd.read_csv(file)
        if 'text' in batch_df.columns:
            batch_df['transformed_text'] = batch_df['text'].apply(transform_text)
            batch_vector = tfidf.transform(batch_df['transformed_text']).toarray()
            batch_df['prediction'] = model.predict(batch_vector)
            batch_df['prediction'] = batch_df['prediction'].map({0:'Ham',1:'Spam'})
            st.dataframe(batch_df)
        else:
            st.error("CSV must have a column named 'text'")

# -----------------------------
# DASHBOARD PAGE
# -----------------------------
elif page == "Dashboard":
    st.title("📊 Dashboard")
    
    st.subheader("Top 10 Words in Spam")
    spam_words = pd.Series(" ".join(df[df['target']==1]['transformed_text']).split()).value_counts().head(10)
    fig1, ax1 = plt.subplots(figsize=(7,4))
    sns.barplot(x=spam_words.values, y=spam_words.index, palette="Reds_r", ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Top 10 Words in Ham")
    ham_words = pd.Series(" ".join(df[df['target']==0]['transformed_text']).split()).value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.barplot(x=ham_words.values, y=ham_words.index, palette="Blues_r", ax=ax2)
    st.pyplot(fig2)
    
    st.subheader("Message Length Distribution")
    fig3, ax3 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_char'], bins=30, kde=True, color='green', ax=ax3)
    st.pyplot(fig3)
    
    st.subheader("Words per Message")
    fig4, ax4 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_words'], bins=30, kde=True, color='orange', ax=ax4)
    st.pyplot(fig4)
    
    st.subheader("Sentences per Message")
    fig5, ax5 = plt.subplots(figsize=(7,4))
    sns.histplot(df['num_sentences'], bins=30, kde=True, color='purple', ax=ax5)
    st.pyplot(fig5)
    
    st.subheader("Spam vs Ham Counts")
    fig6, ax6 = plt.subplots(figsize=(7,4))
    sns.countplot(x='target', data=df, palette=['#5DADE2','#E74C3C'], ax=ax6)
    ax6.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig6)
    
    st.subheader("Boxplot: Characters by Target")
    fig7, ax7 = plt.subplots(figsize=(7,4))
    sns.boxplot(x='target', y='num_char', data=df, palette=['#5DADE2','#E74C3C'], ax=ax7)
    ax7.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig7)
    
    st.subheader("Boxplot: Words by Target")
    fig8, ax8 = plt.subplots(figsize=(7,4))
    sns.boxplot(x='target', y='num_words', data=df, palette=['#5DADE2','#E74C3C'], ax=ax8)
    ax8.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig8)
    
    st.subheader("Boxplot: Sentences by Target")
    fig9, ax9 = plt.subplots(figsize=(7,4))
    sns.boxplot(x='target', y='num_sentences', data=df, palette=['#5DADE2','#E74C3C'], ax=ax9)
    ax9.set_xticklabels(['Ham','Spam'])
    st.pyplot(fig9)
    
    st.subheader("Correlation Heatmap")
    fig10, ax10 = plt.subplots(figsize=(8,5))
    sns.heatmap(df[['target','num_char','num_words','num_sentences']].corr(), annot=True, cmap="coolwarm", ax=ax10)
    st.pyplot(fig10)
