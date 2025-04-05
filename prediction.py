import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Label map
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
# ---------- Page Navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Sentiment Prediction", "EDA Insights", "Streamlit Insights"])

# ---------- Page 1: Prediction ----------
if page == "Sentiment Prediction":
    st.title("Sentiment Predictor")
    st.markdown("Enter a review to analyze its sentiment using a 3-class RoBERTa model.")

    review = st.text_area("Enter your review:", height=150)

    if st.button("🔍 Predict"):
        if review.strip():
            with st.spinner("Analyzing..."):
                result = sentiment_pipeline(review)[0]
                label_id = int(result["label"].split("_")[-1])
                sentiment = label_map[label_id]
                score = result["score"]
            st.success(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence:** {score:.2%}")
        else:
            st.warning("Please enter a review to analyze.")


# ---------- Page 2: EDA ----------
elif page == "EDA Insights":
    st.title("EDA: User Review Sentiment Analysis")

    file_path = r"C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project5_V1/cleaned_reviews.csv"

    if not os.path.exists(file_path):
        st.error("The cleaned_reviews.csv file was not found at the specified path.")
    else:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        question = st.selectbox("Choose an analysis question:", [
            "1. What is the overall sentiment of user reviews?",
            "2. How does sentiment vary by rating?",
            "3. Which keywords are most associated with each sentiment?",
            "4. How has sentiment changed over time?",
            "5. Do verified users tend to leave more positive or negative reviews?",
            "6. Are longer reviews more likely to be negative or positive?",
            "7. Which locations show the most positive or negative sentiment?",
            "8. Is there a difference in sentiment across platforms (Web vs Mobile)?",
            "9. Which ChatGPT versions are associated with higher/lower sentiment?",
            "10. What are the most common negative feedback themes?"
        ])

        if "overall sentiment" in question:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        elif "vary by rating" in question:
            st.subheader("Sentiment vs. Rating")
            st.write(pd.crosstab(df['rating'], df['sentiment'], normalize='index'))
            fig = plt.figure()
            sns.countplot(data=df, x="rating", hue="sentiment")
            plt.legend(title="Sentiment", loc='center left', bbox_to_anchor=(1.0, 0.5))
            st.pyplot(fig)

        elif "keywords" in question:
            st.subheader("Word Clouds by Sentiment")
             # Define colors per sentiment
            color_map = {
                "negative": "Reds",
                "positive": "Greens",
                "neutral": "gray"
            }
            for sentiment in df["sentiment"].unique():
                text = " ".join(df[df["sentiment"] == sentiment]["lemmatized_review"].dropna().astype(str))
                # Pick appropriate color palette
                cmap = color_map.get(sentiment.lower(), "gray")

                wordcloud = WordCloud(
                    width=800,
                    height=300,
                    background_color="white",
                    colormap=cmap
                ).generate(text)
                st.markdown(f"**{sentiment.capitalize()} Reviews**")
                st.image(wordcloud.to_array())

        elif "changed over time" in question:
            st.subheader("Sentiment Over Time")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date"].dt.to_period("M")
            sentiment_by_month = df.groupby(["month", "sentiment"]).size().unstack().fillna(0)
            st.line_chart(sentiment_by_month)

        elif "verified" in question:
            st.subheader("Sentiment by Verified Purchase")
            fig = plt.figure()
            sns.countplot(data=df, x="verified_purchase", hue="sentiment")
            st.pyplot(fig)

        elif "longer reviews" in question:
            st.subheader("Review Length vs Sentiment")
            df["review_length"] = df["review"].astype(str).apply(lambda x: len(x.split()))
            fig = plt.figure()
            sns.boxplot(data=df, x="sentiment", y="review_length")
            st.pyplot(fig)

        elif "locations" in question:
            st.subheader("Sentiment by Location")
            top_locations = df["location"].value_counts().head(10).index
            filtered = df[df["location"].isin(top_locations)]
            fig = plt.figure()
            sns.countplot(data=filtered, x="location", hue="sentiment")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif "platforms" in question:
            st.subheader("Sentiment by Platform")
            fig = plt.figure()
            sns.countplot(data=df, x="platform", hue="sentiment")
            st.pyplot(fig)

        elif "versions" in question:
            st.subheader("Average Sentiment by ChatGPT Version")
            sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
            df["sentiment_score"] = df["sentiment"].map(sentiment_map)
            version_sentiment = df.groupby("version")["sentiment_score"].mean().sort_values()
            st.bar_chart(version_sentiment)

        elif "common negative" in question:
            st.subheader("Most Common Words in Negative Reviews")
            text = " ".join(df[df["sentiment"] == "negative"]["lemmatized_review"].dropna().astype(str))
            wordcloud = WordCloud(width=800, height=300, background_color="white").generate(text)
            st.image(wordcloud.to_array())
