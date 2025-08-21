import pandas as pd
import matplotlib.pyplot as plt
DATA_PATH = "IndianFinancialNews.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8", sep=",")

df.columns = [c.strip().lower() for c in df.columns]
df.drop(['unnamed: 0'],axis=1, inplace=True)
print("Columns:", df.columns)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date", "title", "description"])

df = df.drop_duplicates(subset=["title", "description"])

df["text_clean"] = df["title"].astype(str) + ". " + df["description"].astype(str)
# Basic length features
df["title_len"] = df["title"].apply(len)
df["desc_len"] = df["description"].apply(len)
# Quick EDA
print("\nSample rows:\n", df.head())
print("\nDate range:", df["date"].min(), "to", df["date"].max())
print("\nRow count after cleaning:", len(df))
# Plot: number of articles per year
df["year"] = df["date"].dt.year
df["year"].value_counts().sort_index().plot(kind="bar", figsize=(12,5))
plt.title("Number of Articles per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()
# Plot: avg description length per year
df.groupby("year")["desc_len"].mean().plot(kind="line", marker="o", figsize=(10,5))
plt.title("Avg Description Length per Year")
plt.xlabel("Year")
plt.ylabel("Avg Length")
plt.show()
# Save cleaned dataset
df.to_csv("cleaned_financial_news.csv", index=False)

df = pd.read_csv("cleaned_financial_news.csv")
#VADER Sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

vader = SentimentIntensityAnalyzer()

df["vader_score"] = df["text_clean"].apply(lambda x: vader.polarity_scores(str(x))["compound"])
df["vader_sentiment"] = df["vader_score"].apply(
    lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
)

print("\nVADER Sentiment Distribution:\n", df["vader_sentiment"].value_counts())
#FinBERT Sentiment (HuggingFace)

from transformers import pipeline

finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

# Run on a sample (full dataset may take time — start with 500 rows)
sample_texts = df["text_clean"].head(500).tolist()
finbert_results = finbert(sample_texts)

# Convert to DataFrame
finbert_df = pd.DataFrame(finbert_results)
df.loc[:499, "finbert_label"] = finbert_df["label"]
df.loc[:499, "finbert_score"] = finbert_df["score"]

print("\nFinBERT Sample Sentiment Distribution:\n", df["finbert_label"].value_counts())
df.to_csv("financial_news_with_sentiment.csv", index=False)
print("\n✅ Sentiment labels added & saved to financial_news_with_sentiment.csv")
# Make sure 'date' is datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Group by month-year
df["year_month"] = df["date"].dt.to_period("M")

sentiment_over_time = df.groupby(["year_month", "finbert_label"]).size().unstack(fill_value=0)

# Normalize by total per month
sentiment_over_time = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0)

# Plot
import matplotlib.pyplot as plt

sentiment_over_time.plot(kind="line", figsize=(12,6))
plt.title("Sentiment Trends Over Time (FinBERT)")
plt.xlabel("Time (Monthly)")
plt.ylabel("Proportion of Articles")
plt.legend(title="Sentiment")
plt.show()

# Annotate key events
events = {
    "2008-09": "Global Financial Crisis",
    "2016-11": "Demonetization",
    "2020-03": "COVID Crash"
}

fig, ax = plt.subplots(figsize=(12,6))
sentiment_over_time.plot(ax=ax)

for date, label in events.items():
    ax.axvline(pd.Period(date), color="red", linestyle="--", alpha=0.7)
    ax.text(pd.Period(date), 0.9, label, rotation=90, color="red")

plt.title("Sentiment Trends with Key Financial Events")
plt.show()
# Define simple keyword mapping
sectors = {
    "Banking": ["HDFC", "ICICI", "SBI", "Axis", "Kotak"],
    "IT": ["Infosys", "TCS", "Wipro"],
    "Energy": ["Reliance", "ONGC", "Adani"],
    "Automobile": ["Tata Motors", "Mahindra", "Maruti"]
}

def map_sector(title):
    for sector, keywords in sectors.items():
        if any(kw.lower() in str(title).lower() for kw in keywords):
            return sector
    return "Other"

df["sector"] = df["title"].apply(map_sector)

# Aggregate sentiment by sector
sector_sentiment = df.groupby(["sector", "finbert_label"]).size().unstack(fill_value=0)

# Normalize
sector_sentiment = sector_sentiment.div(sector_sentiment.sum(axis=1), axis=0)

# Plot
sector_sentiment.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Sentiment Distribution Across Sectors")
plt.ylabel("Proportion")
plt.show()


