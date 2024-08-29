import requests

HUGGINGFACEHUB_API_TOKEN= '**********'

# Splitting dataset into 11 parts
df_parts = np.array_split(df, 11)
# Set up the Hugging Face Inference API URL and headers
mistral_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {
    "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"
}

# Function to classify sentiment for a batch of texts
def classify_sentiment_batch(texts, batch_size=100):
    sentiments = []
    session = requests.Session()  # Use a session to reuse the connection

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        payloads = [
            {
                "inputs": f"Classify the sentiment of this text as Pro-Palestine, Pro-Israeli or Neutral:\n\nText: \"{text}\"\nSentiment:",
                "parameters": {"max_new_tokens": 4, "temperature": 0.7}
            }
            for text in batch_texts
        ]

        # Make batch requests
        responses = [session.post(mistral_url, headers=headers, json=payload) for payload in payloads]

        for response in responses:
            if response.status_code == 200:
                result = response.json()
                sentiment = result[0]['generated_text'].strip()
                sentiment = sentiment.split(':')[-1].strip()
                sentiments.append(sentiment)
            else:
                sentiments.append(None)

    session.close()
    return sentiments

# Apply the classification function to the 'Message' column in batches
df_parts[x]['Sentiment'] = classify_sentiment_batch(df_parts[x]['Message'])


####### Put 0-11 manually in x of df_parts[x]. A 'For' loop could also have been used.