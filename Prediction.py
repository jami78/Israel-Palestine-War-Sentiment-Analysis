##Reading the dataset

df1= pd.read_excel('/content/original_translated1.xlsx')


##Dropping rows with missing 'Message' value

df1.dropna(subset=['Message'], inplace=True)


##Labelling (Predicting) Sentiments


# Transforming to Character-Level TF-IDF
X_new_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(df1['Message'])

#Predictions
predictions_df1 = rfct.predict(X_new_tfidf)

# Add predicted sentiments to df1
df1['Sentiment'] = predictions_df1

# Print the first few rows of df1 with predicted sentiments
print(df1[['Message', 'Sentiment']].head())