##Translation

from deep_translator import GoogleTranslator
import pandas as pd
import time
import numpy as np
# Function to translate the first 700 characters of text to English
def translate_first_700_chars(text):
    try:

        # Handle potential float values by converting to string
        if isinstance(text, float):
            text = str(text)

        # Handle potential integer values by converting to string
        if isinstance(text, int):
            text = str(text)

        # Extract the first 700 characters
        snippet = text[:700]

        # Translate the snippet to English
        translation = GoogleTranslator(source='auto', target='en').translate(snippet)
        return translation
    except Exception as e:
        print(f"Error translating text: {text}")
        print(e)
        return snippet  # Keep original snippet on error

# Function to translate a batch of texts to English
def translate_batch(texts, batch_size=100, retry_limit=3):
    translations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_translations = []

        for text in batch_texts:
            retry_count = 0
            translated = False

            while not translated and retry_count < retry_limit:
                try:
                    translated_text = translate_first_700_chars(text)
                    batch_translations.append(translated_text)
                    translated = True
                except Exception as e:
                    retry_count += 1
                    print(f"Retrying ({retry_count}/{retry_limit}) for text: {text}")
                    print(e)
                    time.sleep(1)  # Adding delay before retry

                if retry_count == retry_limit:
                    # Handle cases where 'text' might be an integer
                    if isinstance(text, int):
                        batch_translations.append(str(text))
                    else:
                        batch_translations.append(text[:700])  # Keep original snippet on repeated error

        translations.extend(batch_translations)
        time.sleep(1)  # Adding a delay to avoid rate limiting

    return translations


# Apply the batch translation function to the 'Message' column
df['Message'] = translate_batch(df['Message'].tolist())




## The LLM Model (Mistral-7B) has prompt per token limitation, hence dividing the dataset according to the limit.

df_parts = np.array_split(df, 11)