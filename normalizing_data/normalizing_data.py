import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the CSV file
df = pd.read_csv('../scrapping/scraped_data.csv')

# Download NLTK stop words
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def normalize_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    return ' '.join(words)


# Apply normalization to the text column
df['text_normalized'] = df['text'].apply(normalize_text)

# Fill missing text with an empty string
df['text_normalized'].fillna('', inplace=True)
# If image names are critical, fill missing values with a placeholder
df['images'].fillna('no_image', inplace=True)

# Standardize URLs
df['url'] = df['url'].str.lower().str.strip()
# Standardize image names
df['images'] = df['images'].str.lower().str.strip()

# Save the normalized DataFrame to a new CSV file
df.to_csv('normalized_data.csv', index=False)
