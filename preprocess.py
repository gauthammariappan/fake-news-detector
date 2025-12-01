"""
Text preprocessing utilities for fake news detection.
Handles cleaning, tokenization, and stopword removal.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load stopwords
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without stopwords
    """
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(filtered_tokens)


def preprocess_text(text):
    """
    Complete preprocessing pipeline.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Fully preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def preprocess_dataframe(df, text_column='text'):
    """
    Preprocess text column in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned text
    """
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    return df


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "Breaking News! President @JohnDoe announces new policy. Visit https://example.com for details!!! #Politics"
    print("Original:", sample_text)
    print("Cleaned:", preprocess_text(sample_text))

    