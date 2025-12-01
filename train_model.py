"""
Train LSTM model for fake news detection.
Loads data, preprocesses, trains model, and saves artifacts.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns

from preprocess import preprocess_dataframe

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64


def load_data():
    """Load and combine fake and real news datasets."""
    print("Loading datasets...")
    
    # Load fake news
    fake_df = pd.read_csv('data/Fake.csv')
    fake_df['label'] = 0  # Fake = 0
    
    # Load real news
    true_df = pd.read_csv('data/True.csv')
    true_df['label'] = 1  # Real = 1
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Fake news: {len(fake_df)}, Real news: {len(true_df)}")
    
    return df


def prepare_data(df):
    """Preprocess and prepare data for training."""
    print("\nPreprocessing text...")
    
    # Combine title and text for better context
    df['content'] = df['title'] + ' ' + df['text']
    
    # Preprocess text
    df = preprocess_dataframe(df, text_column='content')
    
    # Remove empty strings
    df = df[df['cleaned_text'].str.len() > 0]
    
    return df


def create_tokenizer(texts):
    """Create and fit tokenizer on text data."""
    print("\nCreating tokenizer...")
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    return tokenizer


def build_model():
    """Build LSTM model architecture."""
    print("\nBuilding model...")
    
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, 
                  output_dim=EMBEDDING_DIM, 
                  input_length=MAX_SEQUENCE_LENGTH),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print(model.summary())
    
    return model


def plot_training_history(history):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved as 'training_history.png'")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")


def main():
    """Main training pipeline."""
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Prepare data
    df = prepare_data(df)
    
    # Create tokenizer
    tokenizer = create_tokenizer(df['cleaned_text'].values)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build model
    model = build_model()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
    
    # Predictions for detailed metrics
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save('models/fake_news_model.h5')
    
    with open('models/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save configuration
    config = {
        'max_words': MAX_WORDS,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM
    }
    
    with open('models/config.pickle', 'wb') as f:
        pickle.dump(config, f)
    
    print("\n✅ Training complete! Model saved to 'models/' directory")


if __name__ == "__main__":
    main()