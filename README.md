# 📰 Fake News Detection Web App

Deep learning-powered news classifier using LSTM neural network.

## 🚀 Quick Start
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Download dataset
# Visit: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
# Place Fake.csv and True.csv in data/ folder

# 5. Train model
python3 train_model.py

# 6. Run application
streamlit run app.py
```

## ✨ Features

- **Single Text Prediction**: Analyze individual news headlines or articles
- **Batch CSV Processing**: Upload and analyze multiple articles at once
- **Confidence Scores**: Get probability scores for each prediction
- **Interactive UI**: Clean, modern interface built with Streamlit
- **Downloadable Results**: Export predictions as CSV
- **High Accuracy**: ~95%+ accuracy on test set

## 🧠 Model Architecture

- **Type**: Bidirectional LSTM
- **Input**: News text (up to 300 words)
- **Layers**:
  - Embedding (128 dimensions)
  - 2x Bidirectional LSTM (64, 32 units)
  - Dropout (0.5)
  - Dense layers
  - Sigmoid output
- **Training**: 10 epochs, batch size 64
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

## 📈 Performance
```
Accuracy:  95.4%
Precision: 94.5%
Recall:    96.4%
F1-Score:  95.4%
```

## 📋 Requirements

- macOS 10.15+
- Python 3.8 - 3.10
- 4GB+ RAM (8GB recommended)
- 5GB free disk space

## 📁 Project Structure
```
fake-news-detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── fake_news_model.h5
│   ├── tokenizer.pickle
│   └── config.pickle
├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Usage

### Single Text Prediction

1. Start app: `streamlit run app.py`
2. Go to "📝 Single Text" tab
3. Enter news headline or article
4. Click "🔍 Analyze News"
5. View prediction and confidence score

### Batch CSV Processing

1. Prepare CSV file with `text` column
2. Go to "📁 Batch CSV" tab
3. Upload CSV file
4. Click "🔍 Analyze CSV"
5. Download results

## 🔧 Configuration

Edit `train_model.py` to change model parameters:
```python
MAX_WORDS = 10000              # Vocabulary size
MAX_SEQUENCE_LENGTH = 300      # Max text length
EMBEDDING_DIM = 128            # Embedding dimension
EPOCHS = 10                    # Training epochs
BATCH_SIZE = 64                # Batch size

## 🐛 Troubleshooting

### TensorFlow Installation Issues
```bash
# If TensorFlow fails to install
pip install tensorflow-macos==2.13.0  # For Apple Silicon (M1/M2)
pip install tensorflow-metal==1.0.1   # GPU acceleration for M1/M2

# For Intel Macs
pip install tensorflow==2.13.0
```

### NLTK Data Issues
```bash
# Manual download
python3
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> exit()
```

### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

## 📚 Dataset

**Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

- **Total Articles**: ~45,000
- **Fake News**: 23,481 articles
- **Real News**: 21,417 articles
- **Features**: title, text, subject, date

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Configure: `Main file path: app.py`
5. Deploy!

### Deploy to Render

1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port=$PORT --server.headless=true`
6. Deploy!

## 📄 License

MIT License - Free to use and modify

## 🙏 Credits

- **Dataset**: Kaggle Fake and Real News Dataset
- **Frameworks**: TensorFlow, Keras, Streamlit, NLTK
- **Libraries**: Pandas, NumPy, Scikit-learn

---

Built with ❤️ using Python and Deep Learning
