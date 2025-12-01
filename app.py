"""
Streamlit web application for fake news detection.
Provides interface for single text input and CSV batch processing.
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .real-news {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .fake-news {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .confidence-text {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """Load trained model and tokenizer."""
    try:
        model = load_model('models/fake_news_model.h5')
        
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('models/config.pickle', 'rb') as f:
            config = pickle.load(f)
        
        return model, tokenizer, config
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please run 'python3 train_model.py' first to train the model.")
        st.stop()


def predict_news(text, model, tokenizer, max_length):
    """
    Predict if news is fake or real.
    
    Args:
        text (str): Input news text
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return None, 0.0
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    prediction_prob = model.predict(padded, verbose=0)[0][0]
    
    # Convert to label
    prediction = "Real" if prediction_prob > 0.5 else "Fake"
    confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
    
    return prediction, confidence


def create_gauge_chart(confidence, prediction):
    """Create a gauge chart for confidence visualization."""
    color = "#4CAF50" if prediction == "Real" else "#F44336"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    """Main application."""
    # Header
    st.markdown('<div class="main-header">📰 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Deep Learning | LSTM Neural Network</div>', 
                unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, config = load_model_and_tokenizer()
    
    max_length = config['max_sequence_length']
    
    # Sidebar
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        """
        This application uses a **Bidirectional LSTM** neural network 
        to classify news articles as Real or Fake.
        
        **Features:**
        - Single text prediction
        - Batch CSV processing
        - Confidence scores
        - Downloadable results
        
        **Model Performance:**
        - Accuracy: ~95%+
        - Dataset: 40,000+ articles
        """
    )
    
    st.sidebar.title("📊 How to Use")
    st.sidebar.markdown(
        """
        **Single Prediction:**
        1. Enter news headline/article in text area
        2. Click 'Analyze News'
        3. View prediction and confidence
        
        **Batch Processing:**
        1. Upload CSV with 'text' column
        2. Click 'Analyze CSV'
        3. Download results
        """
    )
    
    # Main content
    tab1, tab2 = st.tabs(["📝 Single Text", "📁 Batch CSV"])
    
    # Tab 1: Single text input
    with tab1:
        st.markdown("### Enter News Article")
        
        # Sample texts
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📌 Load Real News Sample"):
                st.session_state['input_text'] = (
                    "Scientists at MIT have developed a new breakthrough in renewable energy "
                    "technology that could reduce solar panel costs by 40%. The research, "
                    "published in Nature, demonstrates a novel manufacturing process."
                )
        
        with col2:
            if st.button("📌 Load Fake News Sample"):
                st.session_state['input_text'] = (
                    "BREAKING: Government admits aliens have been living among us for decades! "
                    "Secret documents reveal shocking truth. Click here to learn what they "
                    "don't want you to know!!!"
                )
        
        # Text input
        input_text = st.text_area(
            "Paste news headline or article:",
            value=st.session_state.get('input_text', ''),
            height=200,
            placeholder="Enter news text here..."
        )
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
        
        if analyze_button and input_text:
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_news(input_text, model, tokenizer, max_length)
                
                if prediction:
                    # Display result
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Result")
                    
                    # Result box
                    box_class = "real-news" if prediction == "Real" else "fake-news"
                    icon = "✅" if prediction == "Real" else "❌"
                    
                    st.markdown(
                        f'<div class="result-box {box_class}">'
                        f'<h2>{icon} {prediction} News</h2>'
                        f'<p class="confidence-text">{confidence * 100:.1f}% Confidence</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Gauge chart
                    st.plotly_chart(create_gauge_chart(confidence, prediction), 
                                    use_container_width=True)
                    
                    # Interpretation
                    st.markdown("### 💡 Interpretation")
                    if confidence > 0.9:
                        st.success(f"The model is **very confident** this news is {prediction.lower()}.")
                    elif confidence > 0.7:
                        st.info(f"The model is **fairly confident** this news is {prediction.lower()}.")
                    else:
                        st.warning(f"The model is **somewhat uncertain** but leans towards {prediction.lower()}.")
                else:
                    st.error("❌ Could not analyze text. Please enter valid news content.")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Tab 2: CSV upload
    with tab2:
        st.markdown("### Upload CSV File")
        st.info("📋 CSV must contain a column named **'text'** with news articles.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
                st.dataframe(df.head(), use_container_width=True)
                
                if 'text' not in df.columns:
                    st.error("❌ CSV must contain a 'text' column!")
                else:
                    if st.button("🔍 Analyze CSV", type="primary"):
                        with st.spinner(f"Analyzing {len(df)} articles..."):
                            predictions = []
                            confidences = []
                            
                            progress_bar = st.progress(0)
                            
                            for idx, text in enumerate(df['text']):
                                pred, conf = predict_news(str(text), model, tokenizer, max_length)
                                predictions.append(pred if pred else "Unknown")
                                confidences.append(conf)
                                progress_bar.progress((idx + 1) / len(df))
                            
                            # Add results to dataframe
                            df['prediction'] = predictions
                            df['confidence'] = [f"{c * 100:.2f}%" for c in confidences]
                            
                            st.success("✅ Analysis complete!")
                            
                            # Display results
                            st.markdown("### 📊 Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Real News", len(df[df['prediction'] == 'Real']))
                            with col2:
                                st.metric("Fake News", len(df[df['prediction'] == 'Fake']))
                            with col3:
                                st.metric("Avg Confidence", 
                                         f"{np.mean(confidences) * 100:.1f}%")
                            
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="fake_news_predictions.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing CSV: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit & TensorFlow | "
        "<a href='https://github.com' style='color: #1E88E5;'>View on GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    