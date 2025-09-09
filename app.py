import streamlit as st
import torch
import torch.nn as nn
import re
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
from datetime import datetime
import json
import os

# ==========================
# Configuration
# ==========================
@st.cache_data
def load_config():
    return {
        "max_vocab_size": 20000,
        "max_sequence_length": 256,
        "embed_dim": 128,
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "score_ranges": {
            "Excellent": (4.5, 6.0),
            "Good": (3.5, 4.5),
            "Average": (2.5, 3.5),
            "Below Average": (1.5, 2.5),
            "Poor": (0.0, 1.5)
        }
    }

# ==========================
# Enhanced Text Processing
# ==========================
class TextProcessor:
    def __init__(self, vocab=None):
        self.vocab = vocab
        
    @staticmethod
    def advanced_tokenize(text):
        """Enhanced tokenization with better text cleaning"""
        text = text.lower().strip()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r"[^a-z0-9\s.,!?;:]", "", text)
        # Remove standalone punctuation
        text = re.sub(r'\s+[.,!?;:]+\s+', ' ', text)
        return text.split()
    
    @staticmethod
    def extract_features(text):
        """Extract linguistic features from text"""
        tokens = TextProcessor.advanced_tokenize(text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features = {
            "word_count": len(tokens),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(tokens) / max(len(sentences), 1),
            "unique_words": len(set(tokens)),
            "vocabulary_richness": len(set(tokens)) / max(len(tokens), 1),
            "character_count": len(text),
            "avg_word_length": np.mean([len(word) for word in tokens]) if tokens else 0
        }
        return features

# ==========================
# FIXED Model Definition - Simplified to match your trained model
# ==========================
class EssayScoringLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout_rate=0.3):
        super(EssayScoringLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           bidirectional=True, num_layers=1, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Only one fully connected layer
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        # Global max pooling
        max_pooled = torch.max(lstm_out, dim=1)[0]
        x = self.dropout(max_pooled)
        out = self.fc(x)
        return out.squeeze(-1)

# ==========================
# Data Loading and Vocab Building
# ==========================
@st.cache_data
def build_vocabulary():
    """Build vocabulary from training data with caching"""
    try:
        if os.path.exists("Data/train.csv"):
            data = pd.read_csv("Data/train.csv")
            counter = Counter()
            
            processor = TextProcessor()
            for txt in data["full_text"].values:
                if pd.notna(txt):
                    counter.update(processor.advanced_tokenize(str(txt)))
            
            # Build vocab
            vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(20000))}
            vocab["<PAD>"] = 0
            vocab["<UNK>"] = 1
            
            return vocab, len(data)
        else:
            st.warning("Training data not found. Using default vocabulary.")
            return {"<PAD>": 0, "<UNK>": 1}, 0
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return {"<PAD>": 0, "<UNK>": 1}, 0

# ==========================
# Enhanced Prediction Functions
# ==========================
def encode_text(text, vocab, max_len=256):
    """Encode text with enhanced tokenization"""
    processor = TextProcessor(vocab)
    tokens = processor.advanced_tokenize(text)
    ids = [vocab.get(token, 1) for token in tokens]  # 1 = <UNK>
    
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    return ids, len([token for token in tokens if token in vocab])

def get_score_category(score, config):
    """Categorize score based on ranges"""
    for category, (min_score, max_score) in config["score_ranges"].items():
        if min_score <= score < max_score:
            return category
    return "Unclassified"

def get_feedback(score, features):
    """Generate detailed feedback based on score and features"""
    feedback = []
    
    if score >= 4.5:
        feedback.append("🎉 Excellent work! Your essay demonstrates strong writing skills.")
    elif score >= 3.5:
        feedback.append("👍 Good essay with solid structure and content.")
    elif score >= 2.5:
        feedback.append("👌 Average essay with room for improvement.")
    else:
        feedback.append("📝 Consider revising your essay to improve clarity and structure.")
    
    # Word count feedback
    if features["word_count"] < 100:
        feedback.append("• Consider expanding your essay - it seems quite short.")
    elif features["word_count"] > 800:
        feedback.append("• Your essay is quite lengthy - ensure all content is relevant.")
    
    # Sentence structure feedback
    if features["avg_words_per_sentence"] < 8:
        feedback.append("• Try using more complex sentence structures.")
    elif features["avg_words_per_sentence"] > 25:
        feedback.append("• Consider breaking down some longer sentences for clarity.")
    
    # Vocabulary feedback
    if features["vocabulary_richness"] < 0.4:
        feedback.append("• Try using more varied vocabulary to enhance your writing.")
    elif features["vocabulary_richness"] > 0.8:
        feedback.append("• Great vocabulary diversity!")
    
    return feedback

# ==========================
# Visualization Functions
# ==========================
def create_score_gauge(score):
    """Create a gauge chart for the score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Essay Score"},
        delta = {'reference': 3.0},
        gauge = {
            'axis': {'range': [None, 6]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "lightgray"},
                {'range': [2, 3.5], 'color': "yellow"},
                {'range': [3.5, 5], 'color': "lightgreen"},
                {'range': [5, 6], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5.5
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_features_radar(features):
    """Create radar chart for essay features"""
    # Normalize features for radar chart
    normalized_features = {
        'Word Count': min(features['word_count'] / 500, 1),
        'Sentences': min(features['sentence_count'] / 20, 1),
        'Avg Words/Sentence': min(features['avg_words_per_sentence'] / 20, 1),
        'Vocabulary Richness': features['vocabulary_richness'],
        'Avg Word Length': min(features['avg_word_length'] / 8, 1)
    }
    
    categories = list(normalized_features.keys())
    values = list(normalized_features.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Essay'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=400,
        title="Essay Feature Analysis"
    )
    
    return fig

# ==========================
# History Management
# ==========================
def save_prediction_history(essay_text, score, features):
    """Save prediction to history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction = {
        'timestamp': datetime.now().isoformat(),
        'essay_preview': essay_text[:100] + "..." if len(essay_text) > 100 else essay_text,
        'score': score,
        'word_count': features['word_count'],
        'sentence_count': features['sentence_count']
    }
    
    st.session_state.prediction_history.append(prediction)
    
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]

# ==========================
# Enhanced Model Loading Function
# ==========================
def load_model_safely(vocab_size, config):
    """Safely load the model with proper error handling"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with simplified architecture
        model = EssayScoringLSTM(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            dropout_rate=config["dropout_rate"]
        )
        
        if os.path.exists("essay_scoring_lstm_only.pt"):
            # Load the state dict
            checkpoint = torch.load("essay_scoring_lstm_only.pt", map_location=device)
            
            # Debug: Print model keys vs checkpoint keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(checkpoint.keys())
            
            st.write("Debug Info:")
            st.write(f"Model keys: {model_keys}")
            st.write(f"Checkpoint keys: {checkpoint_keys}")
            
            # Check for key mismatches
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            if missing_keys:
                st.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                st.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            # Try to load the state dict
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            model.eval()
            
            return model, device, None
        else:
            return None, device, "Model file not found"
            
    except Exception as e:
        return None, device, str(e)

# ==========================
# Main Streamlit App
# ==========================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Essay Scoring",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    config = load_config()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        margin-bottom: 2rem;
    }
    .score-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
    }
    .feature-box {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Advanced Essay Scoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        # Load vocabulary and model info
        vocab, training_samples = build_vocabulary()
        st.info(f"📚 Vocabulary Size: {len(vocab):,}")
        if training_samples > 0:
            st.info(f"📝 Training Samples: {training_samples:,}")
        
        # Model compatibility check
        model, device, error = load_model_safely(len(vocab), config)
        
        if error:
            st.error(f"⚠️ Model Error: {error}")
        elif model:
            st.success("✅ Model Ready (Simplified)")
            total_params = sum(p.numel() for p in model.parameters())
            st.info(f"🔧 Parameters: {total_params:,}")
        
        st.header("⚙️ Settings")
        show_advanced = st.checkbox("Show Advanced Features", value=True)
        show_history = st.checkbox("Show Prediction History", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
        
        st.header("📋 Score Guide")
        for category, (min_score, max_score) in config["score_ranges"].items():
            st.write(f"**{category}**: {min_score}-{max_score}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Essay Input")
        essay_text = st.text_area(
            "Enter your essay here:",
            height=300,
            placeholder="Start writing your essay here...",
            help="Write your essay and click 'Analyze Essay' to get detailed scoring and feedback."
        )
        
        # Analysis buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Essay", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("📋 Load Sample", use_container_width=True):
                sample_essay = """
                Education plays a crucial role in shaping our future and society. It provides us with knowledge, skills, and values that are essential for personal growth and social development. Through education, we learn to think critically, solve problems, and communicate effectively.

                Furthermore, education opens doors to better career opportunities and helps reduce poverty and inequality. It empowers individuals to make informed decisions and contributes to economic growth and innovation.

                In conclusion, investing in education is investing in our future. We must ensure that quality education is accessible to all, regardless of their background or circumstances.
                """
                st.session_state.sample_essay = sample_essay
                st.rerun()
        
        with col_btn3:
            if st.button("🗑️ Clear Text", use_container_width=True):
                st.session_state.clear_text = True
                st.rerun()
        
        # Handle sample essay loading
        if hasattr(st.session_state, 'sample_essay'):
            essay_text = st.session_state.sample_essay
            del st.session_state.sample_essay
        
        # Handle text clearing
        if hasattr(st.session_state, 'clear_text'):
            essay_text = ""
            del st.session_state.clear_text
    
    with col2:
        st.header("📊 Quick Stats")
        if essay_text.strip():
            features = TextProcessor.extract_features(essay_text)
            
            st.metric("Word Count", features["word_count"])
            st.metric("Sentences", features["sentence_count"])
            st.metric("Avg Words/Sentence", f"{features['avg_words_per_sentence']:.1f}")
            st.metric("Vocabulary Richness", f"{features['vocabulary_richness']:.2f}")
        else:
            st.info("Enter text to see statistics")
    
    # Analysis Results
    if analyze_btn and essay_text.strip():
        if not model:
            st.error("Model is not loaded. Cannot perform analysis.")
            return
            
        with st.spinner("Analyzing your essay..."):
            try:
                # Encode text
                ids, vocab_coverage = encode_text(essay_text, vocab, config["max_sequence_length"])
                input_ids = torch.tensor([ids], dtype=torch.long).to(device)
                
                # Get prediction
                with torch.no_grad():
                    score = model(input_ids).item()
                
                # Ensure score is in valid range
                score = max(0.0, min(6.0, score))
                
                # Extract features
                features = TextProcessor.extract_features(essay_text)
                
                # Save to history
                save_prediction_history(essay_text, score, features)
                
                # Display results
                st.markdown("---")
                st.header("🎯 Analysis Results")
                
                # Main score display
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    category = get_score_category(score, config)
                    st.markdown(f'<div class="score-container">', unsafe_allow_html=True)
                    st.metric(
                        label="Essay Score",
                        value=f"{score:.2f}/6.0",
                        delta=f"{category}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualizations
                if show_advanced:
                    st.subheader("📈 Visual Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        gauge_fig = create_score_gauge(score)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col2:
                        radar_fig = create_features_radar(features)
                        st.plotly_chart(radar_fig, use_container_width=True)
                
                # Detailed feedback
                st.subheader("💡 Detailed Feedback")
                feedback_list = get_feedback(score, features)
                
                for feedback in feedback_list:
                    st.write(feedback)
                
                # Technical details
                if show_advanced:
                    st.subheader("🔧 Technical Details")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                        st.write("**Text Processing**")
                        st.write(f"Vocabulary Coverage: {(vocab_coverage/len(TextProcessor.advanced_tokenize(essay_text)))*100:.1f}%")
                        st.write(f"Sequence Length: {len(ids)} tokens")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                        st.write("**Model Info**")
                        st.write(f"Device: {device}")
                        st.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                        st.write("**Features**")
                        st.write(f"Character Count: {features['character_count']}")
                        st.write(f"Avg Word Length: {features['avg_word_length']:.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Debug information
                if show_debug:
                    st.subheader("🐛 Debug Information")
                    st.write(f"Raw prediction: {score}")
                    st.write(f"Input tensor shape: {input_ids.shape}")
                    st.write(f"First 10 token IDs: {ids[:10]}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if show_debug:
                    import traceback
                    st.code(traceback.format_exc())
    
    elif analyze_btn:
        st.warning("⚠️ Please enter some text before analyzing.")
    
    # Prediction History
    if show_history and 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.header("📚 Prediction History")
        
        # Create DataFrame from history
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        if not history_df.empty:
            # Show recent predictions
            st.subheader("Recent Predictions")
            for i, row in history_df.tail(5).iterrows():
                with st.expander(f"Score: {row['score']:.2f} - {row['timestamp'][:16]}"):
                    st.write(f"**Essay Preview**: {row['essay_preview']}")
                    st.write(f"**Word Count**: {row['word_count']}")
                    st.write(f"**Sentences**: {row['sentence_count']}")
            
            # Score distribution
            if len(history_df) > 1:
                st.subheader("Score Distribution")
                fig = px.histogram(history_df, x='score', bins=20, title='Distribution of Essay Scores')
                st.plotly_chart(fig, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

if __name__ == "__main__":
    main()