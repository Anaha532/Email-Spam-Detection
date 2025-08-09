import streamlit as st
import pickle
import math
import numpy as np
import re
import os
from collections import Counter

# Load model
@st.cache_resource
def load_model():
    with open('model/spam_detection_model.pkl', 'rb') as f:
        model_info = pickle.load(f)
    return model_info

# Functions needed for inference
def mean(fv):
    return float(sum(fv)) / len(fv)

def bernoulli_random_variable(th):
    return lambda x: th < x

def bernoulli_distribution(fv, rv):
    successCt = sum([rv(f) for f in fv])
    prSuccess = float(successCt + 1) / (len(fv) + 2)  # Laplace smoothing
    prFailure = float(len(fv) - successCt + 1) / (len(fv) + 2)  # Laplace smoothing
    return lambda x: prSuccess if rv(x) else prFailure

def bernoulli_model(fv, th=None):
    rv = bernoulli_random_variable(mean(fv) if th is None else th)
    return bernoulli_distribution(fv, rv)

def naivebayes(testing, training, model_fn):
    # Split training data by label
    T = []  # Spam emails (label=1)
    F = []  # Non-spam emails (label=0)
    
    for dp in training:
        (T if dp['label'] else F).append(dp['features'])
    
    # Calculate prior probabilities and log-odds
    Tprior = float(len(T)) / len(training)
    Fprior = float(len(F)) / len(training)
    priorlogodds = math.log(Tprior / Fprior)
    
    # Generate models for each feature
    Tm = []  # Models for spam class
    Fm = []  # Models for non-spam class
    
    # zip(*seq) rotates data points to feature vectors
    for Tfv, Ffv in zip(zip(*T), zip(*F)):
        try:
            tm = model_fn(Tfv)
            fm = model_fn(Ffv)
        except Exception:  # Handle zero variance cases
            tm = fm = lambda x: 1.0
        
        Tm.append(tm)
        Fm.append(fm)
    
    # Copy testing data and calculate log-odds for each test point
    ret = [dp.copy() for dp in testing]
    
    # Accumulate log-odds over the features of each data point
    for dp in ret:
        dp['logodds'] = priorlogodds
        for idx, val in enumerate(dp['features']):
            gT = Tm[idx](val)
            gF = Fm[idx](val)
            if gT != 0.0 and gF != 0.0:
                dp['logodds'] += math.log(gT / gF)
    
    return ret

def predict_spam_probability(email_features, model_info):
    # Prepare training data in the format needed by naivebayes function
    X_train = model_info['training_data']['X']
    y_train = model_info['training_data']['y']
    
    training_data = [{'features': X_train[i].tolist(), 'label': y_train[i]} 
                    for i in range(len(y_train))]
    
    # Prepare the test data (just one sample)
    test_data = [{'features': email_features, 'label': 0}]  # label doesn't matter for prediction
    
    # Apply Naive Bayes
    result = naivebayes(test_data, training_data, bernoulli_model)
    
    # Convert log-odds to probability: P(spam) = 1 / (1 + e^(-log_odds))
    log_odds = result[0]['logodds']
    probability = 1 / (1 + math.exp(-log_odds))
    
    return probability

def extract_features(email_text, feature_names):
    """Extract features from email text to match the model's expected input format."""
    # Initialize features with zeros
    features = [0.0] * len(feature_names)
    
    # Lowercase the email text
    email_text = email_text.lower()
    total_chars = len(email_text) if len(email_text) > 0 else 1  # Avoid division by zero
    
    # Count words in the text
    words = re.findall(r'\b\w+\b', email_text)
    word_count = len(words) if len(words) > 0 else 1  # Avoid division by zero
    
    # Count occurrences of each word
    word_counts = Counter(words)
    
    # Process each feature
    for i, feature_name in enumerate(feature_names):
        if feature_name.startswith('word_freq_'):
            # Word frequency features
            word = feature_name.replace('word_freq_', '')
            # Calculate as percentage of occurrences (0-100)
            features[i] = (word_counts.get(word, 0) / word_count) * 100
            
        elif feature_name.startswith('char_freq_'):
            # Character frequency features
            char_type = feature_name.replace('char_freq_', '')
            char_map = {
                'semicolon': ';',
                'leftparen': '(',
                'leftbracket': '[',
                'exclamation': '!',
                'dollar': '$',
                'hash': '#'
            }
            char = char_map.get(char_type.lower())
            if char:
                # Calculate as percentage of characters
                features[i] = (email_text.count(char) / total_chars) * 100
                
        elif feature_name.startswith('capital_run_length_'):
            # Capital run length features
            capital_runs = re.findall(r'[A-Z]+', email_text)
            if not capital_runs:
                features[i] = 0
            elif feature_name == 'capital_run_length_avg':
                features[i] = sum(len(run) for run in capital_runs) / len(capital_runs)
            elif feature_name == 'capital_run_length_longest':
                features[i] = max(len(run) for run in capital_runs) if capital_runs else 0
            elif feature_name == 'capital_run_length_total':
                features[i] = sum(len(run) for run in capital_runs)
    
    return features

# Streamlit app
st.title('📧 Email Spam Classifier')
st.write('Enter email content below and the app will classify it using Bernoulli Naive Bayes')

# Text input
email_text = st.text_area('Email Content', height=200)

# Submit button
if st.button('Classify Email'):
    if email_text:
        with st.spinner('Analyzing email...'):
            try:
                # Load the model
                model_info = load_model()
                
                # Extract features
                features = extract_features(email_text, model_info['feature_names'])
                
                # Make prediction
                probability = predict_spam_probability(features, model_info)
                
                # Display result with probability gauge
                st.write(f"### Spam Probability: {probability:.2f}")
                
                # Create a progress bar for the probability
                st.progress(float(probability))
                
                # Show the classification result
                if probability > 0.5:
                    st.error("🚨 This email is classified as SPAM!")
                else:
                    st.success("✅ This email is classified as NOT SPAM")
                
                # Show top contributing features
                if st.checkbox('Show feature details'):
                    feature_values = list(zip(model_info['feature_names'], features))
                    feature_values.sort(key=lambda x: x[1], reverse=True)
                    
                    st.write("### Top Features")
                    for name, value in feature_values[:10]:
                        if value > 0:
                            st.write(f"- {name}: {value:.2f}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter some email content to classify")

# Add some info about the model
with st.expander("About the model"):
    st.write("""
    This app uses a Bernoulli Naive Bayes model trained on the Spambase dataset.
    
    The model analyzes:
    - Word frequencies (e.g., 'free', 'money', 'business')
    - Character frequencies (e.g., '!', '$', '#')
    - Capital letter patterns
    
    Based on these features, it calculates the probability of an email being spam.
    """)