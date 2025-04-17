import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import time
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
import random
import traceback
import yfinance as yf
from news_sentiment import calculate_news_sentiment_score, calculate_sentiment_adjustment
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_logging():
    """Configure logging settings for the application."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # Also create a file handler
    file_handler = logging.FileHandler("stock_prediction.log")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Set up logging
logger = setup_logging()

# Constants
TICKER = "NVDA"               # Stock ticker symbol
START_DATE = "2010-01-01"     # Start date for training data
END_DATE = "2025-04-04"       # End date for training data
SEQUENCE_LENGTH = 60          # Number of time steps in each sequence
BATCH_SIZE = 64               # Batch size for training

# --- Constants ---
TICKER = 'NVDA'
START_DATE = '2010-01-01'
END_DATE = '2025-04-04'  # Updated to include recent data through April 2025
SEQUENCE_LENGTH = 60  # Number of past days' data to use for predicting the next day
SPLIT_RATIO = 0.8 # 80% train, 20% test
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI_14', 
                  'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'] # Added Bollinger Bands
TARGET_COLUMN = 'Close' # Column to predict
USE_NEWS_SENTIMENT = True  # Whether to incorporate news sentiment
NEWS_SENTIMENT_WEIGHT = 0.3  # Weight of news sentiment (0-1)
PREDICT_LIVE = True  # Whether to predict live stock prices
DAYS_TO_PREDICT = 5  # Number of future days to predict
OLLAMA_MODEL = "llama2"  # Specify Llama2 as the model to use
SCALERS_FILE = f"{TICKER}_scalers.pkl"  # File to save scalers

# --- Hyperparameters ---
INITIAL_LR = 0.001
MIN_LR = 0.0001
LR_PATIENCE = 5
LR_FACTOR = 0.5
NUM_EPOCHS = 50
HIDDEN_LSTM = 64
HIDDEN_DENSE = 32
DROPOUT_RATE = 0.2
USE_BIDIRECTIONAL = False
USE_ATTENTION = False
WEIGHT_DECAY = 0

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Data Fetching ---
def fetch_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance for the given ticker and date range."""
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        
        # Download data from Yahoo Finance with auto_adjust=True to account for splits and dividends
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        # Check if data is non-empty
        if df.empty:
            logger.error(f"No data found for {ticker} between {start_date} and {end_date}")
            return None
        
        # Get stock split information
        stock = yf.Ticker(ticker)
        splits = stock.splits
        
        if not splits.empty:
            logger.info(f"Found {len(splits)} stock splits for {ticker}:")
            for date, ratio in splits.items():
                logger.info(f"  {date.strftime('%Y-%m-%d')}: {ratio}")
            logger.info("Using adjusted prices to account for splits")
        else:
            logger.info(f"No stock splits found for {ticker} in the given period")
        
        # Log data info
        logger.info(f"Fetched {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        traceback.print_exc()
        return None

def fetch_and_preprocess_data(ticker, start_date, end_date):
    """Fetch and preprocess stock data for the given ticker and date range."""
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = fetch_data(ticker, start_date, end_date)
    
    if df is None:
        return None
    
    # Basic preprocessing steps
    # Fill missing values if any
    df = df.fillna(method='ffill')
    
    return df

# --- 2. Add Technical Indicators ---
def add_technical_indicators(data):
    """Adds technical indicators to the data."""
    print("Adding technical indicators...")
    
    # Add Simple Moving Average (SMA) with 20-day window
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Add Relative Strength Index (RSI) with 14-day window
    # Custom RSI implementation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Add Bollinger Bands using numpy operations
    # Create temporary column for standard deviation
    data['StdDev_20'] = data['Close'].rolling(window=20).std()
    
    # Now calculate BB components using these columns to avoid alignment issues
    data['BB_Upper'] = data['SMA_20'] + 2 * data['StdDev_20']
    data['BB_Lower'] = data['SMA_20'] - 2 * data['StdDev_20']
    data['BB_Middle'] = data['SMA_20']  # Already calculated above
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # Drop the temporary column
    data = data.drop('StdDev_20', axis=1)
    
    # There will be NaN values for the first few days because indicators
    # need historical data to calculate. Handle by dropping NaN rows.
    data = data.dropna()
    print(f"Added indicators. Data shape: {data.shape}")
    return data

# --- News Sentiment Integration ---
def add_news_sentiment(predictions, dates, test_only=True):
    """
    Adjust predictions based on news sentiment
    
    Parameters:
        predictions (numpy.ndarray): Model predictions
        dates (pandas.DatetimeIndex): Dates corresponding to predictions
        test_only (bool): Whether to only adjust test data predictions
        
    Returns:
        numpy.ndarray: Adjusted predictions
    """
    if not USE_NEWS_SENTIMENT:
        return predictions
    
    logger.info("Integrating news sentiment into predictions...")
    
    # Get the latest news sentiment score
    sentiment_score, articles = news_sentiment.calculate_news_sentiment_score(max_articles=5)
    
    # If we couldn't get a sentiment score, return predictions as is
    if sentiment_score == 0 or not articles:
        logger.warning("No news sentiment available, using original predictions")
        return predictions
    
    logger.info(f"News sentiment score: {sentiment_score:.4f} (based on {len(articles)} articles)")
    
    # Print the articles and their sentiment
    for article in articles:
        logger.info(f"Article: {article['title'][:50]}... - {article['sentiment']} ({article['score']})")
    
    # Adjust predictions based on sentiment
    # We'll use a simple approach - multiply the predictions by a factor based on sentiment
    # Positive sentiment increases predictions, negative sentiment decreases them
    
    # Calculate adjustment factor (between 1-NEWS_SENTIMENT_WEIGHT and 1+NEWS_SENTIMENT_WEIGHT)
    adjustment_factor = 1.0 + (sentiment_score * NEWS_SENTIMENT_WEIGHT)
    
    logger.info(f"Applying adjustment factor: {adjustment_factor:.4f}")
    
    # Apply the adjustment to the most recent predictions (last 5 days)
    adjusted_predictions = predictions.copy()
    
    # If test_only is True, only adjust the last part of the predictions (test set)
    if test_only:
        # Adjust the last 5 days of predictions or all test data if less than 5 days
        num_days_to_adjust = min(5, len(adjusted_predictions))
        adjusted_predictions[-num_days_to_adjust:] *= adjustment_factor
    else:
        # Adjust all predictions
        adjusted_predictions *= adjustment_factor
    
    return adjusted_predictions

# --- 3. Data Preprocessing ---
def preprocess_data(data, features, target_col, sequence_length):
    """Preprocesses data: selects features, scales, and creates sequences."""
    print("Preprocessing data...")
    data_filtered = data[features].copy()

    # Scale features
    scalers = {}
    scaled_data = pd.DataFrame(index=data_filtered.index)
    for col in features:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_column = scaler.fit_transform(data_filtered[[col]])
        scaled_data[col] = scaled_column
        scalers[col] = scaler
    
    # Save the scalers to file
    with open(SCALERS_FILE, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {SCALERS_FILE}")

    # Create sequences
    X, y = [], []
    scaled_np_data = scaled_data.values
    target_col_index = features.index(target_col)
    for i in range(sequence_length, len(scaled_np_data)):
        X.append(scaled_np_data[i-sequence_length:i])
        y.append(scaled_np_data[i, target_col_index])

    X, y = np.array(X), np.array(y)
    # Reshape y to be [samples, 1]
    y = y.reshape(-1, 1)

    print(f"Data preprocessed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scalers

# --- Attention Mechanism Implementation ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

# --- Neural Network Model ---
class StockPredictorNet(nn.Module):
    """
    Neural network for stock price prediction.
    Uses LSTM layers followed by dense layers.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(StockPredictorNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine the size of the output from LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM output
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: [batch, seq_len, hidden_dim]
        
        # We only need the last output
        lstm_out = lstm_out[:, -1, :]  # lstm_out shape: [batch, hidden_dim]
        
        # Dense layers
        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Output layer
        out = self.fc2(fc1_out)
        
        return out

def prepare_data(df):
    """Prepare data for the model, including feature engineering and scaling."""
    logger.info("Preprocessing data...")
    
    # Make a copy of the dataframe to avoid warnings
    df = df.copy()
    
    # First, ensure we have all the necessary features
    # We need all FEATURE_COLUMNS to be present in df
    if not all(col in df.columns for col in FEATURE_COLUMNS):
        logger.error(f"Missing some required columns. Required: {FEATURE_COLUMNS}")
        logger.error(f"Available: {df.columns.tolist()}")
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        logger.error(f"Missing: {missing}")
        return None, None, None, None, None, None
    
    # Make sure df at least has the TARGET_COLUMN
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Missing target column: {TARGET_COLUMN}")
        return None, None, None, None, None, None
    
    # Calculate daily percentage changes for the target variable
    df['Pct_Change'] = df[TARGET_COLUMN].pct_change() * 100  # Convert to percentage
    
    # Drop rows with NaN values (including the first row with NaN percentage change)
    df = df.dropna()
    logger.info(f"After dropping NaN values, data shape: {df.shape}")
    
    # Store the original Close prices for later reference
    original_prices = df[TARGET_COLUMN].values
    
    # Extract features and percentage change target
    X_raw = df[FEATURE_COLUMNS].values
    y_raw = df['Pct_Change'].values.reshape(-1, 1)
    
    # Create scalers
    feature_scaler = MinMaxScaler()
    pct_change_scaler = MinMaxScaler(feature_range=(-1, 1))  # Better range for percentage changes
    price_scaler = MinMaxScaler()
    
    # Scale the features
    X_scaled = feature_scaler.fit_transform(X_raw)
    
    # Scale percentage changes differently - they can be positive or negative
    y_scaled = pct_change_scaler.fit_transform(y_raw)
    
    # Fit a price scaler with the original prices for later use
    close_prices = df[TARGET_COLUMN].values.reshape(-1, 1)
    price_scaler.fit(close_prices)
    
    # Record the min and max of actual percentage changes for reference
    min_pct_change = np.min(y_raw)
    max_pct_change = np.max(y_raw)
    logger.info(f"Percentage change range: {min_pct_change:.2f}% to {max_pct_change:.2f}%")
    
    # Create sequences
    X, y = [], []
    dates = []
    
    for i in range(len(df) - SEQUENCE_LENGTH):
        X.append(X_scaled[i:i+SEQUENCE_LENGTH])
        y.append(y_scaled[i+SEQUENCE_LENGTH])
        dates.append(df.index[i+SEQUENCE_LENGTH])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Data preprocessed. X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, dates, feature_scaler, pct_change_scaler, price_scaler

def split_data(X, y, dates):
    """Split the data into training, validation, and test sets."""
    # Calculate split indices
    train_end = int(len(X) * 0.7)  # 70% for training
    val_end = int(len(X) * 0.8)    # 10% for validation, 20% for testing
    
    # Split the data
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    # Split the dates
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates

def train_model(ticker):
    """Train the model or load existing model."""
    logger.info(f"Training/loading model for {ticker}")
    
    # Fetch and preprocess data
    df = fetch_and_preprocess_data(ticker, START_DATE, END_DATE)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Prepare data for model
    X, y, dates, feature_scaler, pct_change_scaler, price_scaler = prepare_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates = split_data(X, y, dates)
    
    # Training parameters
    input_dim = X_train.shape[2]  # Number of features
    
    # Initialize model
    model = StockPredictorNet(input_dim=input_dim, hidden_dim=HIDDEN_LSTM)
    model.to(device)
    
    model_path = f"{ticker}_model.pth"
    scalers_path = f"{ticker}_scalers.pkl"
    
    if os.path.exists(model_path):
        # Load the saved model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"Loaded model from {model_path}")
    else:
        # Train the model
        train_losses, val_losses = train_model_loop(model, X_train, y_train, X_val, y_val)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate the model
        test_results, metrics = evaluate_model(model, X_test, y_test, test_dates, pct_change_scaler)
    
    # Save scalers
    scalers = {
        'feature_scaler': feature_scaler,
        'pct_change_scaler': pct_change_scaler,
        'price_scaler': price_scaler
    }
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers saved to {scalers_path}")
    
    return model

def generate_predictions(ticker, current_price):
    """Generate predictions using the trained model."""
    logger.info(f"Generating predictions for {ticker}")
    
    # Load the model
    model_path = f"{ticker}_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Load scalers
    scalers_path = f"{ticker}_scalers.pkl"
    if not os.path.exists(scalers_path):
        logger.error(f"Scalers file not found: {scalers_path}")
        return
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Initialize model
    input_dim = None  # This will be set when we get the features
    model = None
    
    try:
        # Process live data
        live_data = process_live_data(ticker, current_price, scalers)
        
        if live_data is None:
            logger.error("Failed to process live data")
            return
        
        input_dim = live_data.shape[2]
        model = StockPredictorNet(input_dim=input_dim, hidden_dim=HIDDEN_LSTM)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        # Generate predictions
        with torch.no_grad():
            live_data_tensor = torch.FloatTensor(live_data).to(device)
            predictions = model(live_data_tensor)
            
            # Convert predictions back to price changes
            predictions_np = predictions.cpu().numpy()
            price_changes = scalers['pct_change_scaler'].inverse_transform(predictions_np)
            
            # Calculate predicted prices
            predicted_prices = []
            last_price = current_price
            
            for change in price_changes[0]:
                next_price = last_price * (1 + change)
                predicted_prices.append(next_price)
                last_price = next_price
            
            # Generate explanation
            explanation = generate_prediction_explanation(predicted_prices, ticker, current_price)
    
            # Log results
            logger.info(f"Current price: ${current_price:.2f}")
            for i, price in enumerate(predicted_prices, 1):
                logger.info(f"Day {i} prediction: ${price:.2f}")
            logger.info(f"Explanation: {explanation}")
            
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise

def process_live_data(ticker, current_price, scalers):
    """Process live data for prediction."""
    try:
        # Get historical data for technical indicators
        df = fetch_and_preprocess_data(ticker, START_DATE, END_DATE)
        df = add_technical_indicators(df)
        
        # Get the most recent SEQUENCE_LENGTH days of data
        recent_data = df.tail(SEQUENCE_LENGTH).copy()
    
        # Update the last price with current price
        recent_data.iloc[-1]['Close'] = current_price
        
        # Scale the features
        feature_columns = [col for col in recent_data.columns if col not in ['Date']]
        scaled_features = scalers['feature_scaler'].transform(recent_data[feature_columns])
        
        # Reshape for LSTM (batch_size, sequence_length, n_features)
        X = scaled_features.reshape(1, SEQUENCE_LENGTH, -1)
        
        return X
        
    except Exception as e:
        logger.error(f"Error processing live data: {str(e)}")
        return None

def train_model_loop(model, X_train, y_train, X_val, y_val):
    """Train the model on the training data and validate on validation data."""
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = LR_PATIENCE
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Convert numpy arrays to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, test_dates, pct_change_scaler):
    """Evaluate the model on the test data."""
    logger.info("Evaluating model on test data...")
    
    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create data loader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    # Concatenate batches
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # Inverse transform to get actual percentage changes
    predictions_pct = pct_change_scaler.inverse_transform(predictions)
    actuals_pct = pct_change_scaler.inverse_transform(actuals)
    
    # Calculate metrics for percentage changes
    mse = np.mean((predictions_pct - actuals_pct) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_pct - actuals_pct))
    
    logger.info("--- Evaluation Metrics (Test Set) ---")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}%")  # Now in percentage points
    logger.info(f"Mean Absolute Error (MAE):      {mae:.2f}%")   # Now in percentage points
    
    # Create a DataFrame with results
    results = pd.DataFrame(index=test_dates[-len(predictions):])
    results['Predicted_Pct_Change'] = predictions_pct.flatten()
    results['Actual_Pct_Change'] = actuals_pct.flatten()
    results['Error'] = results['Predicted_Pct_Change'] - results['Actual_Pct_Change']
    
    # If we have access to the original prices in the test set, we can convert percentage changes
    # to actual prices for additional context, but this would require storing prices during data prep
    
    # Return the results and metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    return results, metrics

# --- Function to explain results using Ollama ---
def explain_predictions_with_ollama(predictions_df, test_results=None, metrics=None):
    """
    Use Ollama with Llama2 to generate an explanation of the predictions
    
    Parameters:
        predictions_df: DataFrame with future predictions
        test_results: DataFrame with test set predictions and actuals
        metrics: Dictionary with model metrics (RMSE, MAE)
        
    Returns:
        str: Explanation text
    """
    logger.info("Explaining prediction results using Ollama with llama2...")
    
    # First check if Ollama is available
    try:
        import ollama
        try:
            # Quick test to see if Ollama is available
            ollama.list()
            logger.info("Ollama is available, proceeding with LLM-based explanation")
            ollama_available = True
        except Exception as e:
            logger.warning(f"Ollama is not available: {str(e)}")
            logger.info("Falling back to rule-based explanation generation")
            ollama_available = False
            return generate_fallback_explanation(predictions_df, metrics)
    except ImportError:
        logger.warning("Ollama package is not installed")
        logger.info("Falling back to rule-based explanation generation")
        return generate_fallback_explanation(predictions_df, metrics)
        
    if not ollama_available:
        return generate_fallback_explanation(predictions_df, metrics)
    
    # Create prompt based on available information
    prompt_parts = [
        f"You are a financial analyst specializing in NVIDIA stock analysis and prediction.",
        f"Below is data from our stock prediction model for NVIDIA (NVDA):"
    ]
    
    # Add model metrics if available
    if metrics:
        prompt_parts.append(f"\nModel Performance Metrics:")
        prompt_parts.append(f"- Root Mean Squared Error (RMSE): ${metrics['rmse']:.2f}")
        prompt_parts.append(f"- Mean Absolute Error (MAE): ${metrics['mae']:.2f}")
    
    # Add future predictions
    prompt_parts.append(f"\nFuture Price Predictions for NVIDIA:")
    
    for date, row in predictions_df.iterrows():
        # Ensure price is a scalar, not a Series
        price = row['Predicted_Price']
        if isinstance(price, pd.Series):
            price = price.iloc[0]
            
        # Handle change and percent change
        change = row.get('Change', 'N/A')
        pct_change = row.get('Pct_Change', 'N/A')
        
        # Convert Series to scalar if needed
        if isinstance(change, pd.Series):
            change = change.iloc[0]
        if isinstance(pct_change, pd.Series):
            pct_change = pct_change.iloc[0]
        
        if isinstance(change, float) and isinstance(pct_change, float):
            change_text = f"${change:.2f} ({pct_change:.2f}%)"
        else:
            change_text = "N/A"
            
        prompt_parts.append(f"- {date.strftime('%Y-%m-%d')}: ${price:.2f} (Change: {change_text})")
    
    # Add information about news sentiment if used
    if USE_NEWS_SENTIMENT:
        prompt_parts.append(f"\nThese predictions incorporate news sentiment with a weight of {NEWS_SENTIMENT_WEIGHT}.")
        
    # Request for analysis
    prompt_parts.append(f"\nBased on this information, please provide:")
    prompt_parts.append(f"1. A concise summary of the predicted price trend")
    prompt_parts.append(f"2. Key factors that might be influencing NVIDIA's stock price")
    prompt_parts.append(f"3. An assessment of the model's reliability based on its metrics (if provided)")
    prompt_parts.append(f"4. Any important caveats investors should keep in mind")
    prompt_parts.append(f"\nKeep your response informative but concise (under 400 words).")
    
    prompt = "\n".join(prompt_parts)
    
    try:
        # Call Ollama API with Llama2 model
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and return the explanation
        explanation = response['message']['content']
        return explanation
        
    except Exception as e:
        logger.error(f"Error calling Ollama API for explanation: {str(e)}")
        
        # Generate a simple fallback explanation
        fallback_explanation = generate_fallback_explanation(predictions_df, metrics)
        return fallback_explanation

def generate_fallback_explanation(predictions_df, metrics=None):
    """
    Generate a simple explanation of the predictions without using Ollama
    
    Parameters:
        predictions_df: DataFrame with future predictions
        metrics: Dictionary with model metrics (RMSE, MAE)
        
    Returns:
        str: Simple explanation text
    """
    explanation_parts = ["## NVIDIA Stock Price Prediction Analysis\n"]
    
    # Add model performance if available
    if metrics:
        explanation_parts.append(f"### Model Performance Metrics")
        explanation_parts.append(f"- Root Mean Squared Error (RMSE): ${metrics['rmse']:.2f}")
        explanation_parts.append(f"- Mean Absolute Error (MAE): ${metrics['mae']:.2f}")
        
        # Add interpretation of metrics
        if metrics['rmse'] < 5:
            explanation_parts.append("The model shows relatively low error rates, suggesting reasonable accuracy in predictions.")
        else:
            explanation_parts.append("The model shows higher error rates, suggesting predictions should be viewed with caution.")
    
    # Add prediction trend analysis
    explanation_parts.append(f"\n### Price Trend Analysis")
    
    # Calculate overall trend
    if len(predictions_df) > 1:
        # Get first price, ensuring it's a scalar
        first_price = predictions_df.iloc[0]['Predicted_Price']
        if isinstance(first_price, pd.Series):
            first_price = first_price.iloc[0]
            
        # Get last price, ensuring it's a scalar
        last_price = predictions_df.iloc[-1]['Predicted_Price']
        if isinstance(last_price, pd.Series):
            last_price = last_price.iloc[0]
            
        overall_change = last_price - first_price
        overall_pct_change = (overall_change / first_price) * 100
        
        # Describe the trend
        if overall_change > 0:
            trend = "upward"
        elif overall_change < 0:
            trend = "downward"
        else:
            trend = "flat"
            
        explanation_parts.append(f"The model predicts an overall {trend} trend for NVIDIA stock over the next {len(predictions_df)} trading days.")
        explanation_parts.append(f"Starting price: ${first_price:.2f}")
        explanation_parts.append(f"Ending price: ${last_price:.2f}")
        explanation_parts.append(f"Overall change: ${overall_change:.2f} ({overall_pct_change:.2f}%)")
        
        # Add day-by-day analysis
        explanation_parts.append(f"\n### Day-by-Day Predictions")
        for date, row in predictions_df.iterrows():
            # Ensure values are scalars, not Series
            price = row['Predicted_Price']
            if isinstance(price, pd.Series):
                price = price.iloc[0]
                
            change = row.get('Change', 0)
            if isinstance(change, pd.Series):
                change = change.iloc[0]
                
            pct_change = row.get('Pct_Change', 0)
            if isinstance(pct_change, pd.Series):
                pct_change = pct_change.iloc[0]
            
            # Format the change if it's a float
            if isinstance(change, float) and isinstance(pct_change, float):
                if change > 0:
                    change_text = f"increase of ${change:.2f} ({pct_change:.2f}%)"
                elif change < 0:
                    change_text = f"decrease of ${-change:.2f} ({-pct_change:.2f}%)"
                else:
                    change_text = "no change"
                    
                explanation_parts.append(f"- {date.strftime('%Y-%m-%d')}: ${price:.2f} ({change_text})")
            else:
                explanation_parts.append(f"- {date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    # Add news sentiment note if used
    if USE_NEWS_SENTIMENT:
        explanation_parts.append(f"\n### News Sentiment Impact")
        explanation_parts.append(f"These predictions incorporate news sentiment with a weight of {NEWS_SENTIMENT_WEIGHT}.")
        explanation_parts.append(f"Recent news articles about NVIDIA have been considered in these predictions.")
    
    # Add caveats
    explanation_parts.append(f"\n### Important Caveats")
    explanation_parts.append(f"- Stock price predictions are inherently uncertain and should not be the sole basis for investment decisions.")
    explanation_parts.append(f"- The model cannot account for unexpected news, market shocks, or other unforeseen events.")
    explanation_parts.append(f"- Past performance is not indicative of future results.")
    explanation_parts.append(f"- Consider consulting with a financial advisor before making investment decisions.")
    
    return "\n".join(explanation_parts)

def generate_prediction_explanation(predicted_prices, ticker, current_price):
    """Generate an explanation of the predictions."""
    logger.info("Generating prediction explanation...")
    
    # Calculate changes
    changes = []
    pct_changes = []
    last_price = current_price
    
    for price in predicted_prices:
        change = price - last_price
        pct_change = (change / last_price) * 100
        changes.append(change)
        pct_changes.append(pct_change)
        last_price = price
    
    # Calculate overall change
    overall_change = predicted_prices[-1] - current_price
    overall_pct_change = (overall_change / current_price) * 100
    
    # Determine direction
    if overall_change > 0:
        direction = "upward"
    elif overall_change < 0:
        direction = "downward"
    else:
        direction = "flat"
    
    # Create explanation parts
    explanation_parts = []
    explanation_parts.append(f"# {ticker} Stock Price Prediction Analysis")
    explanation_parts.append(f"\n## Summary")
    explanation_parts.append(f"The model predicts a {direction} trend for {ticker} stock over the next {len(predicted_prices)} trading days.")
    explanation_parts.append(f"- Current price: ${current_price:.2f}")
    explanation_parts.append(f"- Final predicted price: ${predicted_prices[-1]:.2f}")
    explanation_parts.append(f"- Overall change: ${overall_change:.2f} ({overall_pct_change:.2f}%)")
    
    # Daily breakdown
    explanation_parts.append(f"\n## Daily Breakdown")
    for i, (price, change, pct_change) in enumerate(zip(predicted_prices, changes, pct_changes), 1):
        # Determine confidence based on prediction horizon
        if i <= 2:
            confidence = "Medium"
        elif i <= 4:
            confidence = "Low"
        else:
            confidence = "Low"
            
        # Adjust confidence based on magnitude of change
        if abs(pct_change) > 5:
            confidence = "Low"
        
        explanation_parts.append(f"Day {i}: ${price:.2f} ({'+' if change >= 0 else ''}{change:.2f}, {pct_change:.2f}%) - {confidence}")
    
    # Add disclaimer
    explanation_parts.append(f"\n## Disclaimer")
    explanation_parts.append("These predictions are based on historical data and machine learning models. Stock markets are inherently unpredictable, and these predictions should not be the sole basis for investment decisions. Always consult a qualified financial advisor.")
    
    # Join the parts
    explanation = "\n".join(explanation_parts)
    
    return explanation

def get_latest_stock_price(ticker):
    """Fetch the latest real-time stock price from Yahoo Finance."""
    try:
        import yfinance as yf
        
        logger.info(f"Fetching latest real-time price for {ticker}...")
        
        # Get the stock information
        stock = yf.Ticker(ticker)
        
        # Get real-time price (will try a few methods)
        price = None
        
        # Method 1: Try to get the live price from info
        try:
            info = stock.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price = info['regularMarketPrice']
                logger.info(f"Got real-time price from info['regularMarketPrice']: ${price:.2f}")
                return float(price)
        except Exception as e:
            logger.warning(f"Could not get price from info: {e}")
            
        # Method 2: Try to get the live price from fast_info
        try:
            price = stock.fast_info['last_price']
            logger.info(f"Got real-time price from fast_info: ${price:.2f}")
            return float(price)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not get price from fast_info: {e}")
            
        # Method 3: Get the most recent minute data
        try:
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                logger.info(f"Got price from 1-minute data: ${price:.2f}")
                return price
        except Exception as e:
            logger.warning(f"Could not get 1-minute price data: {e}")
            
        # Method 4: Get the most recent day's data
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                logger.info(f"Got price from daily data: ${price:.2f}")
                return price
        except Exception as e:
            logger.warning(f"Could not get daily price data: {e}")
            
        if price is None:
            logger.error(f"Could not get real-time price for {ticker}")
            return None
            
        return float(price)
    except Exception as e:
        logger.error(f"Error fetching real-time price: {str(e)}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='NVDA',
                      help='Stock ticker symbol (default: NVDA)')
    parser.add_argument('--no-train', action='store_true',
                      help='Skip training and use existing model')
    parser.add_argument('--current-price', type=float,
                      help='Current stock price (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    ticker = args.ticker
    
    # Configure logging
    setup_logging()
    
    try:
        # Get the current price if provided, otherwise fetch it
        current_price = args.current_price
        if current_price is None:
            current_price = get_latest_stock_price(ticker)
            if current_price is None:
                logger.error("Failed to get current price")
                return
        
        logger.info(f"Using current price: ${current_price:.2f}")
        
        # Load or train the model
        if not args.no_train:
            train_model(ticker)
        
        # Generate predictions
        generate_predictions(ticker, current_price)
        
        logger.info("Prediction process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 