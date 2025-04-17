# NVIDIA Stock Price Prediction System

A machine learning system for predicting NVIDIA stock prices based on historical data and news sentiment.

## Features

- Deep learning model using LSTM neural networks
- Technical indicators (SMA, RSI, Bollinger Bands)
- News sentiment analysis integration
- Confidence assessment for predictions
- Daily price forecasting for the next 5 trading days
- HTML report generation with visualizations
- Discord notifications with prediction charts
- Candlestick charts for visual prediction analysis
- Percentage-based predictions accounting for stock splits

## Requirements

- Python 3.7+
- Required libraries: pytorch, pandas, numpy, yfinance, matplotlib, scikit-learn, requests, mplfinance

Install the dependencies:

```bash
pip install torch pandas numpy yfinance matplotlib scikit-learn requests mplfinance
```

## Quick Start

### Option 1: Using the Pipeline Script (Recommended)

The easiest way to run the complete prediction process is using the pipeline script:

```bash
python predict_pipeline.py
```

This will:
1. Train the model using historical data (if needed)
2. Make predictions for the next 5 trading days
3. Generate reports in the "reports" directory

### Option 2: Run Individual Components

If you prefer to run individual components:

```bash
# Train the model and make predictions
python stock_predictor.py

# View the predictions
cat NVDA_live_predictions.csv

# Send predictions to Discord with candlestick charts
python run_prediction_discord.py
```

## Pipeline Script Options

The pipeline script supports several command-line options:

```
python predict_pipeline.py [OPTIONS]

Options:
  --no-train             Skip training and use existing model
  --days N               Number of days to predict (default: 5)
  --ticker SYMBOL        Stock ticker symbol (default: NVDA)
  --output-dir DIR       Directory for output reports (default: "reports")
  --sentiment-weight W   Weight for news sentiment (0-1, default: 0.3)
```

Examples:

```bash
# Skip training, only make predictions using existing model
python predict_pipeline.py --no-train

# Predict 10 days ahead instead of 5
python predict_pipeline.py --days 10

# Change ticker symbol to predict a different stock
python predict_pipeline.py --ticker AAPL

# Customize the output directory
python predict_pipeline.py --output-dir custom_reports

# Adjust the sentiment weight
python predict_pipeline.py --sentiment-weight 0.5
```

## Discord Integration

You can receive daily stock predictions directly in your Discord server:

```bash
# Configure Discord webhook
python discord_notify_runner.py --webhook YOUR_WEBHOOK_URL

# Send a test notification
python discord_notify_runner.py --test

# Schedule daily notifications
python discord_notify_runner.py --schedule 08:00

# Run predictions and send both standard and candlestick charts to Discord
python run_prediction_discord.py --webhook YOUR_WEBHOOK_URL

# Skip training and only send existing predictions to Discord
python run_prediction_discord.py --no-train

# Only send existing prediction files to Discord without running predictions
python run_prediction_discord.py --send-only
```

For detailed setup instructions, see [DISCORD_SETUP.md](DISCORD_SETUP.md).

## Output Files

The system generates several output files:

- `NVDA_live_predictions.csv`: CSV file with daily predictions
- `NVDA_prediction_explanation.md`: Markdown file with detailed prediction analysis
- `NVDA_candlestick_chart.png`: Candlestick chart showing historical and predicted prices
- `reports/NVDA_predictions_YYYYMMDD.csv`: Dated copy of predictions
- `reports/NVDA_prediction_plot_YYYYMMDD.png`: Visualization of predictions
- `reports/NVDA_prediction_report_YYYYMMDD.html`: Detailed HTML report
- `pipeline.log`: Execution log
- `stock_prediction.log`: Model training log
- `discord_notify.log`: Discord notification log
- `prediction_discord.log`: Log for the Discord prediction runner

## Model Information

- Architecture: LSTM neural network
- Features: Open, High, Low, Close, Volume, SMA_20, RSI_14, BB_Upper, BB_Middle, BB_Lower, BB_Width
- Training data: Historical NVDA prices with adjusted prices for stock splits
- Prediction method: Percentage changes rather than absolute prices
- News sentiment: Integrated with 30% weight by default
- Confidence assessment: Based on prediction horizon and model certainty

## Key Improvements

- **Stock Split Handling**: Proper accounting for stock splits using adjusted prices
- **Percentage-based Predictions**: More stable predictions across different price ranges
- **Candlestick Visualization**: Historical and predicted prices in familiar trading chart format
- **Detailed Explanation**: Markdown files with analysis of prediction trends and confidence
- **Enhanced Discord Integration**: Both standard and candlestick charts sent to Discord

## Disclaimer

This software is for educational and research purposes only. Stock price predictions involve significant uncertainty and should not be the sole basis for investment decisions. Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions. 