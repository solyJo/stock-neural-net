#!/usr/bin/env python
# Discord Notification Runner for NVIDIA Stock Predictor
# Run this script daily to send predictions to Discord

import argparse
import os
import sys
import logging
import time
import datetime
import subprocess
from pathlib import Path

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
import discord_notify
import candlestick_chart
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discord_notify.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Discord Notification Runner for NVIDIA Stock Predictor")
    parser.add_argument("--webhook", type=str, default="", help="Discord webhook URL")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol (default: NVDA)")
    parser.add_argument("--predictions", type=str, default="", help="Path to predictions file (optional)")
    parser.add_argument("--schedule", type=str, default="", help="Schedule time (HH:MM) for daily notifications")
    parser.add_argument("--test", action="store_true", help="Send a test notification")
    parser.add_argument("--no-candlestick", action="store_true", help="Skip sending candlestick chart")
    parser.add_argument("--no-train", action="store_true", help="Skip training, use existing model")
    parser.add_argument("--verify", action="store_true", help="Verify outputs without sending to Discord")
    return parser.parse_args()

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

def run_prediction_pipeline(ticker="NVDA", no_train=False):
    """Run the stock prediction pipeline using the new prediction discord script"""
    logger.info(f"Running prediction pipeline for {ticker}...")
    
    try:
        # Run the prediction with discord script
        cmd = ["python", "run_prediction_discord.py", "--ticker", ticker, "--send-only"]
        if no_train:
            cmd.append("--no-train")
            
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Prediction and Discord notification executed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing prediction pipeline: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running prediction pipeline: {str(e)}")
        return False

def verify_outputs(ticker="NVDA", predictions_file=None):
    """Generate outputs and verify they're created without sending to Discord"""
    if predictions_file is None:
        predictions_file = f"{ticker}_live_predictions.csv"
        
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        return False
        
    logger.info(f"Verifying outputs for {ticker} using {predictions_file}")
    
    # Temporarily set a dummy webhook URL
    os.environ["DISCORD_WEBHOOK_URL"] = "https://example.com/dummy-webhook"
    
    try:
        # Create candlestick chart
        candlestick_file = candlestick_chart.create_candlestick_chart(ticker, predictions_file)
        if candlestick_file and os.path.exists(candlestick_file):
            logger.info(f"Candlestick chart created successfully: {candlestick_file}")
            # Keep the file for inspection
        else:
            logger.warning("Failed to create candlestick chart")
        
        # Print a summary of the predictions
        logger.info("Prediction summary:")
        try:
            df = pd.read_csv(predictions_file, index_col=0)
            for idx, row in df.iterrows():
                if 'Predicted_Price' in row and 'Pct_Change' in row and 'Confidence_Assessment' in row:
                    price = row['Predicted_Price']
                    pct_change = row['Pct_Change'] if not pd.isna(row['Pct_Change']) else 0
                    confidence = row['Confidence_Assessment']
                    logger.info(f"{idx}: ${price:.2f} ({'+' if pct_change >= 0 else ''}{pct_change:.2f}%) - {confidence}")
        except Exception as e:
            logger.error(f"Error reading predictions: {str(e)}")
        
        logger.info("Output verification completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying outputs: {str(e)}")
        return False

def send_predictions_to_discord(predictions_file, ticker="NVDA", include_candlestick=True):
    """Send predictions to Discord including optional candlestick chart"""
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        return False
        
    # Send standard prediction results
    standard_result = discord_notify.send_prediction_results(predictions_file, ticker)
    
    # Send candlestick chart if requested
    if include_candlestick:
        logger.info("Sending candlestick chart to Discord...")
        time.sleep(2)  # Brief delay to avoid Discord rate limits
        candlestick_result = candlestick_chart.send_candlestick_chart_to_discord(ticker, predictions_file)
        
        return standard_result and candlestick_result
    else:
        return standard_result

def main():
    """Main function"""
    args = parse_args()
    
    # Verify mode - check outputs without sending to Discord
    if args.verify:
        if verify_outputs(args.ticker, args.predictions):
            logger.info("Verification completed successfully")
            return 0
        else:
            logger.error("Verification failed")
            return 1
    
    # If webhook URL is provided, set it
    if args.webhook:
        if discord_notify.set_webhook_url(args.webhook):
            logger.info("Discord webhook URL configured successfully")
        else:
            logger.error("Failed to configure Discord webhook URL")
            return 1
    
    # If schedule time is provided, print scheduling instructions
    if args.schedule:
        discord_notify.schedule_daily_notification(args.ticker, args.schedule)
        return 0
        
    # If test flag is set, send a test notification
    if args.test:
        if discord_notify.test_webhook_connection():
            logger.info("Test notification sent successfully")
            return 0
        else:
            logger.error("Failed to send test notification")
            return 1
    
    # Determine predictions file to use
    predictions_file = args.predictions
    
    # If no predictions file provided, run the new pipeline to generate and send everything
    if not predictions_file:
        if run_prediction_pipeline(args.ticker, args.no_train):
            logger.info("Predictions generated and sent to Discord successfully")
            return 0
        else:
            logger.error("Failed to generate and send predictions")
            return 1
    else:
        # Use provided predictions file to send to Discord
        include_candlestick = not args.no_candlestick
        if send_predictions_to_discord(predictions_file, args.ticker, include_candlestick):
            logger.info("Predictions sent to Discord successfully")
            return 0
        else:
            logger.error("Failed to send predictions to Discord")
            return 1

if __name__ == "__main__":
    # Exit with the appropriate status code
    sys.exit(main()) 