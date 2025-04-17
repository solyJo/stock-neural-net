import os
import sys
import argparse
import logging
import subprocess
import time
from datetime import datetime
import discord_notify
import candlestick_chart
import yfinance as yf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("prediction_discord.log"),
    ]
)
logger = logging.getLogger(__name__)

def run_prediction_script(ticker="NVDA", no_train=False):
    """Run the stock prediction script"""
    try:
        # First get the live price
        stock = yf.Ticker(ticker)
        try:
            live_price = stock.info['regularMarketPrice']
            logger.info(f"Got live price for {ticker}: ${live_price:.2f}")
        except:
            logger.warning("Could not get live price from info, will try fast_info")
            try:
                live_price = stock.fast_info['last_price']
                logger.info(f"Got live price from fast_info: ${live_price:.2f}")
            except:
                logger.warning("Could not get live price, will use latest close")
                live_price = None

        cmd = ["python", "stock_predictor.py"]
        if ticker != "NVDA":
            cmd.extend(["--ticker", ticker])
        if no_train:
            cmd.append("--no-train")
        if live_price is not None:
            cmd.extend(["--current-price", str(live_price)])
            
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        logger.info("Stock prediction completed successfully")
        if result.stdout:
            logger.debug(f"Stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
            
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running stock prediction: {e}")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running stock prediction: {str(e)}")
        return False

def send_to_discord(ticker="NVDA", predictions_file=None):
    """Send predictions to Discord using both standard notification and candlestick chart"""
    if predictions_file is None:
        predictions_file = f"{ticker}_live_predictions.csv"
        
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        return False
        
    # Send standard notification
    logger.info("Sending standard Discord notification...")
    standard_result = discord_notify.send_prediction_results(predictions_file, ticker)
    
    # Send candlestick chart
    logger.info("Sending candlestick chart to Discord...")
    time.sleep(2)  # Brief delay to avoid Discord rate limits
    candlestick_result = candlestick_chart.send_candlestick_chart_to_discord(ticker, predictions_file)
    
    return standard_result and candlestick_result

def test_mode(ticker="NVDA", predictions_file=None):
    """Test mode that generates outputs without sending to Discord"""
    if predictions_file is None:
        predictions_file = f"{ticker}_live_predictions.csv"
        
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        return False
    
    # Set a dummy webhook URL for testing
    os.environ["DISCORD_WEBHOOK_URL"] = "https://example.com/dummy-webhook"
    
    # Create candlestick chart
    logger.info("Creating candlestick chart (test mode)...")
    try:
        chart_file = candlestick_chart.create_candlestick_chart(ticker, predictions_file)
        if chart_file:
            logger.info(f"Candlestick chart created at: {chart_file}")
            
        # Get content that would be sent to Discord
        with open(predictions_file, 'r') as f:
            logger.info(f"Predictions loaded from: {predictions_file}")
            logger.info(f"First few lines: {f.readline()}\n{f.readline()}\n{f.readline()}")
            
        logger.info("Test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in test mode: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Stock Prediction and Send to Discord")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol")
    parser.add_argument("--webhook", type=str, default="", help="Discord webhook URL")
    parser.add_argument("--no-train", action="store_true", help="Skip training (use saved model)")
    parser.add_argument("--send-only", action="store_true", help="Skip prediction and only send to Discord")
    parser.add_argument("--predictions", type=str, default="", help="Path to existing predictions file")
    parser.add_argument("--test", action="store_true", help="Test mode - create outputs without sending to Discord")
    
    args = parser.parse_args()
    
    # Configure Discord webhook if provided
    if args.webhook:
        if discord_notify.set_webhook_url(args.webhook):
            logger.info("Discord webhook URL configured successfully")
        else:
            logger.error("Failed to configure Discord webhook URL")
            return False
    
    # Set predictions file
    predictions_file = args.predictions if args.predictions else f"{args.ticker}_live_predictions.csv"
    
    # Run stock prediction if not in send-only mode
    if not args.send_only and not args.test:
        logger.info(f"Running prediction for {args.ticker}...")
        if not run_prediction_script(args.ticker, args.no_train):
            logger.error("Failed to run prediction script")
            return False
            
        logger.info(f"Prediction completed. Waiting 2 seconds before sending notifications...")
        time.sleep(2)  # Wait for files to be written
    
    # Test mode or send to Discord
    if args.test:
        logger.info(f"Running in TEST MODE - outputs will be generated but not sent")
        if test_mode(args.ticker, predictions_file):
            logger.info("Test completed successfully")
            return True
        else:
            logger.error("Test failed")
            return False
    else:
        # Send to Discord
        logger.info(f"Sending {args.ticker} predictions to Discord...")
        if send_to_discord(args.ticker, predictions_file):
            logger.info("Discord notifications sent successfully")
            return True
        else:
            logger.error("Failed to send Discord notifications")
            return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 