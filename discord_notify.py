import requests
import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import time
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Store webhook URL in environment or config
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

def set_webhook_url(url):
    """Set the Discord webhook URL"""
    global DISCORD_WEBHOOK_URL
    DISCORD_WEBHOOK_URL = url
    logger.info("Discord webhook URL has been set")
    
    # Save to environment variable for persistence
    os.environ["DISCORD_WEBHOOK_URL"] = url
    
    # Test the webhook connection
    return test_webhook_connection()

def test_webhook_connection():
    """Test the Discord webhook connection"""
    if not DISCORD_WEBHOOK_URL:
        logger.error("No Discord webhook URL configured")
        return False
        
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={
                "content": "🔔 NVIDIA Stock Predictor is now connected! You will receive daily predictions here."
            }
        )
        
        if response.status_code == 204:
            logger.info("Discord webhook test successful")
            return True
        else:
            logger.error(f"Discord webhook test failed with status code {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing Discord webhook: {str(e)}")
        return False

def send_prediction_results(predictions_file, ticker="NVDA"):
    """Send prediction results to Discord"""
    if not DISCORD_WEBHOOK_URL:
        logger.error("No Discord webhook URL configured")
        return False
        
    try:
        # Read the predictions
        predictions = pd.read_csv(predictions_file, index_col=0)
        
        if len(predictions) == 0:
            logger.error("No predictions found in file")
            return False
            
        # Format the prediction message
        current_price = predictions['Predicted_Price'].iloc[0]
        next_day_price = predictions['Predicted_Price'].iloc[1]
        next_day_change = predictions['Change'].iloc[1]
        next_day_pct_change = predictions['Pct_Change'].iloc[1]
        next_day_confidence = predictions['Confidence_Assessment'].iloc[1]
        
        # Determine emoji based on price movement
        if next_day_change > 0:
            emoji = "🟢"  # Green for increase
        elif next_day_change < 0:
            emoji = "🔴"  # Red for decrease
        else:
            emoji = "⚪"  # White for no change
            
        message_content = f"""
## {ticker} Stock Prediction - {datetime.now().strftime('%Y-%m-%d')} 📊

**Current Price:** ${current_price:.2f}

**Tomorrow's Prediction:**
{emoji} **${next_day_price:.2f}** ({'+' if next_day_change >= 0 else ''}{next_day_change:.2f}, {'+' if next_day_pct_change >= 0 else ''}{next_day_pct_change:.2f}%)
**Confidence:** {next_day_confidence}

**5-Day Forecast:**
"""

        # Add all predictions (except the current price)
        for i in range(1, len(predictions)):
            date = predictions.index[i].split(' ')[0]  # Get just the date part
            price = predictions['Predicted_Price'].iloc[i]
            change = predictions['Pct_Change'].iloc[i]
            
            # Emoji based on change
            if change > 0:
                day_emoji = "📈"
            elif change < 0:
                day_emoji = "📉"
            else:
                day_emoji = "➖"
                
            message_content += f"{day_emoji} {date}: ${price:.2f}"
            
            if i < len(predictions) - 1:
                message_content += "\n"
                
        # Add disclaimer
        message_content += "\n\n*This prediction is based on machine learning analysis and should not be the sole basis for investment decisions.*"
        
        # Create a chart of the predictions
        chart_file = create_prediction_chart(predictions, ticker)
        
        # First send the text message
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message_content}
        )
        
        # Check if the message was sent successfully
        if response.status_code != 204:
            logger.error(f"Discord message failed with status code {response.status_code}: {response.text}")
            return False
            
        # Then send the chart as a separate message if we have one
        if chart_file:
            try:
                time.sleep(1)  # Brief delay to avoid rate limits
                with open(chart_file, 'rb') as f:
                    files = {'file': ('prediction_chart.png', f.read())}
                    
                response = requests.post(
                    DISCORD_WEBHOOK_URL,
                    files=files
                )
                
                # Check if the chart was sent successfully
                # Discord returns 204 for text messages and 200 for file uploads
                if response.status_code != 204 and response.status_code != 200:
                    logger.error(f"Discord chart upload failed with status code {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Error sending chart to Discord: {str(e)}")
            finally:
                # Clean up the temporary file
                try:
                    if os.path.exists(chart_file):
                        os.remove(chart_file)
                        logger.info(f"Removed temporary chart file: {chart_file}")
                except Exception as e:
                    logger.error(f"Error removing chart file: {str(e)}")
        
        logger.info("Discord notification sent successfully")
        return True
    except Exception as e:
        logger.error(f"Error sending prediction to Discord: {str(e)}")
        return False

def get_historical_data(ticker, days=7):
    """Get historical stock data for the specified number of days"""
    try:
        # Calculate the start date (7 days before today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+5)  # Add extra days to account for weekends/holidays
        
        # Fetch the historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist_data.empty:
            logger.warning(f"No historical data retrieved for {ticker}")
            return None
            
        # Keep only the most recent 7 trading days
        if len(hist_data) > days:
            hist_data = hist_data.iloc[-days:]
            
        logger.info(f"Retrieved {len(hist_data)} days of historical data for {ticker}")
        return hist_data
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None

def create_prediction_chart(predictions, ticker):
    """Create a chart of the predictions and return the file path"""
    try:
        # Extract just the prices for plotting
        pred_dates = [d.split(' ')[0] for d in predictions.index]  # Get just the date part
        pred_prices = predictions['Predicted_Price'].values
        
        # Get historical data for the past 7 days
        historical_data = get_historical_data(ticker, days=7)
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Plot the historical data if available
        if historical_data is not None and not historical_data.empty:
            # Convert the index to strings for consistency with prediction dates
            hist_dates = [d.strftime('%Y-%m-%d') for d in historical_data.index]
            hist_prices = historical_data['Close'].values
            
            # Plot historical data
            plt.plot(hist_dates, hist_prices, marker='o', linestyle='-', color='#4B0082', 
                     linewidth=2, markersize=8, label='Historical Prices')
            
            # Add a vertical line to separate historical from predictions
            plt.axvline(x=len(hist_dates)-0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Add text label
            plt.text(len(hist_dates)-3, min(hist_prices)*0.98, 'Historical', 
                     fontsize=10, color='#4B0082', fontweight='bold')
            plt.text(len(hist_dates)+1, min(pred_prices)*0.98, 'Predictions', 
                     fontsize=10, color='#1f77b4', fontweight='bold')
        
        # Plot the predictions
        prediction_start = 0 if historical_data is None else len(historical_data)
        all_dates = ([] if historical_data is None else hist_dates) + pred_dates
        plt.plot(all_dates[prediction_start:], pred_prices, marker='o', linestyle='-', 
                 color='#1f77b4', linewidth=2, markersize=8, label='Predicted Prices')
        
        # Add markers for up/down days in predictions
        for i in range(1, len(predictions)):
            idx = prediction_start + i
            if predictions['Change'].iloc[i] > 0:
                plt.plot(all_dates[idx], pred_prices[i], marker='^', color='green', markersize=10)
            elif predictions['Change'].iloc[i] < 0:
                plt.plot(all_dates[idx], pred_prices[i], marker='v', color='red', markersize=10)
        
        # Highlight current price
        current_idx = prediction_start
        plt.plot(all_dates[current_idx], pred_prices[0], marker='o', color='#ff7f0e', 
                 markersize=10, label='Current Price')
        
        # Add labels and title
        plt.title(f'{ticker} Stock Price Prediction - {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save to a temporary file
        temp_file = f'temp_prediction_chart_{int(time.time())}.png'
        plt.savefig(temp_file, dpi=100)
        plt.close()
        
        return temp_file
    except Exception as e:
        logger.error(f"Error creating prediction chart: {str(e)}")
        return None

def schedule_daily_notification(ticker="NVDA", time_str="08:00"):
    """
    Print instructions for setting up a scheduled task 
    to run the prediction and notification daily
    """
    script_path = os.path.abspath("discord_notify_runner.py")
    
    print(f"\n=== DAILY NOTIFICATION SETUP ===")
    print(f"To receive daily notifications at {time_str}, you'll need to set up a scheduled task.")
    
    # Windows instructions
    print("\nFor Windows (using Task Scheduler):")
    print(f"1. Open Task Scheduler")
    print(f"2. Create a new Basic Task")
    print(f"3. Name it 'NVIDIA Stock Prediction'")
    print(f"4. Set it to run Daily at {time_str}")
    print(f"5. Set the action to 'Start a Program'")
    print(f"6. Program/script: python")
    print(f"7. Add arguments: {script_path}")
    
    # Linux/Mac instructions
    print("\nFor Linux/Mac (using crontab):")
    hour, minute = time_str.split(":")
    print(f"Run this command: crontab -e")
    print(f"Add this line: {minute} {hour} * * * python {script_path}")
    
    print("\nAlternatively, you can use the provided discord_notify_runner.py script:")
    print(f"python discord_notify_runner.py --schedule {time_str}")
    
    return True 