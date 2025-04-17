import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime
import io
import os
import time
import logging
import requests
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_discord_webhook_url():
    """Get the Discord webhook URL from environment variables"""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("Discord webhook URL not found in environment variables")
    return webhook_url

def get_historical_candlestick_data(ticker, days=14):
    """Get historical OHLCV data for a ticker for candlestick chart"""
    try:
        # Calculate the start date (days before today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+5)  # Add extra days to account for weekends/holidays
        
        # Fetch the historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist_data.empty:
            logger.warning(f"No historical data retrieved for {ticker}")
            return None
            
        # Keep only the most recent 'days' trading days
        if len(hist_data) > days:
            hist_data = hist_data.iloc[-days:]
            
        logger.info(f"Retrieved {len(hist_data)} days of historical data for {ticker}")
        return hist_data
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None

def create_future_candlestick_data(predictions, current_price):
    """Create future OHLCV data for predictions"""
    future_data = pd.DataFrame()
    future_data['Open'] = [current_price] + list(predictions['Predicted_Price'].iloc[1:-1])
    future_data['Close'] = predictions['Predicted_Price'].iloc[1:].values
    
    # Calculate High and Low based on predicted price movements
    percent_volatility = 0.01  # 1% intraday volatility assumption
    future_data['High'] = future_data.apply(
        lambda x: max(x['Open'], x['Close']) * (1 + percent_volatility), axis=1
    )
    future_data['Low'] = future_data.apply(
        lambda x: min(x['Open'], x['Close']) * (1 - percent_volatility), axis=1
    )
    
    # Set Volume based on price movement (higher for larger changes)
    avg_volume = 100000000  # Example average volume
    future_data['Volume'] = future_data.apply(
        lambda x: avg_volume * (1 + abs(x['Close'] - x['Open']) / x['Open']), axis=1
    )
    
    # Set index to dates from predictions
    future_data.index = pd.DatetimeIndex(predictions.index[1:])
    
    return future_data

def create_candlestick_chart(ticker="NVDA", predictions_file="NVDA_live_predictions.csv"):
    """Create a candlestick chart with historical data and predictions"""
    try:
        # Load predictions
        predictions = pd.read_csv(predictions_file, index_col=0)
        if len(predictions) < 2:
            logger.error("Not enough predictions in file")
            return None
        
        # Get current price
        current_price = float(predictions['Predicted_Price'].iloc[0])
        
        # Get historical data
        historical_data = get_historical_candlestick_data(ticker, days=14)
        if historical_data is None:
            logger.error("Failed to get historical data")
            return None
        
        # Create future data
        future_data = create_future_candlestick_data(predictions, current_price)
        
        # Save the original plots for historical data
        hist_file = f'temp_hist_chart_{int(time.time())}.png'
        future_file = f'temp_future_chart_{int(time.time())}.png'
        combined_file = f'{ticker}_candlestick_chart.png'
        
        # Create historical candlestick chart
        mc = mpf.make_marketcolors(
            up='green', down='red',
            wick={'up':'green', 'down':'red'},
            edge={'up':'green', 'down':'red'},
            volume={'up':'green', 'down':'red'}
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor='white',
            gridstyle='-'
        )
        
        # Plot historical data
        mpf.plot(
            historical_data,
            type='candle',
            style=s,
            title=f'{ticker} Historical Prices',
            volume=True,
            figsize=(10, 6),
            savefig=hist_file
        )
        
        # Plot future predicted data (with different style to distinguish)
        future_mc = mpf.make_marketcolors(
            up='#1f77b4', down='#ff7f0e',  # Different colors for predictions
            wick={'up':'#1f77b4', 'down':'#ff7f0e'},
            edge={'up':'#1f77b4', 'down':'#ff7f0e'},
            volume={'up':'#1f77b4', 'down':'#ff7f0e'}
        )
        
        future_s = mpf.make_mpf_style(
            marketcolors=future_mc,
            facecolor='white',
            gridstyle='-'
        )
        
        # Plot future data
        mpf.plot(
            future_data,
            type='candle',
            style=future_s,
            title=f'{ticker} Predicted Prices',
            volume=True,
            figsize=(10, 6),
            savefig=future_file
        )
        
        # Create a combined chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # Load saved images
        hist_img = plt.imread(hist_file)
        future_img = plt.imread(future_file)
        
        # Display the images
        ax1.imshow(hist_img)
        ax1.axis('off')
        ax1.set_title(f'{ticker} Historical Prices', fontsize=16)
        
        ax2.imshow(future_img)
        ax2.axis('off')
        ax2.set_title(f'{ticker} Predicted Prices (Starting: ${current_price:.2f})', fontsize=16)
        
        plt.tight_layout()
        fig.suptitle(f'{ticker} Stock Price Analysis - {datetime.now().strftime("%Y-%m-%d")}', 
                     fontsize=18, y=0.98)
        
        # Save the combined chart
        plt.savefig(combined_file, dpi=100)
        plt.close(fig)
        
        # Clean up temp files
        for temp_file in [hist_file, future_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return combined_file
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {str(e)}")
        return None

def send_candlestick_chart_to_discord(ticker="NVDA", predictions_file="NVDA_live_predictions.csv"):
    """Create and send a candlestick chart to Discord"""
    webhook_url = get_discord_webhook_url()
    if not webhook_url:
        logger.error("Discord webhook URL not configured")
        return False
    
    chart_file = create_candlestick_chart(ticker, predictions_file)
    if not chart_file:
        logger.error("Failed to create candlestick chart")
        return False
    
    try:
        # Send message
        message_content = f"## {ticker} Candlestick Chart Analysis - {datetime.now().strftime('%Y-%m-%d')} 📊\n\n*Red/Green candles for historical prices, Blue/Orange candles for predictions*\n\n*This chart shows both historical price action and predictions based on our percentage change model.*"
        
        response = requests.post(
            webhook_url,
            json={"content": message_content}
        )
        
        # Check if the message was sent successfully
        if response.status_code != 204:
            logger.error(f"Discord message failed with status code {response.status_code}: {response.text}")
            return False
        
        # Send the chart
        time.sleep(1)  # Brief delay to avoid rate limits
        with open(chart_file, 'rb') as f:
            files = {'file': ('candlestick_chart.png', f.read())}
            
        response = requests.post(
            webhook_url,
            files=files
        )
        
        # Check if the chart was sent successfully
        # Discord returns 204 for text messages and 200 for file uploads
        if response.status_code != 204 and response.status_code != 200:
            logger.error(f"Discord chart upload failed with status code {response.status_code}: {response.text}")
            return False
            
        logger.info("Candlestick chart sent to Discord successfully")
        return True
    except Exception as e:
        logger.error(f"Error sending candlestick chart to Discord: {str(e)}")
        return False
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(chart_file):
                os.remove(chart_file)
                logger.info(f"Removed temporary chart file: {chart_file}")
        except Exception as e:
            logger.error(f"Error removing chart file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    send_candlestick_chart_to_discord() 