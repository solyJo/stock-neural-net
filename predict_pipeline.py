#!/usr/bin/env python
# NVIDIA Stock Prediction Pipeline
# This script runs the entire prediction process from start to finish

import os
import sys
import logging
import argparse
import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NVIDIA Stock Prediction Pipeline")
    parser.add_argument("--no-train", action="store_true", help="Skip training and use existing model")
    parser.add_argument("--days", type=int, default=5, help="Number of days to predict (default: 5)")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol (default: NVDA)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for output reports")
    parser.add_argument("--sentiment-weight", type=float, default=0.3, 
                       help="Weight for news sentiment (0-1, default: 0.3)")
    return parser.parse_args()

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

def get_latest_stock_price(ticker):
    """Fetch the latest real-time stock price from Yahoo Finance."""
    try:
        import yfinance as yf
        import time
        
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

def run_model(args):
    """Run the stock prediction model with given arguments."""
    import yfinance as yf
    import pickle
    import torch
    
    # Configure logging
    logger.info(f"Running prediction model for {args.ticker} for {args.days} days")
    
    # Check for available model
    model_file = f"{args.ticker}_model.pth"
    if not os.path.exists(model_file) and args.no_train:
        logger.error(f"Model file {model_file} not found and --no-train specified")
        return False

    # Check for available scalers
    scalers_file = f"{args.ticker}_scalers.pkl"
    if not os.path.exists(scalers_file):
        logger.error(f"Scalers file {scalers_file} not found")
        return False
        
    # If we're not skipping training, run the full model
    if not args.no_train:
        logger.info("Running full model training...")
        
        # Run the stock_predictor.py script directly as a subprocess
        try:
            result = subprocess.run(
                ["python", "stock_predictor.py"], 
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log a preview of the output
            output_preview = result.stdout.split('\n')[:10]
            logger.info(f"Stock predictor script executed successfully. First few lines of output:")
            for line in output_preview:
                if line.strip():
                    logger.info(line.strip())
            
            # Check if the model was created
            if os.path.exists(model_file):
                logger.info(f"Model file {model_file} was successfully created")
            else:
                logger.warning(f"Model file {model_file} was not created during training")
                
            # If model exists now, we can continue with predictions
            if os.path.exists(model_file) and os.path.exists(scalers_file):
                logger.info("Training completed successfully, proceeding with predictions")
                
                # The result file already contains predictions, so we can skip to updating with live price
                if os.path.exists(f"{args.ticker}_live_predictions.csv"):
                    logger.info(f"Predictions file {args.ticker}_live_predictions.csv already generated")
                    
                    # Update the predictions with the latest stock price
                    live_price = get_latest_stock_price(args.ticker)
                    if live_price is not None:
                        try:
                            # Update the current price in the predictions file
                            update_current_price_in_predictions(f"{args.ticker}_live_predictions.csv", live_price)
                            logger.info(f"Updated current price in predictions to real-time value: ${live_price:.2f}")
                        except Exception as e:
                            logger.warning(f"Could not update current price: {str(e)}")
                    
                    return True
            else:
                logger.error("Training did not produce necessary model files")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing stock_predictor.py: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running stock_predictor.py: {str(e)}")
            return False
    
    # If we're only doing prediction or if we need to run predictions separately
    # Call the stock_predictor.py script to generate predictions
    logger.info("Running stock_predictor.py to generate predictions...")
    
    try:
        # Run stock_predictor.py to generate predictions
        result = subprocess.run(
            ["python", "stock_predictor.py"], 
            check=True,
            capture_output=True,
            text=True
        )
        
        # Check if the predictions file was created
        if os.path.exists(f"{args.ticker}_live_predictions.csv"):
            logger.info(f"Predictions file {args.ticker}_live_predictions.csv generated successfully")
            
            # Get the latest real-time price
            live_price = get_latest_stock_price(args.ticker)
            if live_price is not None:
                try:
                    # Update the current price in the predictions file
                    update_current_price_in_predictions(f"{args.ticker}_live_predictions.csv", live_price)
                    logger.info(f"Updated current price in predictions to real-time value: ${live_price:.2f}")
                except Exception as e:
                    logger.warning(f"Could not update current price: {str(e)}")
            
            # Copy to reports directory
            ensure_directory(args.output_dir)
            import shutil
            shutil.copy2(
                f"{args.ticker}_live_predictions.csv", 
                f"{args.output_dir}/{args.ticker}_predictions_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            )
            
            return True
        else:
            logger.error(f"Predictions file {args.ticker}_live_predictions.csv was not created")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing stock_predictor.py: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running stock_predictor.py: {str(e)}")
        return False

def update_current_price_in_predictions(predictions_file, live_price):
    """Update the current price in an existing predictions CSV file."""
    try:
        # Read the predictions
        predictions = pd.read_csv(predictions_file, index_col=0)
        
        if len(predictions) > 0:
            # Get the current price from the first row
            current_price = predictions['Predicted_Price'].iloc[0]
            
            # Update the current price with the live price
            predictions.loc[predictions.index[0], 'Predicted_Price'] = live_price
            
            # Recalculate changes for future predictions
            for i in range(1, len(predictions)):
                predicted_price = predictions['Predicted_Price'].iloc[i]
                prev_price = predictions['Predicted_Price'].iloc[i-1]
                
                change = predicted_price - prev_price
                pct_change = (change / prev_price) * 100 if prev_price > 0 else 0
                
                predictions.loc[predictions.index[i], 'Change'] = change
                predictions.loc[predictions.index[i], 'Pct_Change'] = pct_change
            
            # Save the updated predictions
            predictions.to_csv(predictions_file)
            
            return True
    except Exception as e:
        logger.error(f"Error updating current price in predictions: {str(e)}")
        return False

def generate_report(args):
    """Generate a summary report of the predictions."""
    try:
        # Read the predictions
        predictions_file = f"{args.ticker}_live_predictions.csv"
        if not os.path.exists(predictions_file):
            logger.error(f"Predictions file {predictions_file} not found")
            return False
            
        predictions = pd.read_csv(predictions_file, index_col=0)
        
        # Create report directory
        ensure_directory(args.output_dir)
        
        # Generate visualization
        today_str = datetime.datetime.now().strftime('%Y%m%d')
        plt.figure(figsize=(14, 8))
        
        # Convert index to datetime if it's not already
        if not isinstance(predictions.index, pd.DatetimeIndex):
            predictions.index = pd.to_datetime(predictions.index)
            
        # Get recent historical data for context
        try:
            import yfinance as yf
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)  # Get 30 days of historical data
            historical_data = yf.download(args.ticker, start=start_date, end=end_date)
            
            if not historical_data.empty:
                # Add historical data to chart
                plt.plot(historical_data.index, historical_data['Close'], 
                         color='gray', alpha=0.7, linestyle='-', 
                         linewidth=1.5, label='Historical Price')
                
                # Add a vertical line separating historical from predictions
                if len(predictions) > 0:
                    first_prediction_date = predictions.index[0]
                    plt.axvline(x=first_prediction_date, color='black', linestyle='--', alpha=0.5)
        except Exception as e:
            logger.warning(f"Could not add historical data to chart: {str(e)}")
            
        # Plot predictions with better styling
        plt.plot(predictions.index, predictions['Predicted_Price'], 
                 marker='o', linewidth=2.5, markersize=8, 
                 color='#1f77b4', label='Predicted Price')
        
        # Highlight confidence levels with shading if available
        confidence_colors = {
            'High': 'green',
            'Medium': 'orange',
            'Low': 'red'
        }
        
        # Add confidence level indicators
        for idx, row in predictions.iterrows():
            if 'Confidence_Assessment' in row:
                conf_str = str(row['Confidence_Assessment'])
                for level, color in confidence_colors.items():
                    if level in conf_str:
                        plt.plot(idx, row['Predicted_Price'], 'o', markersize=12, 
                                 markerfacecolor='none', markeredgecolor=color, 
                                 markeredgewidth=2)
                        break
        
        # Highlight current price
        current_price = predictions['Predicted_Price'].iloc[0]
        plt.axhline(y=current_price, color='#ff7f0e', linestyle='--', alpha=0.8, 
                    linewidth=1.5, label=f'Current Price (${current_price:.2f})')
        
        # Add labels and styling
        plt.title(f"{args.ticker} Stock Price Prediction", fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Format axes
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.2f}'))
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        # Add annotations for price changes
        for i in range(1, len(predictions)):
            price = predictions['Predicted_Price'].iloc[i]
            date = predictions.index[i]
            change = predictions['Change'].iloc[i] if 'Change' in predictions else 0
            pct_change = predictions['Pct_Change'].iloc[i] if 'Pct_Change' in predictions else 0
            
            # Only add annotation for first change (to avoid clutter)
            if i == 1 and isinstance(change, (int, float)) and change != 0:
                text = f"{'+' if change > 0 else ''}{change:.2f} ({pct_change:.1f}%)"
                plt.annotate(text, 
                            xy=(date, price),
                            xytext=(10, 15 if change > 0 else -15),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='black'),
                            color='green' if change > 0 else 'red',
                            fontsize=12,
                            fontweight='bold')
        
        # Adjust y-axis to show all data clearly with some padding
        ymin = min(min(predictions['Predicted_Price']) * 0.95, current_price * 0.95)
        ymax = max(max(predictions['Predicted_Price']) * 1.05, current_price * 1.05)
        plt.ylim(ymin, ymax)
        
        # Rotary the x-axis date labels for better readability
        plt.gcf().autofmt_xdate()
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{args.output_dir}/{args.ticker}_prediction_plot_{today_str}.png"
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        logger.info(f"Generated visualization: {plot_file}")
        
        # Close the plot to free memory
        plt.close()
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>{args.ticker} Stock Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #333366; color: white; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: gray; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .disclaimer {{ font-style: italic; margin-top: 30px; padding: 10px; background-color: #f8f8f8; border-left: 4px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>{args.ticker} Stock Prediction Report</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="chart-container">
                <img src="{os.path.basename(plot_file)}" alt="{args.ticker} Price Prediction Chart" style="max-width:100%; height:auto;">
            </div>
            
            <h2>Price Predictions</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Predicted Price</th>
                    <th>Change</th>
                    <th>% Change</th>
                    <th>Confidence</th>
                </tr>
        """
        
        # Add rows for each prediction
        for idx, row in predictions.iterrows():
            price = row['Predicted_Price']
            change = row.get('Change', 0)
            pct_change = row.get('Pct_Change', 0)
            confidence = row.get('Confidence_Assessment', 'N/A')
            
            # Determine CSS class based on change
            if isinstance(change, (int, float)) and change > 0:
                change_class = "positive"
            elif isinstance(change, (int, float)) and change < 0:
                change_class = "negative"
            else:
                change_class = "neutral"
                
            # Format change values
            if isinstance(change, (int, float)):
                change_str = f"${change:.2f}"
            else:
                change_str = str(change)
                
            if isinstance(pct_change, (int, float)):
                pct_change_str = f"{pct_change:.2f}%"
            else:
                pct_change_str = str(pct_change)
            
            html_report += f"""
                <tr>
                    <td>{idx}</td>
                    <td>${price:.2f}</td>
                    <td class="{change_class}">{change_str}</td>
                    <td class="{change_class}">{pct_change_str}</td>
                    <td>{confidence}</td>
                </tr>
            """
        
        # Add summary and close HTML
        first_price = predictions['Predicted_Price'].iloc[0]
        last_price = predictions['Predicted_Price'].iloc[-1]
        overall_change = last_price - first_price
        overall_pct_change = (overall_change / first_price) * 100 if first_price > 0 else 0
        
        overall_class = "positive" if overall_change > 0 else "negative" if overall_change < 0 else "neutral"
        
        html_report += f"""
            </table>
            
            <h2>Summary</h2>
            <p>Starting price: <strong>${first_price:.2f}</strong></p>
            <p>Ending price: <strong>${last_price:.2f}</strong></p>
            <p>Overall change: <span class="{overall_class}"><strong>${overall_change:.2f} ({overall_pct_change:.2f}%)</strong></span></p>
            
            <p class="disclaimer">
                <strong>Disclaimer:</strong> These predictions are based on historical data and machine learning models. 
                Stock markets are inherently unpredictable, and these predictions should not be the sole basis for investment decisions.
                Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.
            </p>
        </body>
        </html>
        """
        
        # Save HTML report
        report_file = f"{args.output_dir}/{args.ticker}_prediction_report_{today_str}.html"
        with open(report_file, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Generated report: {report_file}")
        
        # Generate console-based summary explanation
        generate_console_summary(predictions, args.ticker)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return False

def generate_console_summary(predictions, ticker):
    """
    Generate a summary of the predictions for console output using a rule-based approach
    """
    try:
        # Create summary box
        print("\n" + "="*80)
        print(f"  STOCK PREDICTION SUMMARY: {ticker}".center(80))
        print("="*80)
        
        # Current price
        current_price = predictions['Predicted_Price'].iloc[0]
        print(f"\n  CURRENT PRICE: ${current_price:.2f}")
        
        # Print prediction table
        print("\n  PRICE PREDICTIONS:")
        print("  " + "-"*60)
        print("  {:<12} {:<12} {:<12} {:<12} {:<20}".format("Date", "Price", "Change", "% Change", "Confidence"))
        print("  " + "-"*60)
        
        for idx, row in predictions.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            price = row['Predicted_Price']
            change = row.get('Change', 0)
            pct_change = row.get('Pct_Change', 0)
            confidence = row.get('Confidence_Assessment', 'N/A')
            
            # Format values
            price_str = f"${price:.2f}"
            
            if isinstance(change, (int, float)):
                if change > 0:
                    change_str = f"+${change:.2f}"
                elif change < 0:
                    change_str = f"-${abs(change):.2f}"
                else:
                    change_str = f"${change:.2f}"
            else:
                change_str = str(change)
                
            if isinstance(pct_change, (int, float)):
                if pct_change > 0:
                    pct_change_str = f"+{pct_change:.2f}%"
                elif pct_change < 0:
                    pct_change_str = f"{pct_change:.2f}%"
                else:
                    pct_change_str = f"{pct_change:.2f}%"
            else:
                pct_change_str = str(pct_change)
                
            # Truncate confidence to fit in table
            if len(confidence) > 20:
                confidence = confidence[:17] + "..."
                
            print("  {:<12} {:<12} {:<12} {:<12} {:<20}".format(
                date_str, price_str, change_str, pct_change_str, confidence
            ))
            
        print("  " + "-"*60)
        
        # Overall change
        first_price = predictions['Predicted_Price'].iloc[0]
        last_price = predictions['Predicted_Price'].iloc[-1]
        overall_change = last_price - first_price
        overall_pct_change = (overall_change / first_price) * 100 if first_price > 0 else 0
        
        direction = "INCREASE" if overall_change > 0 else "DECREASE" if overall_change < 0 else "NO CHANGE"
        print(f"\n  OVERALL {direction}: ${abs(overall_change):.2f} ({abs(overall_pct_change):.2f}%)")
        
        # Try to use Llama2 for analysis if available
        try:
            print("\n  MODEL ANALYSIS:")
            print("  " + "-"*60)
            
            from stock_predictor import explain_predictions_with_ollama, generate_fallback_explanation
            
            metrics = None  # We don't have metrics in the pipeline
            
            # Get explanation
            llama_explanation = None
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("stock_predictor", "stock_predictor.py")
                stock_predictor = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(stock_predictor)
                
                # Try getting explanation from Ollama
                llama_explanation = stock_predictor.explain_predictions_with_ollama(predictions, None, None)
            except Exception as e:
                logger.warning(f"Could not use stock_predictor for explanation: {str(e)}")
                
            if not llama_explanation:
                # Generate a simpler explanation
                llama_explanation = generate_simple_explanation(predictions, ticker)
                
            # Format and print the explanation
            if llama_explanation:
                # Split into lines and add indentation
                explanation_lines = llama_explanation.split('\n')
                for line in explanation_lines:
                    if line.strip():
                        # Wrap long lines
                        import textwrap
                        wrapped_lines = textwrap.wrap(line, width=76)
                        for wrapped in wrapped_lines:
                            print(f"  {wrapped}")
        except Exception as e:
            logger.warning(f"Could not generate Llama2 analysis: {str(e)}")
            print("  Analysis not available. Check the HTML report for more details.")
            
        # Add disclaimer
        print("\n  " + "-"*60)
        print("  DISCLAIMER: These predictions are for educational purposes only.")
        print("  Always consult a financial advisor before making investment decisions.")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating console summary: {str(e)}")
        print("\n  Error generating prediction summary. Check the log for details.")

def generate_simple_explanation(predictions, ticker):
    """Generate a simple explanation of the predictions without using external models"""
    try:
        # Calculate key statistics
        current_price = predictions['Predicted_Price'].iloc[0]
        future_prices = predictions['Predicted_Price'].iloc[1:]
        
        # Get initial change
        initial_change = predictions['Change'].iloc[1] if len(predictions) > 1 else 0
        initial_pct_change = predictions['Pct_Change'].iloc[1] if len(predictions) > 1 else 0
        
        # Get overall change
        last_price = predictions['Predicted_Price'].iloc[-1]
        overall_change = last_price - current_price
        overall_pct_change = (overall_change / current_price) * 100 if current_price > 0 else 0
        
        # Calculate volatility
        if len(future_prices) > 1:
            changes = predictions['Change'].iloc[1:].values
            volatility = changes.std() if len(changes) > 1 else 0
        else:
            volatility = 0
            
        # Generate explanation
        explanation = [
            f"Analysis of {ticker} Stock Predictions",
            "",
            f"The model predicts that {ticker} stock will "
        ]
        
        # Describe price direction
        if overall_change > 0:
            explanation.append(f"increase by ${overall_change:.2f} (or {overall_pct_change:.2f}%) over the next {len(future_prices)} trading days.")
        elif overall_change < 0:
            explanation.append(f"decrease by ${abs(overall_change):.2f} (or {abs(overall_pct_change):.2f}%) over the next {len(future_prices)} trading days.")
        else:
            explanation.append(f"remain stable at ${current_price:.2f} over the next {len(future_prices)} trading days.")
        
        explanation.append("")
        
        # Describe initial movement
        if abs(initial_pct_change) > 5:
            explanation.append(f"The model suggests a significant {'increase' if initial_pct_change > 0 else 'decrease'} of {abs(initial_pct_change):.2f}% on the first day.")
        elif abs(initial_pct_change) > 1:
            explanation.append(f"The model suggests a moderate {'increase' if initial_pct_change > 0 else 'decrease'} of {abs(initial_pct_change):.2f}% on the first day.")
        else:
            explanation.append(f"The model suggests minimal movement on the first day ({initial_pct_change:.2f}%).")
        
        explanation.append("")
        
        # Describe volatility/stability
        if volatility > 5:
            explanation.append("The prediction shows high volatility in the coming days, suggesting uncertainty.")
        elif volatility > 1:
            explanation.append("The prediction shows moderate volatility in the coming days.")
        else:
            explanation.append("The prediction shows stable prices with minimal volatility.")
        
        explanation.append("")
        
        # Add confidence assessment
        has_confidence = 'Confidence_Assessment' in predictions.columns
        if has_confidence:
            first_day_confidence = predictions['Confidence_Assessment'].iloc[1] if len(predictions) > 1 else "Unknown"
            if "High" in str(first_day_confidence):
                explanation.append("The model has high confidence in these predictions.")
            elif "Medium" in str(first_day_confidence):
                explanation.append("The model has moderate confidence in these predictions.")
            elif "Low" in str(first_day_confidence):
                explanation.append("The model has low confidence in these predictions, suggesting caution.")
            else:
                explanation.append("The model's confidence in these predictions could not be determined.")
        else:
            explanation.append("No confidence assessment is available for these predictions.")
        
        explanation.append("")
        explanation.append("Remember that stock predictions are subject to market conditions, unexpected events, and other factors that the model cannot account for.")
        
        return "\n".join(explanation)
        
    except Exception as e:
        logger.error(f"Error generating simple explanation: {str(e)}")
        return "Could not generate explanation due to an error."

def main():
    """Main execution function for the pipeline."""
    start_time = time.time()
    logger.info("Starting NVIDIA Stock Prediction Pipeline")
    
    # Parse arguments
    args = setup_argparse()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Run model (train and predict)
    success = run_model(args)
    if not success:
        logger.error("Failed to run model")
        return 1
        
    # Generate report
    success = generate_report(args)
    if not success:
        logger.warning("Failed to generate report")
        
    # Calculate and log execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 