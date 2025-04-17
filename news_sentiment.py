import requests
from datetime import datetime, timedelta
import ollama
import time
import pandas as pd
import os
import json
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NEWS_CACHE_FILE = 'nvidia_news_cache.json'
SENTIMENT_CACHE_FILE = 'nvidia_sentiment_cache.json'
MAX_NEWS_AGE_DAYS = 3  # Consider news from the last 3 days
OLLAMA_MODEL = "llama2"  # Default Ollama model
OLLAMA_AVAILABLE_MODELS = ["llama2", "llama3", "mistral"]  # Models to try in order of preference

# Check if Ollama is available and which models are accessible
def get_available_ollama_model():
    """
    Checks if Ollama is available and returns the best available model
    
    Returns:
        str: Name of the available model, or None if Ollama is not available
    """
    try:
        # Try to list models
        response = ollama.list()
        available_models = [model['name'] for model in response.get('models', [])]
        logger.info(f"Available Ollama models: {available_models}")
        
        # Try to find one of our preferred models
        for model in OLLAMA_AVAILABLE_MODELS:
            if model in available_models:
                logger.info(f"Using Ollama model: {model}")
                return model
                
        # If none of our preferred models are available but others are
        if available_models:
            logger.info(f"Using available Ollama model: {available_models[0]}")
            return available_models[0]
            
        logger.warning("No Ollama models available")
        return None
        
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return None

# Sample news articles about NVIDIA for demonstration purposes
# In a real application, this would be replaced with a proper API call
SAMPLE_NEWS_ARTICLES = [
    {
        "title": "NVIDIA Announces Next-Generation GPU Architecture",
        "text": "NVIDIA has unveiled its next-generation GPU architecture, promising significant performance improvements for AI and graphics applications. The new chips are expected to deliver 2x the performance of current generation models while consuming less power.",
        "url": "https://example.com/nvidia-new-gpu",
        "source": "Tech News",
        "date": datetime.now() - timedelta(days=1)
    },
    {
        "title": "NVIDIA Stock Reaches All-Time High on AI Demand",
        "text": "NVIDIA shares reached an all-time high today as demand for AI chips continues to surge. The company has become the leading supplier of GPUs for training large language models and other AI applications, driving significant revenue growth.",
        "url": "https://example.com/nvidia-stock-high",
        "source": "Financial News",
        "date": datetime.now() - timedelta(days=2)
    },
    {
        "title": "NVIDIA Faces Challenges from Intel and AMD in AI Chip Market",
        "text": "Competition in the AI chip market is heating up as Intel and AMD introduce new processors targeting the same market as NVIDIA's GPUs. Analysts suggest this could put pressure on NVIDIA's margins and market share in the coming quarters.",
        "url": "https://example.com/nvidia-competition",
        "source": "Market Analysis",
        "date": datetime.now() - timedelta(days=3)
    },
    {
        "title": "NVIDIA Expands Partnership with Microsoft for Cloud AI",
        "text": "NVIDIA and Microsoft have announced an expanded partnership to bring more NVIDIA GPUs to Azure cloud services. The collaboration aims to meet growing demand for AI computing resources and will include the latest NVIDIA hardware.",
        "url": "https://example.com/nvidia-microsoft",
        "source": "Business News",
        "date": datetime.now() - timedelta(days=1)
    },
    {
        "title": "NVIDIA Announces Record Quarterly Results",
        "text": "NVIDIA reported record quarterly revenue, exceeding analyst expectations by 15%. The company cited strong demand for its data center GPUs used in AI applications as the primary driver of growth.",
        "url": "https://example.com/nvidia-earnings",
        "source": "Earnings Report",
        "date": datetime.now() - timedelta(days=2)
    },
    {
        "title": "NVIDIA Introduces New AI Tools for Content Creation",
        "text": "NVIDIA has launched a suite of AI-powered tools designed for content creators, including advanced video editing capabilities and real-time rendering enhancements. These tools leverage NVIDIA's GPU technology to accelerate creative workflows.",
        "url": "https://example.com/nvidia-ai-tools",
        "source": "Technology News",
        "date": datetime.now() - timedelta(hours=12)
    },
    {
        "title": "NVIDIA Partners with Leading Automotive Companies for Self-Driving Technology",
        "text": "NVIDIA announced new partnerships with several major automotive manufacturers to supply chips and software for autonomous driving systems. The company's DRIVE platform will be integrated into future vehicle models, expanding NVIDIA's presence in the automotive sector.",
        "url": "https://example.com/nvidia-automotive",
        "source": "Automotive Industry News",
        "date": datetime.now() - timedelta(hours=36)
    }
]

def fetch_latest_nvidia_news(max_articles=5):
    """
    Fetch (or simulate) the latest news about NVIDIA
    
    Parameters:
        max_articles (int): Maximum number of articles to return
        
    Returns:
        list: List of dictionaries containing news data
    """
    logger.info(f"Fetching latest NVIDIA news (max {max_articles} articles)...")
    
    # Check if we have cached news that's still recent
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, 'r') as f:
                news_cache = json.load(f)
            
            # Convert date strings back to datetime objects
            for article in news_cache:
                if 'date' in article:
                    try:
                        article['date'] = datetime.fromisoformat(article['date'])
                    except (ValueError, TypeError):
                        # If date conversion fails, use current time minus 1 day
                        article['date'] = datetime.now() - timedelta(days=1)
            
            # Filter for recent news
            cutoff_date = datetime.now() - timedelta(days=MAX_NEWS_AGE_DAYS)
            recent_news = [article for article in news_cache 
                           if 'date' in article and article['date'] >= cutoff_date]
            
            if recent_news:
                logger.info(f"Using {len(recent_news)} cached news articles")
                return recent_news[:max_articles]
            else:
                logger.info("Cached news is too old or invalid, generating fresh news")
        except Exception as e:
            logger.warning(f"Error reading news cache: {e}")
    
    # For demonstration purposes, we'll use our sample articles
    # In a real application, this would be replaced with a proper API call
    
    # Ensure we have enough sample articles
    if not SAMPLE_NEWS_ARTICLES:
        logger.warning("No sample news articles available")
        return []
    
    # Randomize which sample articles to use
    try:
        selected_articles = random.sample(SAMPLE_NEWS_ARTICLES, 
                                        min(max_articles, len(SAMPLE_NEWS_ARTICLES)))
    except Exception as e:
        logger.error(f"Error selecting random articles: {e}")
        # If random sampling fails, just take the first few
        selected_articles = SAMPLE_NEWS_ARTICLES[:min(max_articles, len(SAMPLE_NEWS_ARTICLES))]
    
    # Update the date of each article to be more recent
    current_date = datetime.now()
    for i, article in enumerate(selected_articles):
        # Assign dates from today to a few days ago
        days_ago = i % 3  # 0, 1, or 2 days ago
        article['date'] = current_date - timedelta(days=days_ago)
    
    # Sort by date (most recent first)
    selected_articles = sorted(selected_articles, key=lambda x: x.get('date', current_date), reverse=True)
    
    # Save to cache
    try:
        cache_articles = []
        for article in selected_articles:
            # Create a copy to avoid modifying the original
            cache_article = article.copy()
            # Convert datetime objects to ISO format strings for JSON serialization
            if 'date' in cache_article:
                cache_article['date'] = cache_article['date'].isoformat()
            cache_articles.append(cache_article)
        
        with open(NEWS_CACHE_FILE, 'w') as f:
            json.dump(cache_articles, f)
        
        logger.info(f"Cached {len(selected_articles)} news articles")
    except Exception as e:
        logger.error(f"Error caching news articles: {e}")
    
    return selected_articles

def analyze_sentiment_with_ollama(news_items, model_name=None):
    """
    Analyze sentiment of news articles using Ollama
    
    Parameters:
        news_items (list): List of news article dictionaries
        model_name (str): Specific Ollama model to use, or None to auto-detect
        
    Returns:
        list: News items with sentiment data added
    """
    logger.info(f"Analyzing sentiment for {len(news_items)} news items...")
    
    # Check which model to use
    if model_name is None:
        model_name = get_available_ollama_model() or OLLAMA_MODEL
    
    # Check for cached sentiment analysis
    sentiment_cache = {}
    if os.path.exists(SENTIMENT_CACHE_FILE):
        try:
            with open(SENTIMENT_CACHE_FILE, 'r') as f:
                sentiment_cache = json.load(f)
            logger.info(f"Loaded {len(sentiment_cache)} cached sentiment analyses")
        except Exception as e:
            logger.warning(f"Error reading sentiment cache: {e}")
            sentiment_cache = {}  # Ensure we have an empty dict if file read fails
    
    # Process each news item
    for item in news_items:
        try:
            # Create a URL key for the cache
            url_key = item.get('url', '')
            if not url_key:
                # If no URL, use title as key
                url_key = item.get('title', f'item_{id(item)}')
            
            # Skip if we already have sentiment for this URL
            if url_key in sentiment_cache:
                item.update(sentiment_cache[url_key])
                logger.info(f"Using cached sentiment for: {item.get('title', 'Unknown title')}")
                continue
            
            # Create prompt for Ollama
            prompt = f"""
            You are a financial expert specializing in NVIDIA stock analysis. Please analyze this news article about NVIDIA and determine its sentiment impact on NVIDIA's stock price. 
            
            Title: {item.get('title', 'No title')}
            
            Article text:
            {item.get('text', 'No text available')}
            
            Based on this news, please:
            1. Classify the sentiment as "positive", "negative", or "neutral" for NVIDIA stock
            2. Assign a score from -5 to +5 where:
               * -5 means extremely negative for the stock price
               * 0 means neutral
               * +5 means extremely positive for the stock price
            3. Provide a brief explanation (1-2 sentences) of your reasoning
            
            Format your answer as:
            SENTIMENT: [positive/negative/neutral]
            SCORE: [number]
            EXPLANATION: [brief explanation]
            """
            
            try:
                # Call Ollama API
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract response
                response_text = response['message']['content']
                logger.info(f"Received response from Ollama model {model_name}")
                
                # Parse the response
                sentiment = "neutral"
                score = 0.0
                explanation = "No explanation provided"
                
                # Extract sentiment
                if "SENTIMENT:" in response_text:
                    sentiment_line = response_text.split("SENTIMENT:")[1].split("\n")[0].strip().lower()
                    if any(s in sentiment_line for s in ["positive", "negative", "neutral"]):
                        if "positive" in sentiment_line:
                            sentiment = "positive"
                        elif "negative" in sentiment_line:
                            sentiment = "negative"
                        else:
                            sentiment = "neutral"
                
                # Extract score
                if "SCORE:" in response_text:
                    try:
                        score_line = response_text.split("SCORE:")[1].split("\n")[0].strip()
                        # Extract the first number found in the line
                        import re
                        score_match = re.search(r'[-+]?\d*\.?\d+', score_line)
                        if score_match:
                            score = float(score_match.group())
                        # Clamp score to valid range
                        score = max(-5.0, min(5.0, score))
                    except Exception as e:
                        logger.warning(f"Error parsing sentiment score: {e}")
                        # Use default score based on sentiment
                        if sentiment == "positive":
                            score = 3.0
                        elif sentiment == "negative":
                            score = -3.0
                        else:
                            score = 0.0
                
                # Extract explanation
                if "EXPLANATION:" in response_text:
                    explanation = response_text.split("EXPLANATION:")[1].strip()
                
                # Add to item
                item['sentiment'] = sentiment
                item['score'] = score
                item['explanation'] = explanation
                
                # Add to cache
                sentiment_cache[url_key] = {
                    'sentiment': sentiment,
                    'score': score,
                    'explanation': explanation
                }
                
                logger.info(f"Analyzed: {item.get('title', 'Unknown title')} - {sentiment} ({score})")
                
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                # Use fallback sentiment values based on title analysis
                title = item.get('title', '').lower()
                if any(word in title for word in ['high', 'record', 'growth', 'expands', 'partnership', 'introduces']):
                    sentiment = "positive"
                    score = 3.5
                    explanation = "Title suggests positive business developments."
                elif any(word in title for word in ['challenges', 'competition', 'pressure', 'concerns']):
                    sentiment = "negative"
                    score = -2.0
                    explanation = "Title indicates potential challenges for the company."
                else:
                    sentiment = "neutral"
                    score = 0.0
                    explanation = "Default neutral assessment as API call failed."
                
                item['sentiment'] = sentiment
                item['score'] = score
                item['explanation'] = explanation
                
                # Add to cache
                sentiment_cache[url_key] = {
                    'sentiment': sentiment,
                    'score': score,
                    'explanation': explanation
                }
                
                logger.info(f"Using fallback analysis for: {item.get('title', 'Unknown title')} - {sentiment} ({score})")
                
        except Exception as e:
            logger.error(f"Error processing news item: {e}")
            # Ensure the item has sentiment data even in case of error
            item['sentiment'] = "neutral"
            item['score'] = 0.0
            item['explanation'] = f"Error in sentiment analysis: {str(e)}"
    
    # Save updated sentiment cache
    try:
        with open(SENTIMENT_CACHE_FILE, 'w') as f:
            json.dump(sentiment_cache, f)
        logger.info(f"Updated sentiment cache with {len(sentiment_cache)} entries")
    except Exception as e:
        logger.error(f"Error saving sentiment cache: {e}")
    
    return news_items

def calculate_news_sentiment_score(max_articles=5, model_name=None):
    """
    Calculate an overall sentiment score based on recent news
    
    Parameters:
        max_articles (int): Maximum number of articles to analyze
        model_name (str): Specific Ollama model to use, or None to auto-detect
        
    Returns:
        tuple: (float sentiment_score, list analyzed_articles)
    """
    # Fetch and analyze news
    try:
        news_items = fetch_latest_nvidia_news(max_articles=max_articles)
        
        if not news_items:
            logger.warning("No news items found")
            return 0.0, []
        
        news_items = analyze_sentiment_with_ollama(news_items, model_name)
        
        # Calculate weighted average sentiment score
        # More recent news has more weight
        total_weight = 0
        weighted_score = 0
        
        for i, item in enumerate(news_items):
            # Skip items that don't have a score
            if 'score' not in item:
                continue
                
            # Ensure the score is a float
            try:
                score = float(item['score'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid sentiment score for {item.get('title', 'unknown article')}: {item.get('score')}")
                continue
                
            # More recent articles get more weight (reverse index + 1)
            weight = len(news_items) - i
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize to a value between -1 and 1
        if total_weight > 0:
            normalized_score = weighted_score / (total_weight * 5)  # Divide by max score (5)
        else:
            normalized_score = 0.0
        
        logger.info(f"Overall news sentiment score: {normalized_score:.4f} (based on {len(news_items)} articles)")
        
        # Log individual articles for debugging
        for item in news_items:
            if 'title' in item and 'score' in item and 'sentiment' in item:
                title = item['title']
                # Truncate title if too long
                if len(title) > 50:
                    title = title[:47] + "..."
                logger.info(f"Article: {title} - {item['sentiment']} ({item['score']})")
        
        return normalized_score, news_items
        
    except Exception as e:
        logger.error(f"Error calculating news sentiment score: {e}")
        return 0.0, []

def calculate_sentiment_adjustment(sentiment_score, max_impact=0.2):
    """
    Convert a sentiment score into a price adjustment factor
    
    Parameters:
        sentiment_score (float): Sentiment score between -1 and 1
        max_impact (float): Maximum percentage impact on price (default 0.2 = 20%)
        
    Returns:
        float: Adjustment factor to multiply with price predictions
    """
    # Convert sentiment score (-1 to 1) to adjustment factor (0.8 to 1.2 for 20% max impact)
    adjustment = 1.0 + (sentiment_score * max_impact)
    logger.info(f"Applying adjustment factor: {adjustment:.4f}")
    return adjustment

if __name__ == "__main__":
    # Test the functionality
    # First, check if Ollama is available and which model to use
    model_name = get_available_ollama_model()
    if model_name:
        print(f"Using Ollama model: {model_name}")
    else:
        print("Ollama not available, using fallback sentiment analysis")
    
    sentiment_score, articles = calculate_news_sentiment_score(max_articles=3, model_name=model_name)
    print(f"Overall sentiment score: {sentiment_score:.4f}")
    print("\nAnalyzed articles:")
    for article in articles:
        print(f"\n{article['title']}")
        print(f"Sentiment: {article['sentiment']} (Score: {article['score']})")
        print(f"Explanation: {article['explanation']}") 