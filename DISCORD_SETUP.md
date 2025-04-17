# Discord Integration Setup Guide

This guide will walk you through setting up Discord notifications for the NVIDIA Stock Price Prediction System.

## Creating a Discord Webhook

1. Open Discord and go to the server where you want to receive notifications
2. Select or create a channel for the notifications
3. Right-click the channel and select "Edit Channel"
4. Click on "Integrations"
5. Click on "Create Webhook"
6. Give your webhook a name (e.g., "Stock Predictions")
7. (Optional) Set a custom avatar
8. Click "Copy Webhook URL" - you'll need this URL for configuration

## Configuration

### Basic Setup

1. Save your webhook URL in a secure location
2. Configure the notification system:

```bash
python discord_notify_runner.py --webhook YOUR_WEBHOOK_URL
```

Replace `YOUR_WEBHOOK_URL` with the URL you copied from Discord.

### Testing the Setup

To verify your configuration is working:

```bash
python discord_notify_runner.py --test
```

This will send a test message to your Discord channel.

## Notification Options

### Daily Scheduled Notifications

Set up automated daily notifications:

```bash
python discord_notify_runner.py --schedule 08:00
```

This will send predictions at 8:00 AM every trading day. You can adjust the time as needed.

### Manual Notifications

Send predictions on demand:

```bash
# Run predictions and send notifications
python run_prediction_discord.py --webhook YOUR_WEBHOOK_URL

# Send existing predictions without running new ones
python run_prediction_discord.py --send-only

# Run predictions without training
python run_prediction_discord.py --no-train
```

## Notification Content

Each notification includes:
- Current stock price
- Predicted prices for the next 5 trading days
- Confidence levels for each prediction
- Two charts:
  1. Standard prediction chart showing historical and predicted prices
  2. Candlestick chart with technical indicators

## Customizing Notifications

### Changing the Prediction Window

Modify the number of days to predict:

```bash
python run_prediction_discord.py --days 10
```

### Adjusting Chart Appearance

The system generates two types of charts:
1. Line chart (`NVDA_prediction_plot_*.png`)
2. Candlestick chart (`NVDA_candlestick_chart.png`)

Both charts are automatically included in Discord notifications.

## Troubleshooting

### Common Issues

1. **Webhook Invalid**
   - Verify the webhook URL is correct
   - Check if the webhook hasn't been deleted in Discord
   - Try creating a new webhook

2. **Missing Charts**
   - Ensure the prediction process completed successfully
   - Check the `reports` directory for generated files
   - Review `discord_notify.log` for errors

3. **Scheduled Notifications Not Working**
   - Verify the system time is correct
   - Check if the script is running with proper permissions
   - Review `prediction_discord.log` for scheduling issues

### Logs

Check these log files for troubleshooting:
- `discord_notify.log`: Discord notification system logs
- `prediction_discord.log`: Discord prediction runner logs
- `pipeline.log`: Main pipeline execution logs

## Security Considerations

1. **Webhook URL Protection**
   - Keep your webhook URL private
   - Don't commit it to version control
   - Consider using environment variables

2. **Rate Limiting**
   - Discord has rate limits for webhooks
   - The system includes built-in rate limiting
   - Avoid sending too many notifications in quick succession

## Advanced Usage

### Running in Background

For continuous operation, consider using a process manager:

```bash
# Using nohup
nohup python discord_notify_runner.py --schedule 08:00 &

# Using screen
screen -S stock_notifications
python discord_notify_runner.py --schedule 08:00
# Press Ctrl+A, then D to detach
```

### Multiple Channels

You can set up notifications for different channels:
- Create separate webhooks for each channel
- Run multiple instances with different configurations
- Use different schedules or prediction parameters

## Support

If you encounter issues:
1. Check the log files mentioned above
2. Verify your Discord server permissions
3. Ensure all dependencies are correctly installed
4. Try running with the `--test` flag to isolate issues 