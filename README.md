# Project Description

This project processes CSV files containing tweets. It analyzes the sentiment of each tweet using a local language model via Ollama.

## What it does

1. Reads a CSV file with tweet data
2. Sends each tweet to Ollama for sentiment analysis
3. Uses LangGraph to orchestrate the processing
4. Prints the results with timing information

## Requirements

- Python 3
- Ollama running locally (default: http://localhost:11434)
- CSV file with tweet data

## Output

The script prints results to the console. Each row shows the tweet text preview, the sentiment result, and how long it took to process.
