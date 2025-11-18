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

## Git & DVC Controller

A simple controller is provided to streamline common Git/DVC operations.

Examples:

```bash
# Commit all changes
python tools/git_dvc_controller.py git-commit -m "update" --all

# Create a tag from prompts JSON version and push it
python tools/git_dvc_controller.py git-tag-prompts --push

# Initialize DVC and set a default remote
python tools/git_dvc_controller.py dvc-init --remote /home/ahmad/Desktop/dvc_store

# Track datasets with DVC and push artifacts
python tools/git_dvc_controller.py dvc-add "data/free tweet export.csv"
python tools/git_dvc_controller.py dvc-push

# Reproduce a DVC pipeline (if dvc.yaml is present)
python tools/git_dvc_controller.py dvc-repro
```
