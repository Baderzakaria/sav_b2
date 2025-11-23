

from orchestrator import PipelineConfig, run_pipeline

config = PipelineConfig(
    input_csv="test_single_tweet.csv",
    max_rows=1,
    model_name="mistral:7b",
    enable_model_warmup=True,
    enable_live_log=True,
)

if __name__ == "__main__":
    print("=" * 60)
    print(f"Running FreeMind agents on test tweet (Model: {config.model_name}):")
    print("  'Merci Free pour la coupure du dimanche 🙃'")
    print("=" * 60)
    print()

    result = run_pipeline(config)

    print()
    print("=" * 60)
    print("Test completed!")
    print(f"Results saved to: {result['log_csv']}")
    print("=" * 60)

