

import json
import sys
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

RESULTS_DIR = Path("data/results")

def load_run_data(run_id: str):
    meta_path = RESULTS_DIR / f"run_metadata_{run_id.split('_')[-1]}.json"
    csv_path = RESULTS_DIR / f"freemind_log_{run_id.split('_')[-1]}.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    with meta_path.open() as f:
        metadata = json.load(f)

    df = pd.read_csv(csv_path)
    return metadata, df

def compare_runs(run_id_1: str, run_id_2: str):
    meta1, df1 = load_run_data(run_id_1)
    meta2, df2 = load_run_data(run_id_2)

    threads1 = meta1.get("metadata_tags", {}).get("gpu_threads", 1)
    threads2 = meta2.get("metadata_tags", {}).get("gpu_threads", 1)

    print("=" * 80)
    print("GPU THREADS COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"\nRun 1 ({threads1} GPU threads): {run_id_1}")
    print(f"Run 2 ({threads2} GPU threads): {run_id_2}")
    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    print(f"{'Metric':<30} {f'{threads1} Threads':<15} {f'{threads2} Threads':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Total Duration (sec)':<30} {meta1['total_duration_sec']:<15.2f} {meta2['total_duration_sec']:<15.2f} {meta1['total_duration_sec'] - meta2['total_duration_sec']:+.2f}")
    print(f"{'Avg sec/row':<30} {meta1['avg_sec_per_row']:<15.2f} {meta2['avg_sec_per_row']:<15.2f} {meta1['avg_sec_per_row'] - meta2['avg_sec_per_row']:+.2f}")
    print(f"{'Rows processed':<30} {meta1['rows_processed']:<15} {meta2['rows_processed']:<15} {meta1['rows_processed'] - meta2['rows_processed']:+d}")

    print("\n" + "-" * 80)
    print("PER-ROW LATENCY STATISTICS")
    print("-" * 80)
    print(f"{'Statistic':<20} {f'{threads1} Threads (sec)':<20} {f'{threads2} Threads (sec)':<20} {'Difference':<20}")
    print("-" * 80)
    print(f"{'Mean':<20} {df1['elapsed_sec'].mean():<20.3f} {df2['elapsed_sec'].mean():<20.3f} {df1['elapsed_sec'].mean() - df2['elapsed_sec'].mean():+.3f}")
    print(f"{'Median':<20} {df1['elapsed_sec'].median():<20.3f} {df2['elapsed_sec'].median():<20.3f} {df1['elapsed_sec'].median() - df2['elapsed_sec'].median():+.3f}")
    print(f"{'Min':<20} {df1['elapsed_sec'].min():<20.3f} {df2['elapsed_sec'].min():<20.3f} {df1['elapsed_sec'].min() - df2['elapsed_sec'].min():+.3f}")
    print(f"{'Max':<20} {df1['elapsed_sec'].max():<20.3f} {df2['elapsed_sec'].max():<20.3f} {df1['elapsed_sec'].max() - df2['elapsed_sec'].max():+.3f}")
    print(f"{'Std Dev':<20} {df1['elapsed_sec'].std():<20.3f} {df2['elapsed_sec'].std():<20.3f} {df1['elapsed_sec'].std() - df2['elapsed_sec'].std():+.3f}")

    if 'gpu_util' in df1.columns and 'gpu_util' in df2.columns:
        print("\n" + "-" * 80)
        print("GPU UTILIZATION STATISTICS")
        print("-" * 80)
        print(f"{'Statistic':<20} {f'{threads1} Threads (%)':<20} {f'{threads2} Threads (%)':<20} {'Difference':<20}")
        print("-" * 80)
        print(f"{'Mean':<20} {df1['gpu_util'].mean():<20.1f} {df2['gpu_util'].mean():<20.1f} {df1['gpu_util'].mean() - df2['gpu_util'].mean():+.1f}")
        print(f"{'Median':<20} {df1['gpu_util'].median():<20.1f} {df2['gpu_util'].median():<20.1f} {df1['gpu_util'].median() - df2['gpu_util'].median():+.1f}")
        print(f"{'Max':<20} {df1['gpu_util'].max():<20.1f} {df2['gpu_util'].max():<20.1f} {df1['gpu_util'].max() - df2['gpu_util'].max():+.1f}")

    print("\n" + "-" * 80)
    print("CONCLUSION")
    print("-" * 80)
    if meta1['total_duration_sec'] < meta2['total_duration_sec']:
        speedup = (1 - meta1['total_duration_sec'] / meta2['total_duration_sec']) * 100
        print(f"✅ {threads1}-thread run is {speedup:.1f}% FASTER ({meta2['total_duration_sec'] - meta1['total_duration_sec']:.2f}s saved)")
    else:
        slowdown = (meta1['total_duration_sec'] / meta2['total_duration_sec'] - 1) * 100
        print(f"❌ {threads1}-thread run is {slowdown:.1f}% SLOWER ({meta1['total_duration_sec'] - meta2['total_duration_sec']:.2f}s overhead)")

    print("\n💡 The 'gpu_threads' parameter controls GPU watchdog polling workers,")
    print("   NOT the number of parallel inference workers. The agents already run")
    print("   in parallel via LangGraph, so increasing watchdog threads only adds")
    print("   overhead without improving inference throughput.")

    if HAS_PLOTLY:
        create_comparison_charts(df1, df2, run_id_1, run_id_2, threads1, threads2)
    else:
        print("\n⚠️  Plotly not installed. Install with 'pip install plotly' to generate charts.")

def create_comparison_charts(df1: pd.DataFrame, df2: pd.DataFrame, 
                            run_id_1: str, run_id_2: str, 
                            threads1: int, threads2: int):

    df1_plot = df1.copy()
    df1_plot['run'] = f"{threads1} threads ({run_id_1.split('_')[-1]})"
    df2_plot = df2.copy()
    df2_plot['run'] = f"{threads2} threads ({run_id_2.split('_')[-1]})"

    combined = pd.concat([df1_plot, df2_plot], ignore_index=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Per-Row Latency Over Time',
            'Latency Distribution (Box Plot)',
            'GPU Utilization Over Time',
            'GPU Utilization Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    for run_label in combined['run'].unique():
        run_data = combined[combined['run'] == run_label]
        fig.add_trace(
            go.Scatter(
                x=run_data['row_index'],
                y=run_data['elapsed_sec'],
                mode='lines+markers',
                name=run_label,
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

    for run_label in combined['run'].unique():
        run_data = combined[combined['run'] == run_label]
        fig.add_trace(
            go.Box(
                y=run_data['elapsed_sec'],
                name=run_label,
                boxmean='sd'
            ),
            row=1, col=2
        )

    if 'gpu_util' in combined.columns:
        for run_label in combined['run'].unique():
            run_data = combined[combined['run'] == run_label]
            fig.add_trace(
                go.Scatter(
                    x=run_data['row_index'],
                    y=run_data['gpu_util'],
                    mode='lines+markers',
                    name=run_label,
                    line=dict(width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2, col=1
            )

        for run_label in combined['run'].unique():
            run_data = combined[combined['run'] == run_label]
            fig.add_trace(
                go.Box(
                    y=run_data['gpu_util'],
                    name=run_label,
                    boxmean='sd',
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.update_xaxes(title_text="Row Index", row=1, col=1)
    fig.update_yaxes(title_text="Latency (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (seconds)", row=1, col=2)

    if 'gpu_util' in combined.columns:
        fig.update_xaxes(title_text="Row Index", row=2, col=1)
        fig.update_yaxes(title_text="GPU Utilization (%)", row=2, col=1)
        fig.update_yaxes(title_text="GPU Utilization (%)", row=2, col=2)

    fig.update_layout(
        height=800,
        title_text=f"Performance Comparison: {threads1} vs {threads2} GPU Watchdog Threads",
        showlegend=True
    )

    output_path = RESULTS_DIR / f"comparison_{run_id_1.split('_')[-1]}_vs_{run_id_2.split('_')[-1]}.html"
    fig.write_html(str(output_path))
    print(f"\n📊 Interactive chart saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_runs.py <run_id_1> <run_id_2>")
        print("Example: python compare_runs.py run_1763656442 run_1763655076")
        sys.exit(1)

    compare_runs(sys.argv[1], sys.argv[2])

