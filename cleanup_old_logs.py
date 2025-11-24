#!/usr/bin/env python3
"""Clean up old log files in data/results, keeping only the N most recent ones."""

import os
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("data/results")
KEEP_N_RECENT = 5  # Keep the 5 most recent timestamped files

def get_timestamp_from_filename(filename: str) -> int:
    """Extract timestamp from filename like freemind_log_1763934590.csv"""
    try:
        parts = filename.stem.split("_")
        if len(parts) >= 3:
            return int(parts[-1])
    except (ValueError, IndexError):
        pass
    return 0

def cleanup_old_logs():
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return
    
    # Group files by type
    log_files = []
    live_files = []
    metadata_files = []
    judge_files = []
    
    for file in results_dir.iterdir():
        if not file.is_file():
            continue
        
        name = file.name
        if name.startswith("freemind_log_") and name.endswith(".csv") and "_latest" not in name:
            log_files.append(file)
        elif name.startswith("live_run_") and name.endswith(".jsonl"):
            live_files.append(file)
        elif name.startswith("run_metadata_") and name.endswith(".json") and "_latest" not in name:
            metadata_files.append(file)
        elif name.startswith("freemind_log_") and "_judge" in name and "_latest" not in name:
            judge_files.append(file)
    
    # Sort by timestamp (newest first)
    log_files.sort(key=lambda f: get_timestamp_from_filename(f), reverse=True)
    live_files.sort(key=lambda f: get_timestamp_from_filename(f), reverse=True)
    metadata_files.sort(key=lambda f: get_timestamp_from_filename(f), reverse=True)
    judge_files.sort(key=lambda f: get_timestamp_from_filename(f), reverse=True)
    
    # Files to keep
    keep_logs = set(log_files[:KEEP_N_RECENT])
    keep_live = set(live_files[:KEEP_N_RECENT])
    keep_metadata = set(metadata_files[:KEEP_N_RECENT])
    keep_judges = set(judge_files[:KEEP_N_RECENT])
    
    # Get timestamps of files we're keeping
    keep_timestamps = set()
    for f in keep_logs:
        ts = get_timestamp_from_filename(f)
        if ts:
            keep_timestamps.add(ts)
    
    # Also keep metadata and live files that match kept log timestamps
    for f in metadata_files:
        ts = get_timestamp_from_filename(f)
        if ts in keep_timestamps:
            keep_metadata.add(f)
    
    for f in live_files:
        ts = get_timestamp_from_filename(f)
        if ts in keep_timestamps:
            keep_live.add(f)
    
    # Files to delete
    to_delete = []
    
    for f in log_files:
        if f not in keep_logs:
            to_delete.append(f)
    
    for f in live_files:
        if f not in keep_live:
            to_delete.append(f)
    
    for f in metadata_files:
        if f not in keep_metadata:
            to_delete.append(f)
    
    for f in judge_files:
        if f not in keep_judges:
            to_delete.append(f)
    
    # Show what will be deleted
    print(f"Found {len(log_files)} log files, {len(live_files)} live files, {len(metadata_files)} metadata files, {len(judge_files)} judge files")
    print(f"Keeping {len(keep_logs)} most recent log files")
    print(f"\nFiles to delete ({len(to_delete)}):")
    for f in sorted(to_delete):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")
    
    if not to_delete:
        print("\nNo old files to clean up!")
        return
    
    # Ask for confirmation
    response = input(f"\nDelete {len(to_delete)} old files? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        total_size = 0
        deleted = 0
        for f in to_delete:
            try:
                size = f.stat().st_size
                f.unlink()
                total_size += size
                deleted += 1
            except Exception as e:
                print(f"Error deleting {f.name}: {e}")
        
        print(f"\nDeleted {deleted} files, freed {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    cleanup_old_logs()

