#!/usr/bin/env python3
"""GPU memory cleanup utility.

This script helps free up GPU memory by:
1. Listing processes using GPU memory
2. Killing specific processes (e.g., ollama)
3. Killing all GPU processes (with confirmation)
4. Showing GPU status before/after cleanup
"""

import argparse
import subprocess
import sys
from typing import List, Tuple, Optional


def run_nvidia_smi_query(query: str, format_str: str = "csv,noheader,nounits") -> str:
    """Run nvidia-smi query and return output."""
    try:
        cmd = ["nvidia-smi", f"--query-{query}", f"--format={format_str}"]
        result = subprocess.check_output(cmd, text=True, timeout=5)
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return "Error: nvidia-smi not found. Is NVIDIA driver installed?"
    except Exception as e:
        return f"Error: {e}"


def get_gpu_processes() -> List[Tuple[int, str, float]]:
    """Get list of processes using GPU memory.
    
    Returns:
        List of (pid, process_name, memory_mb) tuples
    """
    output = run_nvidia_smi_query("compute-apps=pid,process_name,used_memory")
    if not output or output.startswith("Error") or "No running processes" in output:
        return []
    
    processes = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        try:
            parts = line.split(", ")
            if len(parts) >= 3:
                pid = int(parts[0])
                name = parts[1]
                # Memory might be in MB or with units, try to parse
                mem_str = parts[2].strip()
                try:
                    mem_mb = float(mem_str)
                except ValueError:
                    # Try to extract number if it has units
                    mem_mb = float(''.join(c for c in mem_str if c.isdigit() or c == '.'))
                processes.append((pid, name, mem_mb))
        except (ValueError, IndexError) as e:
            continue
    
    return processes


def get_gpu_status() -> dict:
    """Get current GPU status."""
    try:
        # Get GPU utilization and memory
        util_output = run_nvidia_smi_query("gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu")
        if util_output.startswith("Error"):
            return {"error": util_output}
        
        lines = util_output.split("\n")
        if lines:
            parts = lines[0].split(", ")
            if len(parts) >= 5:
                return {
                    "gpu_util": parts[0],
                    "mem_util": parts[1],
                    "mem_used": parts[2],
                    "mem_total": parts[3],
                    "temp": parts[4],
                }
    except Exception as e:
        return {"error": str(e)}
    
    return {}


def kill_process(pid: int, force: bool = False) -> bool:
    """Kill a process by PID."""
    try:
        signal = "SIGKILL" if force else "SIGTERM"
        subprocess.run(["kill", f"-{9 if force else 15}", str(pid)], check=True, timeout=5)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def print_gpu_status():
    """Print current GPU status."""
    print("\n" + "="*60)
    print("GPU Status:")
    print("="*60)
    
    status = get_gpu_status()
    if "error" in status:
        print(f"  {status['error']}")
    else:
        print(f"  GPU Utilization: {status.get('gpu_util', 'N/A')}%")
        print(f"  Memory Utilization: {status.get('mem_util', 'N/A')}%")
        print(f"  Memory Used: {status.get('mem_used', 'N/A')} MB")
        print(f"  Memory Total: {status.get('mem_total', 'N/A')} MB")
        print(f"  Temperature: {status.get('temp', 'N/A')}°C")
    
    processes = get_gpu_processes()
    if processes:
        print(f"\n  Processes using GPU ({len(processes)}):")
        print(f"  {'PID':<10} {'Process':<30} {'Memory (MB)':<15}")
        print("  " + "-"*55)
        for pid, name, mem_mb in processes:
            print(f"  {pid:<10} {name[:28]:<30} {mem_mb:<15.1f}")
    else:
        print("\n  No processes currently using GPU memory.")
    
    print("="*60 + "\n")


def clean_gpu(process_name: Optional[str] = None, kill_all: bool = False, force: bool = False) -> int:
    """Clean GPU memory by killing processes.
    
    Args:
        process_name: If provided, only kill processes matching this name
        kill_all: If True, kill all GPU processes
        force: If True, use SIGKILL instead of SIGTERM
    
    Returns:
        Number of processes killed
    """
    processes = get_gpu_processes()
    if not processes:
        print("No processes using GPU memory.")
        return 0
    
    to_kill = []
    if kill_all:
        to_kill = processes
        if not force:
            print(f"\n⚠️  WARNING: About to kill {len(to_kill)} process(es) using GPU:")
            for pid, name, mem_mb in to_kill:
                print(f"   PID {pid}: {name} ({mem_mb:.1f} MB)")
            response = input("\nContinue? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                print("Cancelled.")
                return 0
    elif process_name:
        to_kill = [(pid, name, mem) for pid, name, mem in processes if process_name.lower() in name.lower()]
        if not to_kill:
            print(f"No processes found matching '{process_name}'")
            return 0
    else:
        print("Please specify --process-name or --kill-all")
        return 0
    
    killed = 0
    for pid, name, mem_mb in to_kill:
        print(f"Killing PID {pid} ({name}, {mem_mb:.1f} MB)...", end=" ")
        if kill_process(pid, force=force):
            print("✓")
            killed += 1
        else:
            print("✗ Failed")
    
    return killed


def main():
    parser = argparse.ArgumentParser(
        description="Clean GPU memory by killing processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show GPU status
  python clean_gpu.py --status

  # Kill ollama process
  python clean_gpu.py --kill ollama

  # Kill all GPU processes (with confirmation)
  python clean_gpu.py --kill-all

  # Force kill all GPU processes (no confirmation)
  python clean_gpu.py --kill-all --force
        """
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current GPU status and processes"
    )
    parser.add_argument(
        "--kill", "-k",
        type=str,
        metavar="PROCESS_NAME",
        help="Kill processes matching this name (e.g., 'ollama')"
    )
    parser.add_argument(
        "--kill-all", "-a",
        action="store_true",
        help="Kill all processes using GPU memory"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force kill (SIGKILL) instead of graceful termination (SIGTERM)"
    )
    
    args = parser.parse_args()
    
    # Default to showing status if no action specified
    if not any([args.status, args.kill, args.kill_all]):
        print_gpu_status()
        return
    
    if args.status:
        print_gpu_status()
        return
    
    # Show status before cleanup
    print("\n📊 GPU Status BEFORE cleanup:")
    print_gpu_status()
    
    # Perform cleanup
    if args.kill:
        killed = clean_gpu(process_name=args.kill, force=args.force)
        print(f"\n✅ Killed {killed} process(es) matching '{args.kill}'")
    elif args.kill_all:
        killed = clean_gpu(kill_all=True, force=args.force)
        print(f"\n✅ Killed {killed} process(es)")
    
    # Show status after cleanup
    if killed > 0:
        import time
        print("\n⏳ Waiting 2 seconds for processes to terminate...")
        time.sleep(2)
        print("\n📊 GPU Status AFTER cleanup:")
        print_gpu_status()


if __name__ == "__main__":
    main()

