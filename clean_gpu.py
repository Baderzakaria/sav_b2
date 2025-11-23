

import argparse
import subprocess
import sys
from typing import List, Tuple, Optional

def run_nvidia_smi_query(query: str, format_str: str = "csv,noheader,nounits") -> str:
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

                mem_str = parts[2].strip()
                try:
                    mem_mb = float(mem_str)
                except ValueError:

                    mem_mb = float(''.join(c for c in mem_str if c.isdigit() or c == '.'))
                processes.append((pid, name, mem_mb))
        except (ValueError, IndexError) as e:
            continue

    return processes

def get_gpu_status() -> dict:
    try:

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
    try:
        signal = "SIGKILL" if force else "SIGTERM"
        subprocess.run(["kill", f"-{9 if force else 15}", str(pid)], check=True, timeout=5)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def print_gpu_status():
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

  python clean_gpu.py --status

  python clean_gpu.py --kill ollama

  python clean_gpu.py --kill-all

  python clean_gpu.py --kill-all --force

