import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WATCH_TARGETS = (
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "common",
    PROJECT_ROOT / "ml",
    PROJECT_ROOT / "ui",
)
POLL_INTERVAL_SECONDS = 0.5


def collect_file_state() -> dict[Path, int]:
    files = []

    for target in WATCH_TARGETS:
        files.extend(target.rglob("*.py") if target.is_dir() else [target])

    state = {}

    for file_path in files:
        try:
            state[file_path] = file_path.stat().st_mtime_ns
        except FileNotFoundError:
            pass

    return state


def start_app() -> subprocess.Popen:
    print("[dev] Starting application.", flush=True)
    return subprocess.Popen([sys.executable, "main.py"], cwd=PROJECT_ROOT)


def stop_app(process: subprocess.Popen | None):
    if process is None or process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    file_state = collect_file_state()
    process = start_app()

    try:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            current_state = collect_file_state()

            if current_state != file_state:
                file_state = current_state
                print("[dev] Change detected. Restarting.", flush=True)
                stop_app(process)
                process = start_app()
                continue

            if process is not None and process.poll() is not None:
                if process.returncode == 0:
                    return

                print("[dev] Application failed. Waiting for a file change.", flush=True)
                process = None
    except KeyboardInterrupt:
        print("\n[dev] Stopping development runner.", flush=True)
    finally:
        stop_app(process)


if __name__ == "__main__":
    main()
