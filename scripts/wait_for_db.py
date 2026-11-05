import subprocess
import sys
import time


def db_ready() -> bool:
    out = subprocess.run(
        ["docker", "compose", "exec", "-T", "db", "pg_isready", "-U", "nusalens"],
        capture_output=True, text=True,
    )
    return "accepting connections" in out.stdout


def main(timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if db_ready():
            print("database accepting connections")
            return
        time.sleep(2)
    sys.stderr.write(f"database not ready within {timeout_seconds}s\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
