from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pattern counts for the variables already compiled. Growth is decelerating
# (16.3x, 24.1x, 11.8x, 5.4x), so "at or below trend" is read as: step 3 must
# not re-accelerate past the most recent ratio. That bar also coincides with
# what can finish overnight -- step 4 ran ~1.14M patterns in roughly 1.5-2h, so
# ~6.2M is about a 10h step at the same rate.
HISTORY = [46, 751, 18061, 212878, 1144270]
TREND_RATIO = 5.4
THRESHOLD = int(HISTORY[-1] * TREND_RATIO)


def tail(path: Path, n: int = 4000) -> str:
    """Last `n` lines of the log, or empty string if it is not there yet."""
    if not path.exists():
        return ""
    out = subprocess.run(
        ["tail", "-n", str(n), str(path)], capture_output=True, text=True
    )
    return out.stdout


def step_completed(log: str, step: int) -> bool:
    """True once the compiler has printed `step`'s summary line."""
    return re.search(rf"^step={step} var=", log, re.M) is not None


def step_pattern_count(log: str, step: int) -> int | None:
    """Pattern count for `step`, read from its enumeration lines.

    Available as soon as enumeration finishes, which is long before the step's
    expansion completes -- that is the whole point, since expansion is what we
    may need to abort.
    """
    section = log.split(f"# starting step={step} ")
    if len(section) < 2:
        return None
    body = section[1]
    hits = re.findall(r"cumulative_minimal=(\d+)", body)
    if not hits:
        return None
    if "parallel workers" not in body and "orbit-enum size=9" not in body:
        return None
    return int(hits[-1])


def kill_track_a() -> None:
    """Stop the compile without touching the tmux session itself."""
    subprocess.run(
        ["pkill", "-f", "examples/compile_sudoku_group.py --size 9"], check=False
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gate Track A's entry into step 3 on the agreed conditions."
    )
    p.add_argument("--log", type=Path, default=ROOT / "logs" / "trackA_group9.log")
    p.add_argument("--decision", type=Path,
                   default=ROOT / "logs" / "watchdog_decision.log")
    p.add_argument("--threshold", type=int, default=THRESHOLD)
    p.add_argument("--poll", type=int, default=30)
    args = p.parse_args()

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(args.decision, "a") as fh:
            fh.write(line + "\n")

    say(f"watchdog armed; step-3 pattern threshold = {args.threshold}")

    while True:
        log = tail(args.log)
        if step_completed(log, 4):
            say("step 4 COMPLETE")
            break
        if not log:
            say("log missing; aborting watchdog")
            return
        time.sleep(args.poll)

    m = re.search(r"^step=4 var=\S+ .*?expand_elapsed_s=([0-9.]+)", tail(args.log), re.M)
    if m:
        say(f"step 4 expansion wall time = {m.group(1)}s")

    while True:
        log = tail(args.log)
        n = step_pattern_count(log, 3)
        if n is not None:
            say(f"step 3 pattern count = {n} (threshold {args.threshold})")
            if n > args.threshold:
                say("ABOVE TREND -> halting Track A before expansion")
                kill_track_a()
                say("Track A stopped; awaiting human decision")
            else:
                ratio = n / HISTORY[-1]
                say(f"at/below trend ({ratio:.1f}x vs {TREND_RATIO}x) -> "
                    f"proceeding into step 3 overnight")
            return
        if not re.search(r"# starting step=3 ", log) and step_completed(log, 3):
            say("step 3 already complete; nothing to gate")
            return
        if "Traceback" in log or "MemoryError" in log:
            say("Track A crashed; watchdog exiting")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
