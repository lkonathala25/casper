"""LLM solver adapter: fine-tuned Qwen2.5 SFT + full-board contract -> (B,81,9) logits.

Pipeline:
  perception log-probs (B,81,10)
  -> flat boards (perception argmax)
  -> full 9x9 board prompt (SATNet SFT full-board contract)
  -> fine-tuned Qwen2.5 via Unsloth FastModel
  -> parse <SOLUTION> block -> full 9x9 board
  -> (B,81,9) logits (LLM_CONF at chosen digit, 0 elsewhere)

Contract: full-board SATNet SFT.
  - system prompt: exact SFT system prompt (matches satnet_sft_utils.py)
  - user prompt: full 9x9 grid, 0=blank
  - output: <WORKING>...</WORKING><SOLUTION>9x9 grid</SOLUTION>
  - parsing: strict line/token/range/givens validation

Cache:
  key = model_path_hash + system_prompt_hash + schema_version +
        pipeline_version + flat_board_str
  New PIPELINE_VERSION / SCHEMA_VERSION auto-invalidate old blanks-only entries.
"""
import ctypes
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import paths

# Inject libcudart.so.12 for CUDA 12-linked wheels on CUDA 13.x systems.
# Two fixes: LD_LIBRARY_PATH for subprocesses, ctypes for current process.
_CUDART12_DIR = str(paths.CUDART12_DIR)
_CUDART12_SO = os.path.join(_CUDART12_DIR, "libcudart.so.12")
if os.path.isdir(_CUDART12_DIR):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _CUDART12_DIR not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            _CUDART12_DIR + (":" + _ld if _ld else "")
        )
if os.path.exists(_CUDART12_SO):
    try:
        ctypes.CDLL(_CUDART12_SO, ctypes.RTLD_GLOBAL)
    except OSError:
        pass

# Unsloth does 'from datasets import Dataset' at import time.
# visual/datasets.py (local perception dataset file) shadows the HuggingFace
# package because visual/ is on sys.path when scripts run from that directory.
# Fix: temporarily exclude visual/ from sys.path during the Unsloth import,
# then restore everything so later `import datasets` still finds the local file.
_SRC_DIR = str(Path(__file__).resolve().parents[1])  # …/tasks/sudoku/visual
_saved_sys_path = list(sys.path)
_prior_datasets = sys.modules.pop("datasets", None)   # evict any cached local copy
sys.path = [p for p in sys.path if p not in ("", ".", _SRC_DIR)]

_UNSLOTH_AVAILABLE = False
_unsloth_import_error = None
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    _UNSLOTH_AVAILABLE = True
except Exception as _e:
    _unsloth_import_error = _e
finally:
    sys.path = _saved_sys_path
    # Stash the HuggingFace datasets module under a private key so unsloth's
    # already-resolved references keep working, then restore the original entry
    # so later `import datasets` in this process finds the local file again.
    _hf_datasets = sys.modules.pop("datasets", None)
    if _hf_datasets is not None:
        sys.modules["_hf_datasets_cached"] = _hf_datasets
    if _prior_datasets is not None:
        sys.modules["datasets"] = _prior_datasets

# Must stay byte-identical to the path the cache was built with: it is hashed
# into every cache key (see _make_cache_key and paths.LLM_SFT_PATH).
DEFAULT_FINETUNED_MODEL_PATH = paths.LLM_SFT_PATH

# Bump these whenever the prompt/parse contract changes.
PIPELINE_VERSION = "full_board_sft_v1"
SCHEMA_VERSION = "solution_block_v1"

# ─── SATNet SFT full-board contract ──────────────────────────────────────────
# Must match satnet_sft_utils.SYSTEM_PROMPT exactly.
_SYSTEM_PROMPT = (
    "You are a Sudoku solving assistant.\n\n"
    "You will be given a 9x9 Sudoku puzzle.\n"
    "The puzzle uses digits 1-9, and 0 means an empty cell.\n\n"
    "Your task is to return the complete solved 9x9 Sudoku grid such that:\n"
    "- each row contains digits 1 through 9 exactly once\n"
    "- each column contains digits 1 through 9 exactly once\n"
    "- each 3x3 subgrid contains digits 1 through 9 exactly once\n"
    "- all given non-zero cells remain unchanged\n\n"
    "You must respond using exactly this format:\n\n"
    "<WORKING>\n"
    "Solved by Sudoku constraints.\n"
    "</WORKING>\n"
    "<SOLUTION>\n"
    "[r1]\n"
    "[r2]\n"
    "[r3]\n"
    "[r4]\n"
    "[r5]\n"
    "[r6]\n"
    "[r7]\n"
    "[r8]\n"
    "[r9]\n"
    "</SOLUTION>\n\n"
    "Rules for the solution block:\n"
    "- output exactly 9 lines\n"
    "- each line must contain exactly 9 integers separated by single spaces\n"
    "- use only integers 1 through 9 in the solution block\n"
    "- do not output extra text before or after the tags"
)
_SYSTEM_PROMPT_HASH = hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()[:16]

_SOLUTION_RE = re.compile(r"<SOLUTION>\s*(.*?)\s*</SOLUTION>", re.DOTALL)

LLM_CONF = 10.0  # logit value placed at the LLM-predicted digit index


# ─── Prompt helpers ───────────────────────────────────────────────────────────

def _row_to_text(row_digits: list) -> str:
    return " ".join(str(x) for x in row_digits)


def _build_user_prompt(flat_board: list) -> str:
    rows = [_row_to_text(flat_board[i * 9: (i + 1) * 9]) for i in range(9)]
    return "Solve this 9x9 Sudoku puzzle.\n\nGrid:\n" + "\n".join(rows)


def _parse_solution_block(text: str, flat_board: list) -> list | None:
    """Parse <SOLUTION> block; return flat 81-cell list or None if format invalid.

    Only checks syntax (9 lines × 9 integers in 1–9), matching the standalone
    satnet_sft_utils.parse_solution_block contract. Givens preservation is a
    separate metric, not a parse-failure condition.
    """
    match = _SOLUTION_RE.search(text)
    if not match:
        return None

    block = match.group(1).strip()
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if len(lines) != 9:
        return None

    grid: list[list[int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 9:
            return None
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if not all(1 <= n <= 9 for n in nums):
            return None
        grid.append(nums)

    return [grid[i][j] for i in range(9) for j in range(9)]


# ─── Adapter ──────────────────────────────────────────────────────────────────

class LLMSolverAdapter:
    """Fine-tuned Qwen2.5 solver: full-board SFT contract -> (B,81,9) logits.

    Public interface (unchanged from previous blanks-only version):
        solve_batch(perception_logprobs: Tensor[B,81,10]) -> Tensor[B,81,9]
        reset_counters()
        log_counters(prefix="")
        close()

    Args:
        model_path: directory of the fine-tuned LoRA checkpoint.
        cache_dir: directory for persistent disk cache (strongly recommended).
        max_new_tokens: generation budget (must cover full tagged output ~150+ tokens).
        max_seq_length: max sequence length for model loading.
        generation_batch_size: boards per model.generate() call.
        log_raw_first: print the first raw output for diagnostics.
        prompt_path: ignored; kept for CLI compatibility.
        llm_batch_size: ignored; kept for CLI compatibility.
        batch_size_cap: ignored; kept for CLI compatibility.
        model_id: ignored; kept for CLI compatibility with old vLLM arg.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FINETUNED_MODEL_PATH,
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
        max_seq_length: int = 1024,
        generation_batch_size: int = 4,
        log_raw_first: bool = True,
        # Kept for CLI compat with old vLLM-based adapter
        prompt_path=None,
        llm_batch_size: int = 1,
        batch_size_cap: int = 32,
        model_id: str | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLMSolverAdapter requires a CUDA GPU. No GPU detected."
            )
        if not _UNSLOTH_AVAILABLE:
            raise ImportError(
                f"Unsloth import failed: {_unsloth_import_error}\n"
                "  pip install unsloth\n"
                "  (run in the sudoku-fine conda env, not sudoku)\n"
            )
        if model_id is not None and model_path == DEFAULT_FINETUNED_MODEL_PATH:
            print(
                f"[LLMSolverAdapter] NOTE: old model_id={model_id!r} ignored; "
                f"using model_path={model_path!r} (fine-tuned SFT checkpoint).",
                flush=True,
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.generation_batch_size = generation_batch_size
        self.cache_dir = cache_dir

        self._parse_fail_count = 0
        self._fallback_count = 0
        self._strict_fail_count = 0  # always 0; kept for interface compat
        self._cache_hits = 0
        self._cache_misses = 0
        self._log_raw_first = log_raw_first
        self._first_logged = False
        self._mem_cache: dict[str, list] = {}
        # Per-board parse success for the most recent solve_batch call.
        # True = parsed successfully (or served from cache of a valid parse).
        # False = LLM output was malformed; fallback board was used.
        self._last_solve_parse_ok: list[bool] = []

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_disk_cache()

        print(
            f"[LLMSolverAdapter] Loading fine-tuned checkpoint: {model_path}",
            flush=True,
        )
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            load_in_8bit=False,
        )
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="qwen-2.5")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        FastModel.for_inference(self.model)
        # Unsloth sets generation_config.max_length=32768; clear it so it doesn't
        # conflict with the max_new_tokens we pass to model.generate().
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        print("[LLMSolverAdapter] Fine-tuned model ready.", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_last_batch_parse_ok(self) -> list[bool]:
        """Return per-board parse success flags for the most recent solve_batch call.

        True  = LLM output was parsed successfully (or served from cache of a
                previously valid parse).
        False = LLM output was malformed; a fallback board was substituted.

        Use this in symbolic parity evaluation to distinguish genuine LLM outputs
        from fallback-recovery outputs.
        """
        return list(self._last_solve_parse_ok)

    def solve_batch(self, perception_logprobs: torch.Tensor) -> torch.Tensor:
        """Convert perception log-probs to per-cell digit logits.

        Args:
            perception_logprobs: (B, 81, 10) log-probs from perception CNN.
        Returns:
            (B, 81, 9) float32 logits on same device as input.
              Index k = digit k+1. LLM_CONF at predicted digit, 0 elsewhere.
        """
        input_device = perception_logprobs.device
        B = perception_logprobs.shape[0]

        perc_probs = perception_logprobs.exp()
        digits_batch = perc_probs.argmax(dim=2).cpu().tolist()

        result = torch.zeros(B, 81, 9, dtype=torch.float32)
        # Default True: cache hits are entries from previously successful parses.
        self._last_solve_parse_ok = [True] * B

        uncached_idx: list[int] = []
        uncached_boards: list[list] = []

        for b in range(B):
            key = self._make_cache_key(digits_batch[b])
            if key in self._mem_cache:
                self._cache_hits += 1
                result[b] = self._solution_to_logits(self._mem_cache[key])
            else:
                self._cache_misses += 1
                uncached_idx.append(b)
                uncached_boards.append(digits_batch[b])

        if uncached_boards:
            solutions_with_status = self._generate_all(uncached_boards)
            for b, flat_board, (sol, parse_ok) in zip(
                uncached_idx, uncached_boards, solutions_with_status
            ):
                self._last_solve_parse_ok[b] = parse_ok
                result[b] = self._solution_to_logits(sol)
                if parse_ok:
                    # Only cache valid LLM parses; fallback boards must not
                    # masquerade as successful generations in future runs.
                    key = self._make_cache_key(flat_board)
                    self._mem_cache[key] = sol
            if self.cache_dir:
                self._flush_disk_cache()

        return result.to(input_device)

    def solve_batch_cached_only(self, perception_logprobs: torch.Tensor) -> torch.Tensor:
        """Cache-only variant of solve_batch.

        Returns cached logits for cached boards and zero logits for any
        uncached boards (no live LLM generation is ever triggered).
        Pre-filter with get_cached_indices() to avoid uncached boards;
        if any slip through, a stderr warning is printed and zeros are
        returned for those boards so training continues without crashing.

        Zeros mean "no LLM signal": temperature_head falls back to
        solver_probs weighting for those boards.
        """
        input_device = perception_logprobs.device
        B = perception_logprobs.shape[0]
        perc_probs = perception_logprobs.exp()
        digits_batch = perc_probs.argmax(dim=2).cpu().tolist()

        result = torch.zeros(B, 81, 9, dtype=torch.float32)
        n_miss = 0
        for b in range(B):
            key = self._make_cache_key(digits_batch[b])
            if key in self._mem_cache:
                self._cache_hits += 1
                result[b] = self._solution_to_logits(self._mem_cache[key])
            else:
                self._cache_misses += 1
                n_miss += 1
                # result[b] stays zeros — no LLM signal for this board.

        if n_miss > 0:
            print(
                f"[LLMSolverAdapter.solve_batch_cached_only] WARNING: "
                f"{n_miss}/{B} boards have no cache entry; "
                f"returning zero logits for those boards. "
                f"Run minedit_llm_mask_calibrate.py to populate the cache.",
                file=sys.stderr, flush=True,
            )

        return result.to(input_device)

    def get_cached_indices(self, perception_logprobs: torch.Tensor) -> list[int]:
        """Return indices of boards in the batch that have a cache hit.

        Does NOT call the LLM — safe to use as a pre-filter before training.
        """
        perc_probs = perception_logprobs.exp()
        digits_batch = perc_probs.argmax(dim=2).cpu().tolist()
        return [i for i, fb in enumerate(digits_batch)
                if self._make_cache_key(fb) in self._mem_cache]

    def reset_counters(self):
        self._parse_fail_count = 0
        self._fallback_count = 0
        self._strict_fail_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_solve_parse_ok = []

    def log_counters(self, prefix: str = ""):
        print(
            f"{prefix}parse_fail_count={self._parse_fail_count} "
            f"strict_fail_count={self._strict_fail_count} "
            f"fallback_count={self._fallback_count} "
            f"cache_hits={self._cache_hits} "
            f"cache_misses={self._cache_misses}",
            flush=True,
        )

    def close(self):
        """Release model resources explicitly."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            del self.model
            del self.tokenizer
        finally:
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _make_cache_key(self, flat_board: list) -> str:
        model_hash = hashlib.sha256(self.model_path.encode()).hexdigest()[:16]
        board_str = ",".join(str(x) for x in flat_board)
        return (
            f"{model_hash}|{_SYSTEM_PROMPT_HASH}|"
            f"{SCHEMA_VERSION}|{PIPELINE_VERSION}|{board_str}"
        )

    def _solution_to_logits(self, solution: list) -> torch.Tensor:
        logits = torch.zeros(81, 9)
        for j, d in enumerate(solution):
            if 1 <= d <= 9:
                logits[j, d - 1] = LLM_CONF
        return logits

    def _fallback_solution(self, flat_board: list) -> list:
        # Fallback: clamp zeros to 1 (argmax will pick digit 1 for blanks).
        return [max(d, 1) for d in flat_board]

    def _load_disk_cache(self):
        cache_file = os.path.join(self.cache_dir, "llm_cache_sft_fullboard.json")
        if os.path.isfile(cache_file):
            with open(cache_file) as f:
                self._mem_cache = json.load(f)
            print(
                f"[LLMSolverAdapter] Loaded {len(self._mem_cache)} cached entries "
                f"from {cache_file}",
                flush=True,
            )

    def _flush_disk_cache(self):
        cache_file = os.path.join(self.cache_dir, "llm_cache_sft_fullboard.json")
        with open(cache_file, "w") as f:
            json.dump(self._mem_cache, f)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_messages(self, flat_board: list) -> list[dict]:
        user = _build_user_prompt(flat_board)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]

    # ------------------------------------------------------------------
    # Generation (Unsloth/HF batched)
    # ------------------------------------------------------------------

    def _generate_all(self, flat_boards: list[list]) -> list[tuple[list, bool]]:
        """Generate solutions for all boards in chunked HF batches.

        Returns list of (solution, parse_ok) pairs. parse_ok=False means the
        LLM output was malformed and a fallback board was substituted.
        """
        results: list[tuple[list, bool]] = []
        chunk_size = self.generation_batch_size
        for start in range(0, len(flat_boards), chunk_size):
            chunk = flat_boards[start: start + chunk_size]
            results.extend(self._generate_chunk(chunk))
        return results

    def _generate_chunk(self, flat_boards: list[list]) -> list[tuple[list, bool]]:
        """Run one model.generate() call for a chunk of boards.

        Returns list of (solution, parse_ok) pairs.
        """
        prompt_texts: list[str] = []
        for flat_board in flat_boards:
            messages = self._build_messages(flat_board)
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(prompt_text)

        # Left-pad batch for generation.
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results: list[tuple[list, bool]] = []
        for i, flat_board in enumerate(flat_boards):
            # Slice off the input tokens (same offset for all due to left-padding).
            generated_ids = output_ids[i][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            if self._log_raw_first and not self._first_logged:
                self._first_logged = True
                print(
                    f"[LLMSolverAdapter] First output: {repr(text[:300])}",
                    flush=True,
                )

            sol = _parse_solution_block(text, flat_board)
            if sol is not None:
                results.append((sol, True))
            else:
                self._parse_fail_count += 1
                self._fallback_count += 1
                fb = self._fallback_solution(flat_board)
                print(
                    f"[LLMSolverAdapter] WARNING: parse failure (board {i} in chunk); "
                    f"using perception-argmax fallback. text={repr(text[:120])}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append((fb, False))

        return results
