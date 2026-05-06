#!/usr/bin/env python3
"""Download model files in parallel, verify SHA256, cancel all on failure.

Each file is downloaded using multiple connections (HTTP Range requests) for
maximum throughput, similar to aria2. Falls back to single-connection if the
server doesn't support Range.
"""

import hashlib
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODELS_DIR = Path("models")
CHUNKS_PER_FILE = 4  # Number of parallel connections per file download.

# (filename, url, sha256)
MODELS = [
    (
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3",
    ),
    (
        "tinygemma3-Q8_0.gguf",
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/tinygemma3-Q8_0.gguf",
        "7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd",
    ),
    (
        "mmproj-tinygemma3.gguf",
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf",
        "93c2ba8c34574dd8f2dfda64931fc20943de2f941bfe03e6e9eca68951b80604",
    ),
    (
        "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    ),
    (
        "bge-reranker-v2-m3-Q2_K.gguf",
        "https://modelscope.cn/models/gpustack/bge-reranker-v2-m3-GGUF/resolve/master/bge-reranker-v2-m3-Q2_K.gguf",
        "f12135b80de836cbf94c1169dc8efda57c81040c1dfd9dedc20709d2e1725e39",
    ),
    (
        "stories15M_MOE-F16.gguf",
        "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf",
        "1240dfc1957df9f3550dd6c1d9e64b466fc2f452d8bc34bd4e45e1a1e2ca6055",
    ),
    (
        "stories15M-q4_0.gguf",
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
        "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
    ),
    (
        "moe_shakespeare15M.gguf",
        "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf",
        "d1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a",
    ),
]


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest using the fastest available method."""
    with open(path, "rb") as f:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()
        h = hashlib.sha256()
        while chunk := f.read(1 << 23):  # 8 MB
            h.update(chunk)
        return h.hexdigest()


def get_file_info(url: str) -> tuple[str, int] | None:
    """Follow redirects and get (final_url, file_size) via HEAD request.

    Returns None if Range is not supported or Content-Length unknown.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            final_url = resp.url  # After redirects.
            accept_ranges = resp.headers.get("Accept-Ranges", "")
            length = resp.headers.get("Content-Length")
            if accept_ranges.lower() == "bytes" and length:
                return final_url, int(length)
    except Exception:
        pass
    return None


def download_chunk(url: str, start: int, end: int, fd: int) -> None:
    """Download a byte range and write to fd at the correct offset using pwrite.

    os.pwrite is atomic w.r.t. offset — no seek+write race between threads.
    """
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        offset = start
        while data := resp.read(1 << 20):  # 1 MB
            os.pwrite(fd, data, offset)
            offset += len(data)


def download_file(url: str, dest: Path, num_chunks: int = CHUNKS_PER_FILE) -> None:
    """Download a file using multiple parallel connections (Range requests).

    Falls back to single-connection download if the server doesn't support
    Range requests or if the file is too small to benefit from splitting.
    """
    file_info = get_file_info(url)

    # Fallback: single connection (server doesn't support Range, or tiny file).
    min_chunk_size = 1 << 20  # 1 MB — don't split files smaller than this.
    if file_info is None or file_info[1] < min_chunk_size * num_chunks:
        urllib.request.urlretrieve(url, dest)
        return

    final_url, file_size = file_info

    # Pre-allocate the file and keep a single fd open for all threads.
    fd = os.open(str(dest), os.O_CREAT | os.O_RDWR | os.O_TRUNC)
    try:
        os.ftruncate(fd, file_size)

        # Split into chunks and download in parallel.
        chunk_size = file_size // num_chunks
        ranges = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = file_size - 1 if i == num_chunks - 1 else (i + 1) * chunk_size - 1
            ranges.append((start, end))

        errors: list[Exception] = []

        def _download_range(start: int, end: int) -> None:
            try:
                download_chunk(final_url, start, end, fd)
            except Exception as e:
                errors.append(e)

        threads = []
        for start, end in ranges:
            t = threading.Thread(target=_download_range, args=(start, end))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
    finally:
        os.close(fd)

    if errors:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"chunk download failed: {errors[0]}")


def download_one(name: str, url: str, expected_sha: str) -> str:
    """Download and verify a single model file. Returns a status message."""
    dest = MODELS_DIR / name

    # If file exists, verify integrity; redownload if mismatch.
    if dest.exists():
        actual = sha256_file(dest)
        if actual == expected_sha:
            return f"✓ {name} (already exists, verified)"
        else:
            print(f"  {name}: SHA256 mismatch, redownloading...", flush=True)
            dest.unlink()

    print(f"Downloading {name} ({CHUNKS_PER_FILE} connections)...", flush=True)
    try:
        download_file(url, dest, CHUNKS_PER_FILE)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: download failed — {e}")

    actual = sha256_file(dest)
    if actual != expected_sha:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: SHA256 mismatch after download.\n"
            f"  Expected: {expected_sha}\n"
            f"  Got:      {actual}"
        )

    return f"  ✓ {name} downloaded and verified"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {
            executor.submit(download_one, name, url, sha): name
            for name, url, sha in MODELS
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                msg = future.result()
                print(msg, flush=True)
            except Exception as e:
                # Cancel all pending futures immediately.
                for f in futures:
                    f.cancel()
                print(
                    f"\n{'=' * 50}\n"
                    f"ERROR: {e}\n"
                    f"{'=' * 50}",
                    file=sys.stderr,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                return 1

    print(f"\nAll models ready in {MODELS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
